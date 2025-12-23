from typing import Any, Dict, List, Mapping, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from slotcontrast.utils import make_build_fn


@make_build_fn(__name__, "video module")
def build(config, name: str, **kwargs):
    pass  # No special module building needed


class LatentProcessor(nn.Module):
    """Updates latent state based on inputs and state and predicts next state."""

    def __init__(
        self,
        corrector: nn.Module,
        predictor: Optional[nn.Module] = None,
        memory_encoder: Optional[nn.Module] = None,
        memory_bank: Optional[nn.Module] = None,
        state_key: str = "slots",
        first_step_corrector_args: Optional[Dict[str, Any]] = None,
        use_ttt3r: bool = False,
        use_cycle_consistency: bool = False,
        skip_corrector: bool = False,
        skip_predictor: bool = False,
    ):
        super().__init__()
        self.corrector = corrector
        self.predictor = predictor
        self.memory_encoder = memory_encoder
        self.memory_bank = memory_bank
        self.state_key = state_key
        self.use_ttt3r = use_ttt3r
        self.use_cycle_consistency = use_cycle_consistency
        self.skip_corrector = skip_corrector
        self.skip_predictor = skip_predictor
        if first_step_corrector_args is not None:
            self.first_step_corrector_args = first_step_corrector_args
        else:
            self.first_step_corrector_args = None

    def forward(
        self, state: torch.Tensor, inputs: Optional[torch.Tensor], time_step: Optional[int] = None,
        init_state: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # state: batch x n_slots x slot_dim
        assert state.ndim == 3
        # inputs: batch x n_patches x input_dim (encoder features)
        assert inputs.ndim == 3

        # 1. CORRECT: Update slots based on current frame features
        if self.skip_corrector:
            # Bypass slot attention - use input state directly
            corrector_output = {"slots": state, "masks": None}
            updated_state = state
            corrector_masks = None
        elif inputs is not None:
            if time_step == 0 and self.first_step_corrector_args:
                corrector_output = self.corrector(state, inputs, **self.first_step_corrector_args)
            else:
                corrector_output = self.corrector(state, inputs)
            updated_state = corrector_output[self.state_key]
            corrector_masks = corrector_output.get("masks")
        else:
            # Run predictor without updating on current inputs
            corrector_output = None
            updated_state = state
            corrector_masks = None

        # 2. ENCODE MEMORY (if components exist)
        if (
            self.memory_encoder is not None
            and self.memory_bank is not None
            and time_step is not None
            and corrector_masks is not None
        ):
            # Encode current frame into memory
            memory_encoding = self.memory_encoder(
                slots=updated_state.detach(),
                features=inputs.detach(),
                masks=corrector_masks.detach(),
            )

            # Store in memory bank
            self.memory_bank.push(
                frame_idx=time_step,
                slots=updated_state.detach(),
                features=inputs.detach(),
                masks=corrector_masks.detach(),
                **memory_encoding
            )

        # 3. RETRIEVE MEMORY for prediction
        memory = None
        memory_pos = None
        if self.memory_bank is not None and time_step is not None and time_step > 0:
            memory, memory_pos = self.memory_bank.get_memories(time_step, training=self.training)

        # 4. PREDICT: Generate initialization for NEXT frame
        attn_list = None
        if self.predictor and not self.skip_predictor:
            use_memory = (
                hasattr(self.predictor, "use_memory")
                and getattr(self.predictor, "use_memory", False)
                and memory is not None
            )
            # Check if predictor supports init_state (CrossAttentionPredictor)
            use_init_state = hasattr(self.predictor, 'cross_attn') and init_state is not None
            # Check if predictor is HungarianPredictor (matching-based)
            is_hungarian = hasattr(self.predictor, '_hungarian_match')
            
            if use_memory:
                result = self.predictor(
                    updated_state, memory, memory_pos, return_weights=self.use_ttt3r
                )
            elif use_init_state:
                result = self.predictor(
                    updated_state, init_state=init_state, return_weights=self.use_ttt3r
                )
            elif is_hungarian:
                # HungarianPredictor uses internal state, just pass slots
                result = self.predictor(updated_state, return_weights=self.use_ttt3r)
            else:
                result = self.predictor(updated_state, return_weights=self.use_ttt3r)
            
            # Parse result based on return_weights
            if self.use_ttt3r and isinstance(result, tuple):
                predicted_state, attn_list = result[0], result[-1]
            else:
                predicted_state = result if not isinstance(result, tuple) else result[0]
            
            # TTT3R: Adaptive update based on attention
            if self.use_ttt3r and time_step is not None and time_step > 0 and attn_list is not None:
                predicted_state = self._apply_ttt3r_update(
                    updated_state, predicted_state, attn_list
                )
        else:
            predicted_state = updated_state

        result = {
            "state": updated_state,
            "state_predicted": predicted_state,
            "corrector": corrector_output,
            "initial_queries": state,  # Store for cycle consistency
        }

        # Predictor analysis: measure if predictor is ~identity
        if self.predictor and not self.skip_predictor:
            with torch.no_grad():
                # Cosine similarity (1.0 = identical direction)
                cos_sim = F.cosine_similarity(updated_state, predicted_state, dim=-1).mean()
                # Relative change (0.0 = identical)
                delta_norm = (predicted_state - updated_state).norm()
                input_norm = updated_state.norm() + 1e-8
                rel_change = delta_norm / input_norm
                result["predictor_cos_sim"] = cos_sim
                result["predictor_rel_change"] = rel_change

        return result

    def _apply_ttt3r_update(
        self,
        current_state: torch.Tensor,
        predicted_state: torch.Tensor,
        attn_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """Apply TTT3R adaptive update based on attention weights.
        
        Following TTT3R: use mean attention as relevance score.
        Higher attention = more relevant = larger update weight.
        """
        # attn_list: list of [B, n_slots, n_keys] from each layer
        if len(attn_list) == 0 or all(a is None for a in attn_list):
            return predicted_state
        
        # Stack and rearrange: [n_layers, B, n_slots, n_keys] -> [B, n_slots, n_keys, n_layers]
        attn_stack = torch.stack([a for a in attn_list if a is not None], dim=0)
        attn_stack = attn_stack.permute(1, 2, 3, 0)  # [B, n_slots, n_keys, n_layers]
        
        # Mean over keys and layers: [B, n_slots]
        state_relevance = attn_stack.mean(dim=(-1, -2))
        
        # Sigmoid scaling and broadcast: [B, n_slots, 1]
        update_weight = torch.sigmoid(state_relevance).unsqueeze(-1)
        
        # Adaptive blend: predicted_state * weight + current_state * (1 - weight)
        return predicted_state * update_weight + current_state * (1 - update_weight)


class MapOverTime(nn.Module):
    """Wrapper applying wrapped module independently to each time step.

    Assumes batch is first dimension, time is second dimension.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        batch_size = None
        seq_len = None
        flattened_args = []
        for idx, arg in enumerate(args):
            if arg is None:
                flattened_args.append(None)
                continue
            B, T = arg.shape[:2]
            if not batch_size:
                batch_size = B
            elif batch_size != B:
                raise ValueError(
                    f"Inconsistent batch size of {B} of argument {idx}, was {batch_size} before."
                )

            if not seq_len:
                seq_len = T
            elif seq_len != T:
                raise ValueError(
                    f"Inconsistent sequence length of {T} of argument {idx}, was {seq_len} before."
                )

            flattened_args.append(arg.flatten(0, 1))

        # Handle camera_data dict in kwargs (flatten each tensor inside)
        flattened_kwargs = {}
        for key, val in kwargs.items():
            if val is None:
                flattened_kwargs[key] = None
            elif isinstance(val, dict):
                flattened_kwargs[key] = {
                    k: v.flatten(0, 1) if isinstance(v, torch.Tensor) else v
                    for k, v in val.items()
                }
            elif isinstance(val, torch.Tensor):
                flattened_kwargs[key] = val.flatten(0, 1)
            else:
                flattened_kwargs[key] = val

        outputs = self.module(*flattened_args, **flattened_kwargs)

        if isinstance(outputs, Mapping):
            unflattened_outputs = {
                k: v.unflatten(0, (batch_size, seq_len)) for k, v in outputs.items()
            }
        else:
            unflattened_outputs = outputs.unflatten(0, (batch_size, seq_len))

        return unflattened_outputs


class ScanOverTime(nn.Module):
    """Wrapper applying wrapped module recurrently over time steps"""

    def __init__(
        self, module: nn.Module, next_state_key: str = "state_predicted", pass_step: bool = True
    ) -> None:
        super().__init__()
        self.module = module
        self.next_state_key = next_state_key
        self.pass_step = pass_step

    def forward(self, initial_state: torch.Tensor, inputs: torch.Tensor):
        # initial_state: [B, n_slots, D] or [B, T, n_slots, D] for per-frame init
        # inputs: batch x n_frames x ...
        seq_len = inputs.shape[1]
        per_frame_init = initial_state.ndim == 4  # [B, T, n_slots, D]

        # Clear memory bank at start of sequence
        if hasattr(self.module, "memory_bank") and self.module.memory_bank is not None:
            self.module.memory_bank.clear()
        
        # Reset HungarianPredictor state at start of sequence
        is_hungarian = (
            hasattr(self.module, "predictor") 
            and hasattr(self.module.predictor, "_hungarian_match")
        )
        if hasattr(self.module, "predictor") and hasattr(self.module.predictor, "reset"):
            self.module.predictor.reset()

        # Check if pre-matching mode is enabled (True or "greedy")
        use_pre_match = (
            is_hungarian
            and hasattr(self.module.predictor, "pre_match")
            and self.module.predictor.pre_match  # True or "greedy" are both truthy
        )

        state = initial_state[:, 0] if per_frame_init else initial_state
        outputs = []
        for t in range(seq_len):
            # For per-frame init with Hungarian
            if per_frame_init and t > 0:
                if use_pre_match:
                    # Pre-match: align greedy init to reference BEFORE slot attention
                    state = self.module.predictor.match_to_reference(initial_state[:, t])
                else:
                    # Post-match (original): use fresh init, Hungarian matches after slot attention
                    state = initial_state[:, t]
            
            init_state_t = initial_state[:, t] if per_frame_init else None
            if self.pass_step:
                output = self.module(state, inputs[:, t], t, init_state=init_state_t)
            else:
                output = self.module(state, inputs[:, t], init_state=init_state_t)
            outputs.append(output)
            state = output[self.next_state_key]

        return merge_dict_trees(outputs, axis=1)


def merge_dict_trees(trees: List[Mapping], axis: int = 0):
    """Stack all leafs given a list of dictionaries trees.

    Example:
    x = merge_dict_trees([
        {
            "a": torch.ones(2, 1),
            "b": {"x": torch.ones(2, 2)}
        },
        {
            "a": torch.ones(3, 1),
            "b": {"x": torch.ones(1, 2)}
        }
    ])

    x == {
        "a": torch.ones(5, 1),
        "b": {"x": torch.ones(3, 2)}
    }
    """
    out = {}
    if len(trees) > 0:
        ref_tree = trees[0]
        for key, value in ref_tree.items():
            values = [tree[key] for tree in trees]
            if isinstance(value, torch.Tensor):
                if value.ndim == 0:
                    # Handle scalar tensors - stack along new dimension (axis 0)
                    out[key] = torch.stack(values, 0)
                else:
                    out[key] = torch.stack(values, axis)
            elif isinstance(value, Mapping):
                out[key] = merge_dict_trees(values, axis)
            else:
                out[key] = values

    return out
