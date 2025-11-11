from typing import Any, Dict, List, Mapping, Optional

import torch
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
    ):
        super().__init__()
        self.corrector = corrector
        self.predictor = predictor
        self.memory_encoder = memory_encoder
        self.memory_bank = memory_bank
        self.state_key = state_key
        if first_step_corrector_args is not None:
            self.first_step_corrector_args = first_step_corrector_args
        else:
            self.first_step_corrector_args = None

    def forward(
        self, state: torch.Tensor, inputs: Optional[torch.Tensor], time_step: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        # state: batch x n_slots x slot_dim
        assert state.ndim == 3
        # inputs: batch x n_patches x input_dim (encoder features)
        assert inputs.ndim == 3

        # 1. CORRECT: Update slots based on current frame features
        if inputs is not None:
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
        if self.predictor:
            # Check if predictor supports memory
            if (
                hasattr(self.predictor, "use_memory")
                and getattr(self.predictor, "use_memory", False)
                and memory is not None
            ):
                # Predict with memory conditioning
                predicted_state = self.predictor(updated_state, memory, memory_pos)
            else:
                # Standard prediction without memory
                predicted_state = self.predictor(updated_state)
        else:
            # Just pass updated_state along as prediction
            predicted_state = updated_state

        return {
            "state": updated_state,
            "state_predicted": predicted_state,
            "corrector": corrector_output,
        }


class MapOverTime(nn.Module):
    """Wrapper applying wrapped module independently to each time step.

    Assumes batch is first dimension, time is second dimension.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args):
        batch_size = None
        seq_len = None
        flattened_args = []
        for idx, arg in enumerate(args):
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

        outputs = self.module(*flattened_args)

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
        # initial_state: batch x ...
        # inputs: batch x n_frames x ...
        seq_len = inputs.shape[1]

        # Clear memory bank at start of sequence
        if hasattr(self.module, "memory_bank") and self.module.memory_bank is not None:
            self.module.memory_bank.clear()

        state = initial_state
        outputs = []
        for t in range(seq_len):
            if self.pass_step:
                output = self.module(state, inputs[:, t], t)
            else:
                output = self.module(state, inputs[:, t])
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
