from typing import Any, Dict, Optional, Tuple

import einops
import torch
from torch import nn

from slotcontrast import modules, utils


@utils.make_build_fn(__name__, "loss")
def build(config, name: str):
    target_transform = None
    if config.get("target_transform"):
        target_transform = modules.build_module(config.get("target_transform"))

    cls = utils.get_class_by_name(__name__, name)
    if cls is not None:
        return cls(
            target_transform=target_transform,
            **utils.config_as_kwargs(config, ("target_transform",)),
        )
    else:
        raise ValueError(f"Unknown loss `{name}`")


class Loss(nn.Module):
    """Base class for loss functions.

    Args:
        video_inputs: If true, assume inputs contain a time dimension.
        patch_inputs: If true, assume inputs have a one-dimensional patch dimension. If false,
            assume inputs have height, width dimensions.
        pred_dims: Dimensions [from, to) of prediction tensor to slice. Useful if only a
            subset of the predictions should be used in the loss, i.e. because the other dimensions
            are used in other losses.
        remove_last_n_frames: Number of frames to remove from the prediction before computing the
            loss. Only valid with video inputs. Useful if the last frame does not have a
            correspoding target.
        target_transform: Transform that can optionally be applied to the target.
    """

    def __init__(
        self,
        pred_key: str,
        target_key: str,
        video_inputs: bool = False,
        patch_inputs: bool = True,
        keep_input_dim: bool = False,
        pred_dims: Optional[Tuple[int, int]] = None,
        remove_last_n_frames: int = 0,
        target_transform: Optional[nn.Module] = None,
        input_key: Optional[str] = None,
    ):
        super().__init__()
        self.pred_path = pred_key.split(".")
        self.target_path = target_key.split(".")
        self.video_inputs = video_inputs
        self.patch_inputs = patch_inputs
        self.keep_input_dim = keep_input_dim
        self.input_key = input_key
        self.n_expected_dims = (
            2 + (1 if patch_inputs or keep_input_dim else 2) + (1 if video_inputs else 0)
        )

        if pred_dims is not None:
            assert len(pred_dims) == 2
            self.pred_dims = slice(pred_dims[0], pred_dims[1])
        else:
            self.pred_dims = None

        self.remove_last_n_frames = remove_last_n_frames
        if remove_last_n_frames > 0 and not video_inputs:
            raise ValueError("`remove_last_n_frames > 0` only valid with `video_inputs==True`")

        self.target_transform = target_transform
        self.to_canonical_dims = self.get_dimension_canonicalizer()

    def get_dimension_canonicalizer(self) -> torch.nn.Module:
        """Return a module which reshapes tensor dimensions to (batch, n_positions, n_dims)."""
        if self.video_inputs:
            if self.patch_inputs:
                pattern = "B F P D -> B (F P) D"
            elif self.keep_input_dim:
                return torch.nn.Identity()
            else:
                pattern = "B F D H W -> B (F H W) D"
        else:
            if self.patch_inputs:
                return torch.nn.Identity()
            else:
                pattern = "B D H W -> B (H W) D"

        return einops.layers.torch.Rearrange(pattern)

    def get_target(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> torch.Tensor:
        target = utils.read_path(outputs, elements=self.target_path, error=False)
        if target is None:
            target = utils.read_path(inputs, elements=self.target_path)

        target = target.detach()

        if self.target_transform:
            with torch.no_grad():
                if self.input_key is not None:
                    target = self.target_transform(target, inputs[self.input_key])
                else:
                    target = self.target_transform(target)

        # Convert to dimension order (batch, positions, dims)
        target = self.to_canonical_dims(target)

        return target

    def get_prediction(self, outputs: Dict[str, Any]) -> torch.Tensor:
        prediction = utils.read_path(outputs, elements=self.pred_path)
        if prediction.ndim != self.n_expected_dims:
            raise ValueError(
                f"Prediction has {prediction.ndim} dimensions (and shape {prediction.shape}), but "
                f"expected it to have {self.n_expected_dims} dimensions."
            )

        if self.video_inputs and self.remove_last_n_frames > 0:
            prediction = prediction[:, : -self.remove_last_n_frames]

        # Convert to dimension order (batch, positions, dims)
        prediction = self.to_canonical_dims(prediction)

        if self.pred_dims:
            prediction = prediction[..., self.pred_dims]

        return prediction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implement in subclasses")


class TorchLoss(Loss):
    """Wrapper around PyTorch loss functions."""

    def __init__(
        self,
        pred_key: str,
        target_key: str,
        loss: str,
        loss_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(pred_key, target_key, **kwargs)
        loss_kwargs = loss_kwargs if loss_kwargs is not None else {}
        if hasattr(torch.nn, loss):
            self.loss_fn = getattr(torch.nn, loss)(reduction="mean", **loss_kwargs)
        else:
            raise ValueError(f"Loss function torch.nn.{loss} not found")

        # Cross entropy loss wants dimension order (batch, classes, positions)
        self.positions_last = loss == "CrossEntropyLoss"

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.positions_last:
            prediction = prediction.transpose(-2, -1)
            target = target.transpose(-2, -1)

        return self.loss_fn(prediction, target)


class MSELoss(TorchLoss):
    def __init__(self, pred_key: str, target_key: str, **kwargs):
        super().__init__(pred_key, target_key, loss="MSELoss", **kwargs)


class CrossEntropyLoss(TorchLoss):
    def __init__(self, pred_key: str, target_key: str, **kwargs):
        super().__init__(pred_key, target_key, loss="CrossEntropyLoss", **kwargs)


class Slot_Slot_Contrastive_Loss(Loss):
    def __init__(
        self,
        pred_key: str,
        target_key: str,
        temperature: float = 0.1,
        batch_contrast: bool = True,
        mask_key: str = "existence_mask",
        **kwargs,
    ):
        super().__init__(pred_key, target_key, **kwargs)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.temperature = temperature
        self.batch_contrast = batch_contrast
        self.mask_key = mask_key

    def forward(self, slots, _, existence_mask=None):
        """
        Args:
            slots: [B, T, K, D] slot representations
            _: unused target
            existence_mask: [B, T, K] or [B, K] - 1.0 for valid slots, 0.0 for empty
        """
        slots = nn.functional.normalize(slots, p=2.0, dim=-1)
        
        # Handle existence mask
        if existence_mask is not None:
            # Expand [B, K] to [B, T, K] if needed
            if existence_mask.ndim == 2 and slots.ndim == 4:
                existence_mask = existence_mask.unsqueeze(1).expand(-1, slots.shape[1], -1)
        
        if self.batch_contrast:
            slots_list = slots.split(1)  # [1xTxKxD]
            slots = torch.cat(slots_list, dim=-2)  # 1xTxK*BxD
            if existence_mask is not None:
                mask_list = existence_mask.split(1)  # [1xTxK]
                existence_mask = torch.cat(mask_list, dim=-1)  # 1xTxK*B
        
        s1 = slots[:, :-1, :, :]  # [B, T-1, S, D]
        s2 = slots[:, 1:, :, :]   # [B, T-1, S, D]
        ss = torch.matmul(s1, s2.transpose(-2, -1)) / self.temperature  # [B, T-1, S, S]
        B, T, S, _ = ss.shape
        ss = ss.reshape(B * T, S, S)
        target = torch.eye(S).expand(B * T, S, S).to(ss.device)
        
        # Compute per-element loss
        loss_per_slot = self.criterion(ss, target)  # [B*T, S, S] -> [B*T, S]
        
        if existence_mask is not None:
            # Build valid pair mask: both slots at t and t+1 must exist
            m1 = existence_mask[:, :-1, :]  # [B, T-1, S]
            m2 = existence_mask[:, 1:, :]   # [B, T-1, S]
            # For slot i, it's valid if slot i exists at both t and t+1
            # The loss matrix ss[i,j] compares slot i at t with slot j at t+1
            # For diagonal (target), we need both slot i to exist at t and t+1
            valid_mask = (m1 * m2).reshape(B * T, S)  # [B*T, S]
            
            # Mask invalid slots (average only over valid)
            valid_count = valid_mask.sum()
            if valid_count > 0:
                loss = (loss_per_slot * valid_mask).sum() / valid_count
            else:
                loss = loss_per_slot.mean()  # Fallback if no valid pairs
        else:
            loss = loss_per_slot.mean()
        
        return loss


class DynamicsLoss(Loss):
    def __init__(self, pred_key: str, target_key: str, **kwargs):
        super().__init__(pred_key, target_key, **kwargs)
        self.criterion = nn.MSELoss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        rollout_length = prediction.shape[1]
        target = target[:, -rollout_length:]
        loss = self.criterion(prediction, target)
        return loss


class EntropyLoss(Loss):
    def __init__(self, pred_key: str, target_key: str = "dummy", **kwargs):
        kwargs.pop('video_inputs', None)
        kwargs.pop('patch_inputs', None)
        kwargs.pop('keep_input_dim', None)
        super().__init__(pred_key, target_key, video_inputs=False, patch_inputs=False, keep_input_dim=True, **kwargs)
    
    def get_prediction(self, outputs: Dict[str, Any]) -> torch.Tensor:
        return utils.read_path(outputs, elements=self.pred_path)
    
    def get_target(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> torch.Tensor:
        return None
    
    def forward(self, entropy_loss: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return entropy_loss


class CycleConsistencyLoss(Loss):
    """Cycle/Temporal Cross-Consistency Loss for re-slotting.
    
    When window=0: Same-frame cycle consistency
    When window>0: Temporal cross-consistency with random sampling
    
    Computes MSE between cycle slots and target slots (detached).
    """
    def __init__(
        self,
        pred_key: str = "processor.cycle_slots",
        target_key: str = "processor.cycle_targets",
        **kwargs,
    ):
        kwargs.pop('video_inputs', None)
        kwargs.pop('patch_inputs', None)
        kwargs.pop('keep_input_dim', None)
        super().__init__(pred_key, target_key, video_inputs=False, patch_inputs=False, keep_input_dim=True, **kwargs)
        self.criterion = nn.MSELoss()
    
    def get_prediction(self, outputs: Dict[str, Any]) -> torch.Tensor:
        return utils.read_path(outputs, elements=self.pred_path)
    
    def get_target(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> torch.Tensor:
        return utils.read_path(outputs, elements=self.target_path)
    
    def forward(self, cycle_slots: torch.Tensor, target_slots: torch.Tensor) -> torch.Tensor:
        return self.criterion(cycle_slots, target_slots)
