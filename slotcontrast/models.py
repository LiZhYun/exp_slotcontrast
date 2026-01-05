from copy import copy, deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn
from torchvision.utils import make_grid

from slotcontrast import configuration, losses, modules, optimizers, utils, visualizations
from slotcontrast.data.transforms import Denormalize


def build(
    model_config: configuration.ModelConfig,
    optimizer_config,
    train_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
    val_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
):
    optimizer_builder = optimizers.OptimizerBuilder(**optimizer_config)

    initializer = modules.build_initializer(model_config.initializer)
    encoder = modules.build_encoder(model_config.encoder, "FrameEncoder")
    grouper = modules.build_grouper(model_config.grouper)
    decoder = modules.build_decoder(model_config.decoder)

    target_encoder = None
    if model_config.target_encoder:
        target_encoder = modules.build_encoder(model_config.target_encoder, "FrameEncoder")
        assert (
            model_config.target_encoder_input is not None
        ), "Please specify `target_encoder_input`."

    dynamics_predictor = None
    if model_config.dynamics_predictor:
        dynamics_predictor = modules.build_dynamics_predictor(model_config.dynamics_predictor)

    input_type = model_config.get("input_type", "image")
    if input_type == "image":
        processor = modules.LatentProcessor(grouper, predictor=None)
    elif input_type == "video":
        encoder = modules.MapOverTime(encoder)
        decoder = modules.MapOverTime(decoder)
        if target_encoder:
            target_encoder = modules.MapOverTime(target_encoder)
        if model_config.predictor is not None:
            predictor = modules.build_module(model_config.predictor)
        else:
            predictor = None
        
        # Build memory components if specified
        memory_encoder = None
        memory_bank = None
        if model_config.latent_processor:
            latent_proc_config = model_config.latent_processor
            if hasattr(latent_proc_config, "memory_encoder") and latent_proc_config.memory_encoder:
                memory_encoder = modules.build_memory_encoder(latent_proc_config.memory_encoder)
            if hasattr(latent_proc_config, "memory_bank") and latent_proc_config.memory_bank:
                memory_bank = modules.build_memory_bank(latent_proc_config.memory_bank)
            
            # Create filtered config without memory components (they're passed as kwargs)
            filtered_config = {k: v for k, v in latent_proc_config.items() 
                             if k not in ("memory_encoder", "memory_bank")}
            
            processor = modules.build_video(
                filtered_config,
                "LatentProcessor",
                corrector=grouper,
                predictor=predictor,
                memory_encoder=memory_encoder,
                memory_bank=memory_bank,
            )
        else:
            processor = modules.LatentProcessor(grouper, predictor)
        processor = modules.ScanOverTime(processor)
    else:
        raise ValueError(f"Unknown input type {input_type}")

    target_type = model_config.get("target_type", "features")
    if target_type == "input":
        default_target_key = input_type
    elif target_type == "features":
        if model_config.target_encoder_input is not None:
            default_target_key = "target_encoder.backbone_features"
        else:
            default_target_key = "encoder.backbone_features"
    else:
        raise ValueError(f"Unknown target type {target_type}. Should be `input` or `features`.")

    loss_defaults = {
        "pred_key": "decoder.reconstruction",
        "target_key": default_target_key,
        "video_inputs": input_type == "video",
        "patch_inputs": target_type == "features",
    }
    if model_config.losses is None:
        loss_fns = {"mse": losses.build(dict(**loss_defaults, name="MSELoss"))}
    else:
        loss_fns = {
            name: losses.build({**loss_defaults, **loss_config})
            for name, loss_config in model_config.losses.items()
        }

    if model_config.mask_resizers:
        mask_resizers = {
            name: modules.build_utils(resizer_config, "Resizer")
            for name, resizer_config in model_config.mask_resizers.items()
        }
    else:
        mask_resizers = {
            "decoder": modules.build_utils(
                {
                    "name": "Resizer",
                    # When using features as targets, assume patch-shaped outputs. With other
                    # targets, assume spatial outputs.
                    "patch_inputs": target_type == "features",
                    "video_inputs": input_type == "video",
                    "resize_mode": "bilinear",
                }
            ),
            "grouping": modules.build_utils(
                {
                    "name": "Resizer",
                    "patch_inputs": True,
                    "video_inputs": input_type == "video",
                    "resize_mode": "bilinear",
                }
            ),
        }

    if model_config.masks_to_visualize:
        masks_to_visualize = model_config.masks_to_visualize
    else:
        masks_to_visualize = "decoder"

    # Check if cycle consistency loss is enabled
    use_cycle_consistency = (
        model_config.losses is not None 
        and "loss_cycle" in model_config.losses
        # and model_config.get("loss_weights", {}).get("loss_cycle", 0.0) != 0.0
    )
    # Window for temporal cross-consistency (0 = same-frame only)
    temporal_cross_window = model_config.get("temporal_cross_window", 0)
    temporal_cross_mode = model_config.get("temporal_cross_mode", "both")

    model = ObjectCentricModel(
        optimizer_builder,
        initializer,
        encoder,
        processor,
        decoder,
        loss_fns,
        loss_weights=model_config.get("loss_weights", None),
        target_encoder=target_encoder,
        dynamics_predictor=dynamics_predictor,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        mask_resizers=mask_resizers,
        input_type=input_type,
        target_encoder_input=model_config.get("target_encoder_input", None),
        visualize=model_config.get("visualize", False),
        visualize_every_n_steps=model_config.get("visualize_every_n_steps", 1000),
        masks_to_visualize=masks_to_visualize,
        use_cycle_consistency=use_cycle_consistency,
        temporal_cross_window=temporal_cross_window,
        temporal_cross_mode=temporal_cross_mode,
        use_backbone_features=model_config.get("use_backbone_features", False),
    )

    if model_config.load_weights:
        model.load_weights_from_checkpoint(model_config.load_weights, model_config.modules_to_load)

    return model


class ObjectCentricModel(pl.LightningModule):
    def __init__(
        self,
        optimizer_builder: Callable,
        initializer: nn.Module,
        encoder: nn.Module,
        processor: nn.Module,
        decoder: nn.Module,
        loss_fns: Dict[str, losses.Loss],
        *,
        loss_weights: Optional[Dict[str, float]] = None,
        target_encoder: Optional[nn.Module] = None,
        dynamics_predictor: Optional[nn.Module] = None,
        train_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
        val_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
        mask_resizers: Optional[Dict[str, modules.Resizer]] = None,
        input_type: str = "image",
        target_encoder_input: Optional[str] = None,
        visualize: bool = False,
        visualize_every_n_steps: Optional[int] = None,
        masks_to_visualize: Union[str, List[str]] = "decoder",
        use_cycle_consistency: bool = False,
        temporal_cross_window: int = 0,
        temporal_cross_mode: str = "both",
        use_backbone_features: bool = False,
    ):
        super().__init__()
        self.optimizer_builder = optimizer_builder
        self.initializer = initializer
        self.encoder = encoder
        self.processor = processor
        self.decoder = decoder
        self.target_encoder = target_encoder
        self.dynamics_predictor = dynamics_predictor
        self.use_cycle_consistency = use_cycle_consistency
        self.temporal_cross_window = temporal_cross_window
        self.temporal_cross_mode = temporal_cross_mode
        self.use_backbone_features = use_backbone_features

        if loss_weights is not None:
            # Filter out losses that are not used
            assert (
                loss_weights.keys() == loss_fns.keys()
            ), f"Loss weight keys {loss_weights.keys()} != {loss_fns.keys()}"
            # loss_fns_filtered = {k: loss for k, loss in loss_fns.items() if loss_weights[k] != 0.0}
            # loss_weights_filtered = {
            #     k: loss for k, loss in loss_weights.items() if loss_weights[k] != 0.0
            # }
            self.loss_fns = nn.ModuleDict(loss_fns)
            self.loss_weights = loss_weights
        else:
            self.loss_fns = nn.ModuleDict(loss_fns)
            self.loss_weights = {}

        self.mask_resizers = mask_resizers if mask_resizers else {}
        self.mask_resizers["segmentation"] = modules.Resizer(
            video_inputs=input_type == "video", resize_mode="nearest-exact"
        )
        self.mask_soft_to_hard = modules.SoftToHardMask()
        self.train_metrics = torch.nn.ModuleDict(train_metrics)
        self.val_metrics = torch.nn.ModuleDict(val_metrics)

        self.visualize = visualize
        if visualize:
            assert visualize_every_n_steps is not None
        self.visualize_every_n_steps = visualize_every_n_steps
        if isinstance(masks_to_visualize, str):
            masks_to_visualize = [masks_to_visualize]
        for key in masks_to_visualize:
            if key not in ("decoder", "grouping", "dynamics_predictor"):
                raise ValueError(f"Unknown mask type {key}. Should be `decoder` or `grouping`.")
        self.mask_keys_to_visualize = [f"{key}_masks" for key in masks_to_visualize]

        if input_type == "image":
            self.input_key = "image"
            self.expected_input_dims = 4
        elif input_type == "video":
            self.input_key = "video"
            self.expected_input_dims = 5
        else:
            raise ValueError(f"Unknown input type {input_type}. Should be `image` or `video`.")

        self.target_encoder_input_key = (
            target_encoder_input if target_encoder_input else self.input_key
        )

    def configure_optimizers(self):
        modules = {
            "initializer": self.initializer,
            "encoder": self.encoder,
            "processor": self.processor,
            "decoder": self.decoder,
        }
        if self.dynamics_predictor:
            modules["dynamics_predictor"] = self.dynamics_predictor
        return self.optimizer_builder(modules)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        encoder_input = inputs[self.input_key]  # batch [x n_frames] x n_channels x height x width
        assert encoder_input.ndim == self.expected_input_dims
        batch_size = len(encoder_input)

        # Pack camera_data from individual keys if present (for 3D positional embedding)
        camera_data = None
        if "depths" in inputs and "intrinsics" in inputs and "extrinsics" in inputs:
            camera_data = {
                "depth": inputs["depths"],
                "intrinsics": inputs["intrinsics"],
                "extrinsics": inputs["extrinsics"],
            }
        encoder_output = self.encoder(encoder_input, camera_data=camera_data)
        features = encoder_output["features"]

        # Use backbone features for initialization (more stable early in training)
        if self.use_backbone_features and "backbone_features" in encoder_output:
            backbone_features = encoder_output["backbone_features"]
            init_output = self.initializer(batch_size=batch_size, features=backbone_features)
            # Get output_transform from encoder (handle MapOverTime wrapper for video)
            encoder_module = getattr(self.encoder, 'module', self.encoder)
            # Handle both single tensor and tuple output from initializer
            if isinstance(init_output, tuple):
                raw_slots, n_objects, existence_mask = init_output
                slots_initial = encoder_module.output_transform(raw_slots)
            else:
                slots_initial = encoder_module.output_transform(init_output)
                n_objects, existence_mask = None, None
        else:
            init_output = self.initializer(batch_size=batch_size, features=features)
            # Handle both single tensor and tuple output from initializer
            if isinstance(init_output, tuple):
                slots_initial, n_objects, existence_mask = init_output
            else:
                slots_initial = init_output
                n_objects, existence_mask = None, None
        
        # Pass existence_mask through processor for variable slot support
        processor_output = self.processor(slots_initial, features, existence_mask=existence_mask)
        slots = processor_output["state"]
        
        # Use processor output existence_mask if available (from memory matcher)
        out_existence_mask = processor_output.get("existence_mask", existence_mask)
        decoder_output = self.decoder(slots, existence_mask=out_existence_mask)

        outputs = {
            "batch_size": batch_size,
            "encoder": encoder_output,
            "processor": processor_output,
            "decoder": decoder_output,
        }
        
        # Add variable slot info if available (from GreedyFeatureInitV2)
        if n_objects is not None:
            outputs["n_objects"] = n_objects
        if out_existence_mask is not None:
            outputs["existence_mask"] = out_existence_mask

        # Cycle/Temporal Cross-Consistency: Re-slot the reconstructed features
        # When window=0, this is same-frame cycle consistency
        # When window>0, this includes cross-frame temporal consistency
        if self.use_cycle_consistency:
            cycle_slots, cycle_targets = self._compute_cycle_slots(
                processor_output, decoder_output, 
                window=self.temporal_cross_window,
                mode=self.temporal_cross_mode
            )
            outputs["processor"]["cycle_slots"] = cycle_slots
            outputs["processor"]["cycle_targets"] = cycle_targets

        if self.dynamics_predictor:
            outputs["dynamics_predictor"] = self.dynamics_predictor(slots)
            predicted_slots = outputs["dynamics_predictor"].get("next_state")
            decoded_predicted_slots = self.decoder(predicted_slots)
            decoded_predicted_slots = {
                f"predicted_{key}": value for key, value in decoded_predicted_slots.items()
            }
            outputs["decoder"].update(decoded_predicted_slots)

        outputs["targets"] = self.get_targets(inputs, outputs)

        return outputs

    def _compute_cycle_slots(
        self, processor_output: Dict[str, Any], decoder_output: Dict[str, Any], 
        window: int = 0, mode: str = "both"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cycle/temporal cross-consistency slots.
        
        When window=0: Same-frame cycle consistency (queries from t, features from t)
        When window>0: Temporal cross-consistency with mode:
            - "both": queries from [t-window, t+window]
            - "backward": queries from [t-window, t]
            - "forward": queries from [t, t+window]
        
        Returns both cycle slots and detached target slots.
        """
        recon_features = decoder_output["reconstruction"]  # [B, T, P, D_feat] or [B, P, D_feat]
        initial_queries = processor_output["initial_queries"]  # [B, T, K, D_slot] or [B, K, D_slot]
        real_slots = processor_output["corrector"]["slots"]  # [B, T, K, D_slot] or [B, K, D_slot]
        
        is_video = recon_features.ndim == 4
        
        if is_video:
            B, T, P, D_feat = recon_features.shape
            _, _, K, D_slot = initial_queries.shape
            
            # Ensure window size is valid
            assert 0 <= window <= T - 1, f"Window size {window} must be in range [0, {T - 1}]"
            assert mode in ("both", "backward", "forward"), f"Mode must be 'both', 'backward', or 'forward', got '{mode}'"
            
            # Transform reconstructed features to slot space
            output_transform = self.encoder.module.output_transform
            recon_flat = recon_features.flatten(0, 1)  # [B*T, P, D_feat]
            if output_transform is not None:
                recon_transformed = output_transform(recon_flat).view(B, T, P, -1)
            else:
                recon_transformed = recon_flat.view(B, T, P, -1)
            
            if window > 0:
                # Temporal cross-consistency: random sampling within window based on mode
                if mode == "both":
                    # Sample from [t-window, t+window]
                    offsets = torch.randint(-window, window + 1, (B, T), device=recon_features.device)
                elif mode == "backward":
                    # Sample from [t-window, t]
                    offsets = torch.randint(-window, 1, (B, T), device=recon_features.device)
                else:  # mode == "forward"
                    # Sample from [t, t+window]
                    offsets = torch.randint(0, window + 1, (B, T), device=recon_features.device)
                
                j_indices = torch.arange(T, device=recon_features.device).unsqueeze(0).expand(B, T)
                i_indices = (j_indices + offsets).clamp(0, T - 1)
                
                # Gather queries from time i
                i_expanded = i_indices.view(B, T, 1, 1).expand(B, T, K, D_slot)
                queries = torch.gather(initial_queries, dim=1, index=i_expanded)
            else:
                # Same-frame cycle consistency
                queries = initial_queries
            
            # Flatten and run slot attention
            queries_flat = queries.flatten(0, 1)
            features_flat = recon_transformed.flatten(0, 1)
            
            corrector = self.processor.module.corrector
            cycle_output = corrector(queries_flat, features_flat)
            cycle_slots = cycle_output["slots"].view(B, T, K, D_slot)
            target_slots = real_slots.detach()
        else:
            # Image case: always same-frame
            output_transform = self.encoder.output_transform
            if output_transform is not None:
                recon_features = output_transform(recon_features)
            
            corrector = self.processor.corrector
            cycle_output = corrector(initial_queries, recon_features)
            cycle_slots = cycle_output["slots"]
            target_slots = real_slots.detach()
        
        return cycle_slots, target_slots

    def process_masks(
        self,
        masks: torch.Tensor,
        inputs: Dict[str, Any],
        resizer: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Handle None or list of Nones (from merge_dict_trees when skip_corrector=True)
        if masks is None or (isinstance(masks, list) and all(m is None for m in masks)):
            return None, None, None

        if resizer is None:
            masks_for_vis = masks
            masks_for_vis_hard = self.mask_soft_to_hard(masks)
            masks_for_metrics_hard = masks_for_vis_hard
        else:
            masks_for_vis = resizer(masks, inputs[self.input_key])
            masks_for_vis_hard = self.mask_soft_to_hard(masks_for_vis)
            target_masks = inputs.get("segmentations")
            if target_masks is not None and masks_for_vis.shape[-2:] != target_masks.shape[-2:]:
                masks_for_metrics = resizer(masks, target_masks)
                masks_for_metrics_hard = self.mask_soft_to_hard(masks_for_metrics)
            else:
                masks_for_metrics_hard = masks_for_vis_hard

        return masks_for_vis, masks_for_vis_hard, masks_for_metrics_hard

    @torch.no_grad()
    def aux_forward(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Compute auxilliary outputs only needed for metrics and visualisations."""
        decoder_masks = outputs["decoder"].get("masks")
        decoder_masks, decoder_masks_hard, decoder_masks_metrics_hard = self.process_masks(
            decoder_masks, inputs, self.mask_resizers.get("decoder")
        )

        grouping_masks = outputs["processor"]["corrector"].get("masks")
        grouping_masks, grouping_masks_hard, grouping_masks_metrics_hard = self.process_masks(
            grouping_masks, inputs, self.mask_resizers.get("grouping")
        )

        aux_outputs = {}
        if decoder_masks is not None:
            aux_outputs["decoder_masks"] = decoder_masks
        if decoder_masks_hard is not None:
            aux_outputs["decoder_masks_vis_hard"] = decoder_masks_hard
        if decoder_masks_metrics_hard is not None:
            aux_outputs["decoder_masks_hard"] = decoder_masks_metrics_hard
        if grouping_masks is not None:
            aux_outputs["grouping_masks"] = grouping_masks
        if grouping_masks_hard is not None:
            aux_outputs["grouping_masks_vis_hard"] = grouping_masks_hard
        if grouping_masks_metrics_hard is not None:
            aux_outputs["grouping_masks_hard"] = grouping_masks_metrics_hard

        if self.dynamics_predictor:
            dynamics_predictor_masks = outputs["decoder"].get("predicted_masks")
            (
                dynamics_predictor_masks,
                dynamics_predictor_masks_hard,
                dynamics_predictor_masks_metrics_hard,
            ) = self.process_masks(
                dynamics_predictor_masks, inputs, self.mask_resizers.get("decoder")
            )
            if dynamics_predictor_masks is not None:
                aux_outputs["dynamics_predictor_masks"] = dynamics_predictor_masks
            if dynamics_predictor_masks_hard is not None:
                aux_outputs["dynamics_predictor_masks_vis_hard"] = dynamics_predictor_masks_hard
            if dynamics_predictor_masks_metrics_hard is not None:
                aux_outputs["dynamics_predictor_masks_hard"] = dynamics_predictor_masks_metrics_hard

        return aux_outputs

    def get_targets(
        self, inputs: Dict[str, Any], outputs: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        if self.target_encoder:
            target_encoder_input = inputs[self.target_encoder_input_key]
            assert target_encoder_input.ndim == self.expected_input_dims

            with torch.no_grad():
                encoder_output = self.target_encoder(target_encoder_input)

            outputs["target_encoder"] = encoder_output

        targets = {}
        for name, loss_fn in self.loss_fns.items():
            targets[name] = loss_fn.get_target(inputs, outputs)

        return targets

    def compute_loss(self, outputs: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses = {}
        existence_mask = outputs.get("existence_mask", None)
        
        for name, loss_fn in self.loss_fns.items():
            prediction = loss_fn.get_prediction(outputs)
            target = outputs["targets"][name]
            
            # Pass existence_mask to losses that support it
            if hasattr(loss_fn, 'mask_key') and existence_mask is not None:
                loss = loss_fn(prediction, target, existence_mask=existence_mask)
            else:
                loss = loss_fn(prediction, target)
            
            # Reduce all losses to scalars for logging
            if loss.ndim > 0:
                loss = loss.mean()
            losses[name] = loss

        losses_weighted = [loss * self.loss_weights.get(name, 1.0) for name, loss in losses.items()]
        
        total_loss = torch.stack(losses_weighted).sum()

        return total_loss, losses

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        outputs = self.forward(batch)
        if self.train_metrics or (
            self.visualize and self.trainer.global_step % self.visualize_every_n_steps == 0
        ):
            aux_outputs = self.aux_forward(batch, outputs)

        total_loss, losses = self.compute_loss(outputs)
        if len(losses) == 1:
            to_log = {"train/loss": total_loss}  # Log only total loss if only one loss configured
        else:
            to_log = {f"train/{name}": loss for name, loss in losses.items()}
            to_log["train/loss"] = total_loss

        # Log predictor analysis metrics (if available, averaged over frames)
        if "predictor_cos_sim" in outputs["processor"]:
            to_log["train/predictor_cos_sim"] = outputs["processor"]["predictor_cos_sim"].mean()
            to_log["train/predictor_rel_change"] = outputs["processor"]["predictor_rel_change"].mean()

        # Log Hungarian match indices (fraction of identity matches)
        if "hungarian_match_indices" in outputs["processor"]:
            to_log["train/hungarian_identity_ratio"] = self._compute_identity_ratio(
                outputs["processor"]["hungarian_match_indices"]
            )

        # Log n_objects (for variable slot support)
        if "n_objects" in outputs:
            to_log["train/n_objects"] = outputs["n_objects"].float().mean()

        if self.train_metrics and self.dynamics_predictor:
            prediction_batch = copy.deepcopy(batch)
            for k, v in prediction_batch.items():
                if isinstance(v, torch.Tensor) and v.dim() == 5:
                    prediction_batch[k] = v[:, self.dynamics_predictor.history_len :]

        if self.train_metrics:
            for key, metric in self.train_metrics.items():
                if "predicted" in key.lower():
                    values = metric(**prediction_batch, **outputs, **aux_outputs)
                else:
                    values = metric(**batch, **outputs, **aux_outputs)
                self._add_metric_to_log(to_log, f"train/{key}", values)
                metric.reset()
        self.log_dict(to_log, on_step=True, on_epoch=False, batch_size=outputs["batch_size"])

        del outputs  # Explicitly delete to save memory

        if (
            self.visualize
            and self.trainer.global_step % self.visualize_every_n_steps == 0
            and self.global_rank == 0
        ):
            self._log_inputs(
                batch[self.input_key],
                {key: aux_outputs[f"{key}_hard"] for key in self.mask_keys_to_visualize},
                mode="train",
            )
            self._log_masks(aux_outputs, self.mask_keys_to_visualize, mode="train")

        return total_loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        if "batch_padding_mask" in batch:
            batch = self._remove_padding(batch, batch["batch_padding_mask"])
            if batch is None:
                return

        outputs = self.forward(batch)
        aux_outputs = self.aux_forward(batch, outputs)

        total_loss, losses = self.compute_loss(outputs)
        if len(losses) == 1:
            to_log = {"val/loss": total_loss}  # Log only total loss if only one loss configured
        else:
            to_log = {f"val/{name}": loss for name, loss in losses.items()}
            to_log["val/loss"] = total_loss

        # Log predictor analysis metrics (if available, averaged over frames)
        if "predictor_cos_sim" in outputs["processor"]:
            to_log["val/predictor_cos_sim"] = outputs["processor"]["predictor_cos_sim"].mean()
            to_log["val/predictor_rel_change"] = outputs["processor"]["predictor_rel_change"].mean()

        # Log Hungarian match indices (fraction of identity matches)
        if "hungarian_match_indices" in outputs["processor"]:
            to_log["val/hungarian_identity_ratio"] = self._compute_identity_ratio(
                outputs["processor"]["hungarian_match_indices"]
            )

        # Log n_objects (for variable slot support)
        if "n_objects" in outputs:
            to_log["val/n_objects"] = outputs["n_objects"].float().mean()

        if self.dynamics_predictor:
            prediction_batch = deepcopy(batch)
            for k, v in prediction_batch.items():
                if isinstance(v, torch.Tensor) and v.dim() == 5:
                    prediction_batch[k] = v[:, self.dynamics_predictor.history_len :]

        if self.val_metrics:
            for key, metric in self.val_metrics.items():
                if "predicted" in key.lower():
                    metric.update(**prediction_batch, **outputs, **aux_outputs)
                else:
                    metric.update(**batch, **outputs, **aux_outputs)

        self.log_dict(
            to_log, on_step=False, on_epoch=True, batch_size=outputs["batch_size"], prog_bar=True
        )

        if self.visualize and batch_idx == 0 and self.global_rank == 0:
            masks_to_vis = {
                key: aux_outputs[f"{key}_vis_hard"] for key in self.mask_keys_to_visualize
            }
            if batch["segmentations"].shape[-2:] != batch[self.input_key].shape[-2:]:
                masks_to_vis["segmentations"] = self.mask_resizers["segmentation"](
                    batch["segmentations"], batch[self.input_key]
                )
            else:
                masks_to_vis["segmentations"] = batch["segmentations"]
            self._log_inputs(
                batch[self.input_key],
                masks_to_vis,
                mode="val",
            )
            self._log_masks(aux_outputs, self.mask_keys_to_visualize, mode="val")

    def validation_epoch_end(self, outputs):
        if self.val_metrics:
            to_log = {}
            for key, metric in self.val_metrics.items():
                self._add_metric_to_log(to_log, f"val/{key}", metric.compute())
                metric.reset()
            self.log_dict(to_log, prog_bar=True)

    @staticmethod
    def _add_metric_to_log(
        log_dict: Dict[str, Any], name: str, values: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ):
        if isinstance(values, dict):
            for k, v in values.items():
                log_dict[f"{name}/{k}"] = v
        else:
            log_dict[name] = values

    @staticmethod
    def _compute_identity_ratio(match_indices_list: List[Optional[torch.Tensor]]) -> torch.Tensor:
        """Compute fraction of slots that maintain identity mapping across frames."""
        total_matches = 0
        identity_matches = 0
        for indices in match_indices_list:
            if indices is not None:  # Skip first frame (no matching)
                B, N = indices.shape
                identity = torch.arange(N, device=indices.device).unsqueeze(0).expand(B, N)
                identity_matches += (indices == identity).sum().item()
                total_matches += B * N
        if total_matches == 0:
            return torch.tensor(1.0)  # No matching happened, assume identity
        return torch.tensor(identity_matches / total_matches)

    def _log_inputs(
        self,
        inputs: torch.Tensor,
        masks_by_name: Dict[str, torch.Tensor],
        mode: str,
        step: Optional[int] = None,
    ):
        denorm = Denormalize(input_type=self.input_key)
        if step is None:
            step = self.trainer.global_step

        if self.input_key == "video":
            video = torch.stack([denorm(video) for video in inputs])
            self._log_video(f"{mode}/{self.input_key}", video, global_step=step)
            for mask_name, masks in masks_by_name.items():
                if "dynamics_predictor" in mask_name:
                    rollout_length = masks.shape[1]
                    trimmed_video = video[:, -rollout_length:]
                    video_with_masks = visualizations.mix_videos_with_masks(trimmed_video, masks)
                else:
                    video_with_masks = visualizations.mix_videos_with_masks(video, masks)
                self._log_video(
                    f"{mode}/video_with_{mask_name}",
                    video_with_masks,
                    global_step=step,
                )
        elif self.input_key == "image":
            image = denorm(inputs)
            self._log_images(f"{mode}/{self.input_key}", image, global_step=step)
            for mask_name, masks in masks_by_name.items():
                image_with_masks = visualizations.mix_images_with_masks(image, masks)
                self._log_images(
                    f"{mode}/image_with_{mask_name}",
                    image_with_masks,
                    global_step=step,
                )
        else:
            raise ValueError(f"input_type should be 'image' or 'video', but got '{self.input_key}'")

    def _log_masks(
        self,
        aux_outputs,
        mask_keys=("decoder_masks",),
        mode="val",
        types: tuple = ("frames",),
        step: Optional[int] = None,
    ):
        if step is None:
            step = self.trainer.global_step
        for mask_key in mask_keys:
            if mask_key in aux_outputs:
                masks = aux_outputs[mask_key]
                if self.input_key == "video":
                    _, f, n_obj, H, W = masks.shape
                    first_masks = masks[0].permute(1, 0, 2, 3)
                    first_masks_inverted = 1 - first_masks.reshape(n_obj, f, 1, H, W)
                    self._log_video(
                        f"{mode}/{mask_key}",
                        first_masks_inverted,
                        global_step=step,
                        n_examples=n_obj,
                        types=types,
                    )
                elif self.input_key == "image":
                    _, n_obj, H, W = masks.shape
                    first_masks_inverted = 1 - masks[0].reshape(n_obj, 1, H, W)
                    self._log_images(
                        f"{mode}/{mask_key}",
                        first_masks_inverted,
                        global_step=step,
                        n_examples=n_obj,
                    )
                else:
                    raise ValueError(
                        f"input_type should be 'image' or 'video', but got '{self.input_key}'"
                    )

    def _log_video(
        self,
        name: str,
        data: torch.Tensor,
        global_step: int,
        n_examples: int = 8,
        max_frames: int = 8,
        types: tuple = ("frames",),
    ):
        data = data[:n_examples]
        logger = self._get_tensorboard_logger()

        if logger is not None:
            if "video" in types:
                logger.experiment.add_video(f"{name}/video", data, global_step=global_step)
            if "frames" in types:
                _, num_frames, _, _, _ = data.shape
                num_frames = min(max_frames, num_frames)
                data = data[:, :num_frames]
                data = data.flatten(0, 1)
                logger.experiment.add_image(
                    f"{name}/frames", make_grid(data, nrow=num_frames), global_step=global_step
                )

    def _save_video(self, name: str, data: torch.Tensor, global_step: int):
        assert (
            data.shape[0] == 1
        ), f"Only single videos saving are supported, but shape is: {data.shape}"
        data = data.cpu().numpy()[0].transpose(0, 2, 3, 1)
        data_dir = self.save_data_dir / name
        data_dir.mkdir(parents=True, exist_ok=True)
        np.save(data_dir / f"{global_step}.npy", data)

    def _log_images(
        self,
        name: str,
        data: torch.Tensor,
        global_step: int,
        n_examples: int = 8,
    ):
        n_examples = min(n_examples, data.shape[0])
        data = data[:n_examples]
        logger = self._get_tensorboard_logger()

        if logger is not None:
            logger.experiment.add_image(
                f"{name}/images", make_grid(data, nrow=n_examples), global_step=global_step
            )

    @staticmethod
    def _remove_padding(
        batch: Dict[str, Any], padding_mask: torch.Tensor
    ) -> Optional[Dict[str, Any]]:
        if torch.all(padding_mask):
            # Batch consists only of padding
            return None

        mask = ~padding_mask
        mask_as_idxs = torch.arange(len(mask))[mask.cpu()]

        output = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                output[key] = value[mask]
            elif isinstance(value, list):
                output[key] = [value[idx] for idx in mask_as_idxs]

        return output

    def _get_tensorboard_logger(self):
        if self.loggers is not None:
            for logger in self.loggers:
                if isinstance(logger, pl.loggers.tensorboard.TensorBoardLogger):
                    return logger
        else:
            if isinstance(self.logger, pl.loggers.tensorboard.TensorBoardLogger):
                return self.logger

    def on_load_checkpoint(self, checkpoint):
        # Reset timer during loading of the checkpoint
        # as timer is used to track time from the start
        # of the current run.
        if "callbacks" in checkpoint and "Timer" in checkpoint["callbacks"]:
            checkpoint["callbacks"]["Timer"]["time_elapsed"] = {
                "train": 0.0,
                "sanity_check": 0.0,
                "validate": 0.0,
                "test": 0.0,
                "predict": 0.0,
            }

    def load_weights_from_checkpoint(
        self, checkpoint_path: str, module_mapping: Optional[Dict[str, str]] = None
    ):
        """Load weights from a checkpoint into the specified modules."""
        checkpoint = torch.load(checkpoint_path)
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        if module_mapping is None:
            module_mapping = {
                key.split(".")[0]: key.split(".")[0]
                for key in checkpoint
                if hasattr(self, key.split(".")[0])
            }

        for dest_module, source_module in module_mapping.items():
            try:
                module = utils.read_path(self, dest_module)
            except ValueError:
                raise ValueError(f"Module {dest_module} could not be retrieved from model") from None

            state_dict = {}
            for key, weights in checkpoint.items():
                if key.startswith(source_module):
                    if key != source_module:
                        key = key[len(source_module + ".") :]  # Remove prefix
                    state_dict[key] = weights
            if len(state_dict) == 0:
                raise ValueError(
                    f"No weights for module {source_module} found in checkpoint {checkpoint_path}."
                )

            module.load_state_dict(state_dict)
