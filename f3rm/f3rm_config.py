from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.plugins.types import MethodSpecification

from f3rm.feature_datamanager import FeatureDataManagerConfig
from f3rm.model import FeatureFieldModelConfig
from f3rm.trainer import F3RMTrainerConfig
from f3rm.pipeline import FeaturePipelineConfig

# TODO: Look at https://docs.nerf.studio/nerfology/methods/nerfacto.html, try bigger model for better scenes!
# TODO: (maybe) replace the f3rm utils's pca with ur sam2 pca (did speed testing, both were almost same)
# TODO: optimize code by calling super().bla at places (e.g. super().get_train_loss_dict() in pipeline.py)
# TODO: reduce training time, look at original feature loading (.pt based) in f3rm, maybe thats the issue?
# TODO: do model compression so training time is reduced as well, instead of having separate entire MLPs, just have a larger common trunk where possible, and have separate output heads
f3rm_method = MethodSpecification(
    config=F3RMTrainerConfig(
        method_name="f3rm",
        steps_per_eval_batch=500,
        steps_per_save=5000,
        max_num_iterations=30000,
        mixed_precision=True,
        # Seeding configuration - now with comprehensive support
        enable_comprehensive_seeding=False,   # causing some issues with normals training, TODO: fix later
        seed_deterministic_algorithms=True,
        seed_warn_only=False,  # Set to True if you encounter issues with deterministic algorithms
        seed_cublas_workspace=True,
        print_seed_info=True,
        pipeline=FeaturePipelineConfig(
            datamanager=FeatureDataManagerConfig(
                feature_type="CLIP",
                sam2_feature_type="SAM2",
                foreground_feature_type="FOREGROUND_",
                dataparser=NerfstudioDataParserConfig(train_split_fraction=0.95),
                train_num_rays_per_batch=8192,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3",
                    optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                ),
            ),
            # To support more GPUs, we reduce the num rays per chunk. The default was 1 << 15 which uses ~16GB of GPU
            # memory when training and using viewer. 1 << 14 uses ~12GB of GPU memory in comparison. The decrease in
            # rendering speed is not too important.
            model=FeatureFieldModelConfig(
                eval_num_rays_per_chunk=1 << 14,
                predict_normals=True,
                feat_condition_on_density=False,  # degraded performance
                feat_condition_density_grad_to_nerf=False,   # degraded performance
                # Centroid head controls
                centroid_enable=False,
                centroid_loss_weight=2e-3,
                centroid_condition_on_density=False,
                centroid_condition_density_grad_to_nerf=False,
                centroid_hidden_dim=64,
                centroid_num_layers=2,
                centroid_gt_blend=0.5,
                centroid_blend_after_steps=0,
                # Foreground head controls
                foreground_enable=True,
                foreground_condition_on_density=False,
                foreground_condition_density_grad_to_nerf=False,
                enable_campose_refine_feature_field=False,
                foreground_loss_weight=2e-3,
                foreground_hidden_dim=64,
                foreground_num_layers=2,
                # spread-fg sharing trunk
                centroid_spread_trunk_fg=0,
                foreground_trunk_grad_to_spread=False,
            ),
            steps_per_train_cache_update=0,
            train_cache_cold_start_skip_steps=0,
            steps_per_train_image_viz=500,
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "feature_field": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="F3RM with parallel NeRF training, feature field distillation, and comprehensive seeding for reproducibility.",
)
