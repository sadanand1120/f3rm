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
# TODO: hardcoded model config in orientany, then later after you unstash orientany, add asserts for these assumed model config in pipeline etc
# TODO: review all code here and clean up abuse of usage of no_grad() vs detach(): https://www.geeksforgeeks.org/deep-learning/difference-between-detach-and-with-torchnograd-in-pytorch/
f3rm_method = MethodSpecification(
    config=F3RMTrainerConfig(
        method_name="f3rm",
        steps_per_eval_batch=500,
        steps_per_eval_image=100000,   # HACK TODO: for rapid testing, since its too slow
        steps_per_eval_all_images=100000,  # HACK TODO: for rapid testing, since its too slow
        steps_per_save=5000,
        max_num_iterations=30000,
        mixed_precision=True,
        enable_comprehensive_seeding=False,   # TODO: fix normals training issues
        seed_deterministic_algorithms=False,
        seed_warn_only=True,  # Set to True if you encounter issues with deterministic algorithms
        seed_cublas_workspace=True,
        print_seed_info=True,
        pipeline=FeaturePipelineConfig(
            datamanager=FeatureDataManagerConfig(
                feature_type="CLIP",
                sam2_feature_type="CLIPSAM_",
                foreground_feature_type="FOREGROUND_",
                orientany_feature_type="ORIENTANY_",
                cpu_feature_cache_images=256,
                gpu_feature_cache_images=128,
                dataparser=NerfstudioDataParserConfig(train_split_fraction=0.95),
                train_num_rays_per_batch=1 << 13,
                train_num_images_to_sample_from=64,
                train_num_times_to_repeat_images=2048,
                eval_num_rays_per_batch=1 << 12,
                eval_num_images_to_sample_from=64,
                eval_num_times_to_repeat_images=2048,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3",
                    optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=0.0, max_norm=1.0),
                    scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-5, warmup_steps=3000, max_steps=30000),
                ),
            ),
            model=FeatureFieldModelConfig(
                eval_num_rays_per_chunk=1 << 14,
                predict_normals=True,
                feat_condition_on_density=False,  # degraded performance
                feat_condition_density_grad_to_nerf=False,   # degraded performance
                num_proposal_iterations=2,  # May reduce proposal iterations for speed
                centroid_enable=True,
                centroid_loss_weight=1e-3,
                centroid_condition_on_density=False,
                centroid_condition_density_grad_to_nerf=False,
                centroid_hidden_dim=64,
                centroid_num_layers=2,
                centroid_gt_blend=0.5,
                centroid_blend_after_steps=10000,
                foreground_enable=True,
                foreground_condition_on_density=False,
                foreground_condition_density_grad_to_nerf=False,
                enable_campose_refine_feature_field=False,
                foreground_loss_weight=1e-3,
                foreground_hidden_dim=64,
                foreground_num_layers=2,
                orientany_enable=True,
                orientany_condition_on_density=False,
                orientany_condition_density_grad_to_nerf=False,
                orientany_loss_weight=1e-4,
                orientany_hidden_dim=64,
                orientany_num_layers=2,
                orientany_use_xyz_encoding=False,  # xyz encoding (True) or encoded centroid prediction (False)
            ),
            steps_per_train_cache_update=0,
            train_cache_cold_start_skip_steps=3000,
            steps_per_train_image_viz=8000,
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, max_norm=1.0),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, warmup_steps=1000, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, max_norm=1.0),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, warmup_steps=1000, max_steps=200000),
            },
            "feature_field": {
                "optimizer": AdamOptimizerConfig(lr=5e-3, eps=1e-15, max_norm=1.0),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=6e-5, warmup_steps=1000, max_steps=28000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="F3RM with parallel NeRF training, feature field distillation, and comprehensive seeding for reproducibility.",
)
