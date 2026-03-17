from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig

from f3rm.feature_datamanager import FeatureDataManagerConfig
from f3rm.model import FeatureFieldModelConfig
from f3rm.trainer import F3RMTrainerConfig
from f3rm.pipeline import FeaturePipelineConfig
from nerfstudio.plugins.types import MethodSpecification

# TODO: Look at https://docs.nerf.studio/nerfology/methods/nerfacto.html, try bigger model for better scenes!
# TODO: (maybe) revisit PCA implementation for feature visualization (previous speed tests were similar)
# TODO: optimize code by calling super().bla at places (e.g. super().get_train_loss_dict() in pipeline.py)
# TODO: reduce training time, look at original feature loading (.pt based) in f3rm, maybe thats the issue?
# TODO: do model compression so training time is reduced as well, instead of having separate entire MLPs, just have a larger common trunk where possible, and have separate output heads
# TODO: hardcoded model config in orientany, then later after you unstash orientany, add asserts for these assumed model config in pipeline etc
# TODO: review all code here and clean up abuse of usage of no_grad() vs detach(): https://www.geeksforgeeks.org/deep-learning/difference-between-detach-and-with-torchnograd-in-pytorch/
f3rm_method = MethodSpecification(
    config=F3RMTrainerConfig(
        method_name="f3rm",
        steps_per_eval_batch=500,
        steps_per_eval_image=6169,   # Match baseline eval-image residue while staying near run end.
        steps_per_eval_all_images=6171,  # Keep one near-end eval-all pass near the final iterations.
        save_only_latest_checkpoint=True,
        steps_per_save=3700,
        max_num_iterations=6176,
        mixed_precision=True,
        use_grad_scaler=True,
        pipeline=FeaturePipelineConfig(
            datamanager=FeatureDataManagerConfig(
                feature_type="CLIP",
                foreground_feature_type="FOREGROUND_",
                images_on_gpu=True,
                pin_cpu_feature_cache=False,
                cpu_feature_cache_images=32,
                gpu_feature_cache_images=0,
                dataparser=NerfstudioDataParserConfig(train_split_fraction=0.95),
                train_num_rays_per_batch=10_752,
                train_num_images_to_sample_from=32,
                train_num_times_to_repeat_images=392,
                eval_num_rays_per_batch=1 << 12,
                eval_num_images_to_sample_from=32,
                eval_num_times_to_repeat_images=512,
            ),
            model=FeatureFieldModelConfig(
                camera_optimizer=CameraOptimizerConfig(mode="off"),  # "SO3xR3" or "off"
                implementation="tcnn",  # other option is "torch"
                eval_num_rays_per_chunk=1 << 16,
                predict_normals=False,
                use_appearance_embedding=True,
                num_proposal_iterations=1,  # May reduce proposal iterations for speed
                num_proposal_samples_per_ray=(96,),
                num_nerf_samples_per_ray=20,
                distortion_loss_mult=0.0030,
                feat_loss_weight=1e-3,
                feat_train_ray_ratio=0.125,
                feat_use_pe=False,
                feat_num_levels=10,
                feat_features_per_level=4,
                feat_hidden_dim=64,
                feat_num_layers=2,
                foreground_loss_weight=1e-3,
                foreground_train_ray_ratio=0.125,
                foreground_hidden_dim=64,
                foreground_num_layers=2,
            ),
            steps_per_train_image_viz=0,
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, max_norm=1.0),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, warmup_steps=772, max_steps=6176),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, max_norm=1.0),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, warmup_steps=772, max_steps=6176),
            },
            "feature_field": {
                "optimizer": AdamOptimizerConfig(lr=5e-2, eps=1e-15, max_norm=1.0),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-3, warmup_steps=772, max_steps=6176),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-8, weight_decay=0.0, max_norm=0.5),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, warmup_steps=2316, max_steps=6176),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="F3RM with parallel NeRF training and feature field distillation.",
)
