from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import LocalWriterConfig, LoggingConfig, ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig

from f3rm.feature_datamanager import FeatureDataManagerConfig
from f3rm.model import FeatureFieldModelConfig
from f3rm.trainer import F3RMTrainerConfig
from f3rm.pipeline import FeaturePipelineConfig
from f3rm.train_schedule import derive_train_schedule, env_float, env_int, env_int_tuple
from nerfstudio.plugins.types import MethodSpecification

# Transfer these knobs to another dataset and keep the same schedule density.
NUM_IMAGES_TOTAL = env_int("F3RM_NUM_IMAGES_TOTAL", 159)  # Increase => total work: up; end-to-end time: up; quality: usually up because training covers more images.
TRAIN_SPLIT_FRACTION = env_float("F3RM_TRAIN_SPLIT_FRACTION", 0.95)  # Increase => total work: up; end-to-end time: up; quality: usually up because more images move into train.
TRAIN_IMAGE_WH = env_int_tuple("F3RM_TRAIN_IMAGE_WH", (1050, 1904))  # Increase => total work: up; end-to-end time: up; quality: usually up because each train image has more pixels to visit.
TRAIN_NUM_RAYS_PER_BATCH = env_int("F3RM_TRAIN_NUM_RAYS_PER_BATCH", 1 << 16)  # Increase => total work: about flat; end-to-end time: usually down until GPU saturation, then can go up; quality: often near-flat, but too high can hurt.
TRAIN_NUM_IMAGES_TO_SAMPLE_FROM = env_int("F3RM_TRAIN_NUM_IMAGES_TO_SAMPLE_FROM", 32)  # Increase => total work: about flat; end-to-end time: usually up from wider window/cache churn; quality: usually up because each refresh sees more images.
PIXEL_VISITATION = env_float("F3RM_PIXEL_VISITATION", 0.4)  # Increase => total work: up directly; end-to-end time: up directly; quality: usually up because pixels are revisited more.
WINDOW_COVERAGE = env_float("F3RM_WINDOW_COVERAGE", 2.0)  # Increase => total work: about flat; end-to-end time: usually up from more window refreshes; quality: usually up because training touches more images.
GPU_FEATURE_CACHE_IMAGES = env_int("F3RM_GPU_FEATURE_CACHE_IMAGES", 8)  # Increase => total work: unchanged; end-to-end time: can go down or up depending on cache-hit gains vs VRAM pressure; quality: unchanged.

TRAIN_SCHEDULE = derive_train_schedule(
    num_images_total=NUM_IMAGES_TOTAL,
    train_split_fraction=TRAIN_SPLIT_FRACTION,
    train_image_wh=TRAIN_IMAGE_WH,
    train_num_rays_per_batch=TRAIN_NUM_RAYS_PER_BATCH,
    train_num_images_to_sample_from=TRAIN_NUM_IMAGES_TO_SAMPLE_FROM,
    pixel_visitation=PIXEL_VISITATION,
    window_coverage=WINDOW_COVERAGE,
)

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
        logging=LoggingConfig(steps_per_log=10, local_writer=LocalWriterConfig(enable=True), profiler="none"),
        steps_per_eval_batch=0,
        steps_per_eval_image=0,
        steps_per_eval_all_images=TRAIN_SCHEDULE.steps_per_eval_all_images,  # Keep the only eval-all pass at the final step.
        save_only_latest_checkpoint=True,
        steps_per_save=0,
        max_num_iterations=TRAIN_SCHEDULE.max_num_iterations,
        mixed_precision=True,
        use_grad_scaler=True,
        pipeline=FeaturePipelineConfig(
            datamanager=FeatureDataManagerConfig(
                feature_type="CLIP",
                images_on_gpu=True,
                pin_cpu_feature_cache=False,
                cpu_feature_cache_images=32,
                gpu_feature_cache_images=GPU_FEATURE_CACHE_IMAGES,
                dataparser=NerfstudioDataParserConfig(train_split_fraction=TRAIN_SPLIT_FRACTION),
                train_num_rays_per_batch=TRAIN_NUM_RAYS_PER_BATCH,
                train_num_images_to_sample_from=TRAIN_NUM_IMAGES_TO_SAMPLE_FROM,
                train_num_times_to_repeat_images=TRAIN_SCHEDULE.train_num_times_to_repeat_images,
                eval_num_rays_per_batch=1 << 12,
                eval_num_images_to_sample_from=TRAIN_NUM_IMAGES_TO_SAMPLE_FROM,
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
                num_nerf_samples_per_ray=19,
                distortion_loss_mult=0.0030,
                feat_loss_weight=1e-3,
                feat_train_ray_ratio=0.125,
                feat_use_pe=False,
                feat_num_levels=10,
                feat_features_per_level=4,
                feat_hidden_dim=64,
                feat_num_layers=2,
            ),
            steps_per_train_image_viz=0,
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, max_norm=1.0),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, warmup_steps=600, max_steps=TRAIN_SCHEDULE.max_num_iterations
                ),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, max_norm=1.0),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, warmup_steps=600, max_steps=TRAIN_SCHEDULE.max_num_iterations
                ),
            },
            "feature_field": {
                "optimizer": AdamOptimizerConfig(lr=5e-2, eps=1e-15, max_norm=1.0),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-3, warmup_steps=600, max_steps=TRAIN_SCHEDULE.max_num_iterations
                ),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-8, weight_decay=0.0, max_norm=0.5),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-5, warmup_steps=1800, max_steps=TRAIN_SCHEDULE.max_num_iterations
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="F3RM with parallel NeRF training and feature field distillation.",
)
