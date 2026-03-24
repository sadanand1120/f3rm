from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import LocalWriterConfig, LoggingConfig, MachineConfig, ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig

from f3rm.feature_datamanager import FeatureDataManagerConfig
from f3rm.model import FeatureFieldModelConfig
from f3rm.trainer import F3RMTrainerConfig
from f3rm.pipeline import FeaturePipelineConfig
from f3rm.train_schedule import derive_train_schedule, env_bool, env_float, env_int, env_int_tuple
from nerfstudio.plugins.types import MethodSpecification

# Modify for your dataset
NUM_IMAGES_TOTAL = env_int("F3RM_NUM_IMAGES_TOTAL", 660)  # Increase => total work: up; end-to-end time: up; quality: usually up because training covers more images.
TRAIN_IMAGE_WH = env_int_tuple("F3RM_TRAIN_IMAGE_WH", (1280, 720))  # Increase => total work: up; end-to-end time: up; quality: usually up because each train image has more pixels to visit.

TRAIN_SPLIT_FRACTION = env_float("F3RM_TRAIN_SPLIT_FRACTION", 0.95)  # Increase => total work: up; end-to-end time: up; quality: usually up because more images move into train.
NUM_DEVICES = env_int("F3RM_NUM_DEVICES", 2)  # Keep this aligned with machine.num_devices / --machine.num-devices so schedule math matches the actual distributed run.
TRAIN_NUM_RAYS_PER_BATCH = env_int("F3RM_TRAIN_NUM_RAYS_PER_BATCH", 1 << 15)  # Per-device RGB rays per train step. Increase => total work: about flat; end-to-end time: usually down until GPU saturation, then can go up; quality: often near-flat, but too high can hurt.
TRAIN_NUM_IMAGES_TO_SAMPLE_FROM = env_int("F3RM_TRAIN_NUM_IMAGES_TO_SAMPLE_FROM", 32)  # Increase => total work: about flat; end-to-end time: usually up from wider window/cache churn; quality: usually up because each refresh sees more images.
FEAT_TRAIN_RAY_RATIO = env_float("F3RM_FEAT_TRAIN_RAY_RATIO", 0.125)  # Fraction of the RGB batch that receives CLIP feature supervision.
INSTANCE_PATCH_SIZE = env_int("F3RM_INSTANCE_PATCH_SIZE", 720)  # Side length for the SAM-supervised patch branch; rays/step = patch_size^2.
NUM_INST_PATCHES = env_int("F3RM_NUM_INST_PATCHES", 2)  # Number of random SAM-supervised patches per step; total instance rays/step = num_inst_patches * patch_size^2.
# Average global visits of the RGB train pixel budget across the train pixel pool, accounting for all devices.
PIXEL_VISITATION = env_float("F3RM_PIXEL_VISITATION", 4.0)
# Number of times an image gets chosen in a sampled batch, throughout full run
WINDOW_COVERAGE = env_float("F3RM_WINDOW_COVERAGE", 4.0)  # Increase => total work: about flat; end-to-end time: usually up from more window refreshes; quality: usually up because training touches more images.
GPU_FEATURE_CACHE_IMAGES = env_int("F3RM_GPU_FEATURE_CACHE_IMAGES", 8)  # Increase => total work: unchanged; end-to-end time: can go down or up depending on cache-hit gains vs VRAM pressure; quality: unchanged.
CPU_FEATURE_CACHE_IMAGES = env_int("F3RM_CPU_FEATURE_CACHE_IMAGES", 32)  # Increase => total work: unchanged; end-to-end time: can go down by reducing rereads or up by increasing RAM pressure; quality: unchanged.
PIN_CPU_FEATURE_CACHE = env_bool("F3RM_PIN_CPU_FEATURE_CACHE", False)  # Increase => total work: unchanged; end-to-end time: may go down if host-to-device transfer dominates; quality: unchanged.

TRAIN_SCHEDULE = derive_train_schedule(
    num_images_total=NUM_IMAGES_TOTAL,
    train_split_fraction=TRAIN_SPLIT_FRACTION,
    train_image_wh=TRAIN_IMAGE_WH,
    num_devices=NUM_DEVICES,
    train_num_rays_per_batch=TRAIN_NUM_RAYS_PER_BATCH,
    train_num_images_to_sample_from=TRAIN_NUM_IMAGES_TO_SAMPLE_FROM,
    pixel_visitation=PIXEL_VISITATION,
    window_coverage=WINDOW_COVERAGE,
)

# TODO: move training/extraction to /scratch, see if speed up happens
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
        machine=MachineConfig(num_devices=NUM_DEVICES),
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
                pin_cpu_feature_cache=PIN_CPU_FEATURE_CACHE,
                cpu_feature_cache_images=CPU_FEATURE_CACHE_IMAGES,
                gpu_feature_cache_images=GPU_FEATURE_CACHE_IMAGES,
                instance_patch_size=INSTANCE_PATCH_SIZE,
                num_instance_patches=NUM_INST_PATCHES,
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
                predict_normals=True,
                use_appearance_embedding=True,
                num_proposal_iterations=1,  # May reduce proposal iterations for speed
                num_proposal_samples_per_ray=(96,),
                num_nerf_samples_per_ray=19,
                distortion_loss_mult=0.0030,
                feat_loss_weight=1e-3,
                feat_train_ray_ratio=FEAT_TRAIN_RAY_RATIO,
                feat_use_pe=False,
                feat_num_levels=10,
                feat_features_per_level=4,
                feat_hidden_dim=64,
                feat_num_layers=2,
                inst_feature_dim=8,
                inst2d_lambda=0.5,
                inst_var_lambda=0.0,
                inst_gamma=10.0,
                inst_pos_weight=1.0,
                inst_neg_weight=1.0,
                inst_min_mask_pixels=256,
                inst_use_pe=False,
                inst_num_levels=10,
                inst_features_per_level=4,
                inst_hidden_dim=64,
                inst_num_layers=2,
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
            "instance_field": {
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
