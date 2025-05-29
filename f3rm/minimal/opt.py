import os
import cv2
import torch
import torch.multiprocessing as mp
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import patches
import math

# Set the multiprocessing start method to 'spawn' for CUDA compatibility
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # Start method can only be set once, so if it's already set, we continue
    pass

import open_clip
from f3rm.minimal.interface import NERFinterface
from f3rm.features.clip_extract import extract_clip_features, CLIPArgs
from f3rm.minimal.homography import Homography
from f3rm.minimal.utils import run_pca, viz_pca3, compute_similarity_text2vis
from f3rm.minimal.utils import exp_to_homo_T, homo_T_to_exp, se3_distance, cluster_xyz
from f3rm.minimal.parallel_cmaes import cma_es_optimize


class NERFOpt:
    def __init__(self, config_path: str = "outputs/ahgroom_colmap/f3rm/2025-04-14_190026/config.yml"):
        self.config_path = config_path

    @staticmethod
    def generate_new_pose(c2w_44: torch.Tensor, cbev2w_44: torch.Tensor, px: float, py: float, ry: float, device: torch.device) -> torch.Tensor:
        _T1 = Homography.get_std_trans(cx=px, cy=py, cz=0.0, device=device)
        _T2 = Homography.get_std_rot(axis="Y", alpha=np.deg2rad(ry), device=device)
        bev2c_44 = torch.linalg.inv(c2w_44) @ cbev2w_44
        bev2c_44_new = _T2 @ (bev2c_44 @ _T1)   # _T1 is space-fixed (ie, in bev frame)
        c2bev_44_new = torch.linalg.inv(bev2c_44_new)
        c2w_44_new = cbev2w_44 @ c2bev_44_new
        return c2w_44_new

    @staticmethod
    def get_cbev2w_44(device: torch.device) -> torch.Tensor:
        # TODO: TEMPORARY
        up_table_frame = [-0.9721904271024854, 0.23419192040271516, -1.387778837771707e-17, 0, -0.13145933632821866, -0.5457212533710385, 0.8275909172140401, 0, 0.19381509642125355, 0.8045759266360382, 0.5613316644722955, 0, 0.02624862066627097, 0.20843032201619785, 0.30019989352841997, 1]
        up_table_frame = NERFinterface.get_c2w_from_viewer(up_table_frame, device=device)
        bottom_row = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32, device=device)
        up_table_frame_44 = torch.cat([up_table_frame, bottom_row], dim=0)
        # relative transformation
        T1 = Homography.get_std_rot(axis="Z", alpha=np.deg2rad(-22), device=device)
        T2 = Homography.get_std_trans(cz=0.75, device=device)
        T = T2 @ T1
        c2w_new_44 = up_table_frame_44 @ torch.linalg.inv(T)
        return c2w_new_44

    @staticmethod
    def get_init_frame(device: torch.device) -> torch.Tensor:
        # TODO: TEMPORARY
        frame1_viewer = [-0.9840990923496374, 0.17762063799873318, -1.387778860305648e-17, 0, 0.13809956332830098, 0.7651343698373133, 0.6288862510965866, 0, 0.11170317094415382, 0.6188863545209322, -0.7774973282441247, 0, 0.6842811216887466, 0.10119783492707385, -0.5749794616971657, 1]
        frame1_c2w = NERFinterface.get_c2w_from_viewer(frame1_viewer, device=device)
        bottom_row = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32, device=device)
        frame1_c2w_44 = torch.cat([frame1_c2w, bottom_row], dim=0)
        return frame1_c2w_44

    @staticmethod
    def c2w_pose_viz(c2w_44: torch.Tensor, cbev2w_44: torch.Tensor, K: torch.Tensor, WIDTH: int, HEIGHT: int, device: torch.device, fwdT=None, d=None):
        c2w_44_1 = c2w_44.clone()
        fwdT = Homography.get_std_trans(cz=-0.02, device=device) if fwdT is None else fwdT
        c2w_44_2 = c2w_44_1 @ torch.linalg.inv(fwdT)
        c2bev_44_1 = torch.linalg.inv(cbev2w_44) @ c2w_44_1
        c2bev_44_2 = torch.linalg.inv(cbev2w_44) @ c2w_44_2
        bevccs_coords = torch.stack([c2bev_44_1[:3, 3], c2bev_44_2[:3, 3]], dim=0)
        bevccsnormal_coords = Homography.general_project_A_to_B(bevccs_coords, NERFinterface.get_nerf_ccs_to_normal_ccs_T(device=device), device=device)
        bevpcs_coords, _ = Homography.projectCCStoPCS(bevccsnormal_coords, K, WIDTH, HEIGHT, d=d, mode="none", device=device)
        return bevpcs_coords

    @torch.inference_mode()
    def __call__(self, device: torch.device):
        self.device = device
        nerfint = NERFinterface(self.config_path, device)
        frame1_c2w_44 = self.get_init_frame(device)
        c2w_new_44 = self.get_cbev2w_44(device)
        c2w_new = c2w_new_44[:3, :4]
        FX, FY, WIDTH, HEIGHT = 1150.0, 1150.0, 1920, 1440
        K_np = nerfint.get_cam_intrinsics_nodist(fx=FX, fy=FY, width=WIDTH, height=HEIGHT)
        K = torch.from_numpy(K_np).to(device, dtype=torch.float32)
        outputs, ray_bundle = nerfint.get_custom_camera_outputs(
            fx=FX,
            fy=FY,
            width=WIDTH,
            height=HEIGHT,
            c2w=c2w_new,
            fars=10.0,
            nears=1.1,
            return_rays=True
        )

        model, _, _ = open_clip.create_model_and_transforms(CLIPArgs.model_name, pretrained=CLIPArgs.model_pretrained, device=device)
        model.eval()
        tokenizer = open_clip.get_tokenizer(CLIPArgs.model_name)

        text_queries = ["magazine", "object"]
        text = tokenizer(text_queries).to(device)
        text_features = model.encode_text(text)
        sims = compute_similarity_text2vis(outputs["feature"], text_features, has_negatives=True, softmax_temp=1.0).squeeze()
        magazine_mask = (sims > 0.502)

        text_queries = ["floor", "object"]
        text = tokenizer(text_queries).to(device)
        text_features = model.encode_text(text)
        sims = compute_similarity_text2vis(outputs["feature"], text_features, has_negatives=True, softmax_temp=1.0).squeeze()
        floor_mask = (sims > 0.502)

        depth = outputs["depth"].squeeze(-1)  # (H,W)
        origins = ray_bundle.origins.squeeze(0)      # (H,W,3)
        directions = ray_bundle.directions.squeeze(0)   # (H,W,3)
        pts_nerf = origins + directions * depth.unsqueeze(-1)

        magazine_pts = pts_nerf[magazine_mask]
        labels, stats = cluster_xyz(magazine_pts.cpu().numpy(), max_auto_K=5)

        # Convert cluster centers to torch tensor on device
        nerfworld_coords = torch.stack([
            torch.from_numpy(stats[0]['center']).to(device, dtype=torch.float32),
            torch.from_numpy(stats[1]['center']).to(device, dtype=torch.float32)
        ])

        def obj(x: np.ndarray):
            x = torch.from_numpy(x).to(device, dtype=torch.float32)
            newc2w_pose_44 = self.generate_new_pose(frame1_c2w_44, c2w_new_44, px=x[0].item(), py=x[1].item(), ry=x[2].item(), device=device)
            bevpcs_coords = self.c2w_pose_viz(newc2w_pose_44, c2w_new_44, K, WIDTH, HEIGHT, device)
            if not floor_mask[bevpcs_coords[0][1], bevpcs_coords[0][0]].item():
                return torch.inf
            score1 = torch.norm(newc2w_pose_44[:3, 3].squeeze() - nerfworld_coords[1].squeeze())
            score2 = torch.abs(x[2] - 90)
            return (score1 + score2).item()  # Return scalar for optimizer compatibility
        return obj


if __name__ == "__main__":
    best_x, best_f = cma_es_optimize(
        obj_source=NERFOpt(),
        x0=np.zeros(3),
        sigma0=0.5,
        popsize=101,
        max_epochs=1000,
        repeats=1,
        n_workers=4,
        target=None
    )

    print(f"Best fitness: {best_f}")
    print(f"Best params: {best_x}")
