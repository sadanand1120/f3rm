"""
Orient-Anything: Object Orientation Estimation
Replicates Hugging Face Space demo exactly: https://huggingface.co/spaces/Viglong/Orient-Anything
"""
import torch
import torch.nn.functional as F
import numpy as np
np.set_printoptions(precision=2, suppress=True)
from PIL import Image, ImageOps, ImageDraw
from transformers import AutoImageProcessor
import rembg
import matplotlib.pyplot as plt
import json
import os
import math

from f3rm.features.orientany.vision_tower import DINOv2_MLP
from f3rm.features.orientany.homography import Homography


class OrientAny:
    def __init__(self, ckpt_dir='f3rm/features/orientany/ckpts', model_name='ronormsigma1_dino_weight.pt'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with open(os.path.join(ckpt_dir, "..", "model_config.json"), "r") as f:
            self.model_config = json.load(f)[model_name.split('_')[0]]
        self._load_model(ckpt_path=f"{ckpt_dir}/{model_name}")

    def _load_model(self, ckpt_path):
        self.model = DINOv2_MLP(dino_mode=self.model_config['dino_mode'],
                                in_dim=self.model_config['in_dim'],
                                out_dim=self.model_config['out_dim'],
                                evaluate=True, mask_dino=False, frozen_back=False)
        self.model.eval()
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model = self.model.to(self.device)
        # Ensure DINOv2 backbone is also on the correct device
        self.model.dinov2 = self.model.dinov2.to(self.device)
        preprocessors = {'small': "facebook/dinov2-small", 'base': "facebook/dinov2-base", 'large': "facebook/dinov2-large"}
        self.val_preprocess = AutoImageProcessor.from_pretrained(preprocessors[self.model_config['dino_mode']])

    def get_model_outputs(self, image, viz_distn=False):
        """
        azimuth(phi): 0-360, angle from x-axis to y-axis (about +z-axis) [normal spherical coordinates convention]
        polar(theta): 0-180, angle from +z-axis to rho vector joining origin to the point [normal spherical coordinates convention]
            * theta = pi - theta_model
            * theta_elev = pi/2 - theta
            => theta_elev = theta_model - pi/2
        roll(delta): the rotation of the camera about the +z of opencv-style normal CCS coordinate system facing the origin
            * ro_offset is ro_range / 2
        final ranges:
            * phi: (0, 360)
            * theta_elev: (-90, 90)
            * delta: (-ro_range / 2, ro_range / 2)
        logits: interpret them as belonging to these ranges, ie, gaus_pl_logits[0] is for theta_elev = -90, so on
        """
        image_inputs = self.val_preprocess(images=image)
        image_inputs['pixel_values'] = torch.from_numpy(np.array(image_inputs['pixel_values'])).to(self.device)
        with torch.no_grad():
            preds = self.model(image_inputs)
            preds = preds[0]

        gaus_ax_logits = preds[0:360]
        gaus_pl_logits = preds[360:540]
        gaus_ro_logits = preds[540:540 + self.model_config['ro_range']]
        conf_logits = preds[-2:]

        gaus_ax_distn = F.softmax(gaus_ax_logits)
        gaus_pl_distn = F.softmax(gaus_pl_logits)
        gaus_ro_distn = F.softmax(gaus_ro_logits)
        conf_distn = F.softmax(conf_logits)  # order: [yes, no]

        if viz_distn:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            azimuth_values = gaus_ax_distn.cpu().numpy()
            azimuth_x_labels = np.linspace(0, 360, len(azimuth_values))
            axes[0].plot(azimuth_x_labels, azimuth_values)
            axes[0].set_title('Azimuth Distribution')
            polar_values = gaus_pl_distn.cpu().numpy()
            polar_x_labels = np.linspace(-90, 90, len(polar_values))
            axes[1].plot(polar_x_labels, polar_values)
            axes[1].set_title('Polar Distribution')
            roll_values = gaus_ro_distn.cpu().numpy()
            roll_range = self.model_config['ro_range']
            roll_x_labels = np.linspace(-roll_range // 2, roll_range // 2, len(roll_values))
            axes[2].plot(roll_x_labels, roll_values)
            axes[2].set_title('Roll Distribution')
            plt.tight_layout()
            plt.show()

        gaus_ax_pred = torch.argmax(gaus_ax_distn)
        gaus_pl_pred = torch.argmax(gaus_pl_distn)
        gaus_ro_pred = torch.argmax(gaus_ro_distn)
        confidence = conf_distn[0]
        return {
            'phi': float(gaus_ax_pred),
            'theta_elev': float(gaus_pl_pred) - 90,
            'delta': float(gaus_ro_pred) - self.model_config['ro_offset'],
            'confidence': float(confidence),    # confidence < 0.5: no axes was plotted
            'gaus_ax_logits': gaus_ax_logits.cpu().numpy(),
            'gaus_pl_logits': gaus_pl_logits.cpu().numpy(),
            'gaus_ro_logits': gaus_ro_logits.cpu().numpy(),
            'conf_logits': conf_logits.cpu().numpy(),
        }

    @staticmethod
    def get_K(r=0.5, t=0.5, n=3.0, img_w=512, img_h=512):
        fx = (img_w / 2.0) * (n / r)
        fy = (img_h / 2.0) * (n / t)
        cx, cy = img_w / 2.0, img_h / 2.0
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    @staticmethod
    def get_T_from_R(R):
        R = np.asarray(R)
        T = np.zeros((4, 4))
        T[:3, :3] = R
        T[3, 3] = 1
        return T

    def get_R_objw2cam(self, phi, theta_elev, delta):
        T1 = Homography.get_std_rot("Z", np.deg2rad(phi))
        T2 = Homography.get_std_rot("Y", -np.deg2rad(theta_elev))
        T3 = Homography.get_std_rot("X", -np.deg2rad(delta))
        T_wcs_to_wcs_intermed = T3 @ T2 @ T1
        T_wcs_intermed_to_ccs_facing_origin = np.asarray([[0, 1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
        T_wcs_to_ccs_facing_origin = T_wcs_intermed_to_ccs_facing_origin @ T_wcs_to_wcs_intermed
        return T_wcs_to_ccs_facing_origin[:3, :3]

    @staticmethod
    def draw_axes_on_image(image, R_objw2cam, radius=16, axes_len=2):
        T_objw2cam = OrientAny.get_T_from_R(R_objw2cam)
        T5 = Homography.get_std_trans(cz=-radius)
        T_wcs_to_ccs = T5 @ T_objw2cam
        wcs_pts = [
            np.array([0, 0, 0]),
            np.array([axes_len, 0, 0]),
            np.array([0, axes_len, 0]),
            np.array([0, 0, axes_len])
        ]
        ccs_pts = Homography.general_project_A_to_B(wcs_pts, T_wcs_to_ccs)
        K = OrientAny.get_K(r=0.5, t=0.5, n=3.0, img_w=image.width, img_h=image.height)
        pcs_pts, _ = Homography.projectCCStoPCS(ccs_pts, K, image.width, image.height)
        if pcs_pts is None:
            print(f"Warning: No valid projected points. ccs_pts shape: {ccs_pts.shape}")
            return image
        draw = ImageDraw.Draw(image)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for idx in range(3):
            if len(pcs_pts) > idx + 1:
                start_point = (int(pcs_pts[0][0]), int(pcs_pts[0][1]))
                end_point = (int(pcs_pts[idx + 1][0]), int(pcs_pts[idx + 1][1]))
                draw.line([start_point, end_point], fill=colors[idx], width=3)
        return image, T_wcs_to_ccs

    @staticmethod
    def preprocess_remove_bkg(input_image, do_remove_background):
        if do_remove_background:
            input_image = input_image.convert('RGB')
            rembg_session = rembg.new_session()
            image = rembg.remove(input_image, session=rembg_session)
        elif not input_image.mode == 'RGBA':
            return input_image.convert('RGB')
        else:
            image = input_image
        image_array = np.array(image)
        # find non-transparent pixels (ie, foreground), black (0,0,0,0) in background
        alpha = np.where(image_array[..., 3] > 0)
        if len(alpha[0]) == 0:
            # if no foreground found
            return image.convert('RGB')
        # crop to foreground
        y1, y2 = alpha[0].min(), alpha[0].max()
        x1, x2 = alpha[1].min(), alpha[1].max()
        foreground = image_array[y1:y2, x1:x2]
        # create a square image from foreground by padding with zeros
        size = max(foreground.shape[0], foreground.shape[1])
        ph0, pw0 = (size - foreground.shape[0]) // 2, (size - foreground.shape[1]) // 2
        ph1, pw1 = size - foreground.shape[0] - ph0, size - foreground.shape[1] - pw0
        square_image = np.pad(foreground, ((ph0, ph1), (pw0, pw1), (0, 0)), mode="constant", constant_values=((0, 0), (0, 0), (0, 0)))
        # add 15% margins
        ratio = 0.85
        new_size = int(square_image.shape[0] / ratio)
        ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
        ph1, pw1 = new_size - size - ph0, new_size - size - pw0
        final_image = np.pad(square_image, ((ph0, ph1), (pw0, pw1), (0, 0)), mode="constant", constant_values=((0, 0), (0, 0), (0, 0)))
        return Image.fromarray(final_image).convert('RGB')

    @staticmethod
    def _angles_from_R(R):
        """
        R is a 3x3 rotation matrix of obj world frame to camera frame (CCS)
        Returns:
        phi in [0, 360)
        theta_elev in [-90, 90]
        delta in (-180, 180]
        """
        R = np.asarray(R, dtype=np.float64)
        if R.shape != (3, 3):
            raise ValueError("Expected a 3x3 rotation matrix.")

        # 1) Camera-center direction in world coords: v = -R^T e_z
        v = -R.T @ np.array([0.0, 0.0, 1.0])
        nv = np.linalg.norm(v)
        if nv < 1e-6:
            raise ValueError("Degenerate rotation: cannot extract viewing direction.")
        v /= nv

        # 2) Azimuth and elevation
        phi = math.degrees(math.atan2(v[1], v[0])) % 360.0       # [0, 360)
        theta_elev = math.degrees(math.asin(np.clip(v[2], -1.0, 1.0)))  # [-90, 90]

        # 3) Zero-roll reference and residual roll about +Z_C
        A2 = np.array([[0, 1, 0],
                       [0, 0, -1],
                       [-1, 0, 0]], dtype=np.float64)
        R1 = Homography.get_std_rot("Z", math.radians(phi))[:3, :3]
        R2 = Homography.get_std_rot("Y", -math.radians(theta_elev))[:3, :3]
        R0 = A2 @ R2 @ R1

        R_delta = R @ R0.T
        delta = math.degrees(math.atan2(R_delta[0, 1], R_delta[0, 0]))
        delta = ((delta + 180.0) % 360.0) - 180.0                # (-180, 180]
        return {'phi': phi, 'theta_elev': theta_elev, 'delta': delta}


if __name__ == "__main__":
    image_path = "f3rm/features/orientany/tt2.png"
    orient_any = OrientAny("f3rm/features/orientany/ckpts", "ronormsigma1_dino_weight.pt")
    origin_img = Image.open(image_path).convert('RGB')
    rm_bkg_img = orient_any.preprocess_remove_bkg(origin_img, do_remove_background=True)
    outs = orient_any.get_model_outputs(rm_bkg_img, viz_distn=True)
    R_objw2cam = orient_any.get_R_objw2cam(outs['phi'], outs['theta_elev'], outs['delta'])
    result_img, T_viz_wcs_to_ccs = orient_any.draw_axes_on_image(rm_bkg_img, R_objw2cam, radius=16, axes_len=2)
    print(f"Azimuth: {round(outs['phi'], 2)}°")
    print(f"Elevation: {round(outs['theta_elev'], 2)}°")
    print(f"Rotation: {round(outs['delta'], 2)}°")
    print(f"Confidence: {round(outs['confidence'], 2)}")
    # result_img.save("output.png")
    plt.imshow(result_img)
    plt.show()

    debug_angles = orient_any._angles_from_R(R_objw2cam)
    print(f"Debug angles rot: phi: {debug_angles['phi']}, theta_elev: {debug_angles['theta_elev']}, delta: {debug_angles['delta']}")
    debug_angles = orient_any._angles_from_R(T_viz_wcs_to_ccs[:3, :3])
    print(f"Debug angles: phi: {debug_angles['phi']}, theta_elev: {debug_angles['theta_elev']}, delta: {debug_angles['delta']}")
