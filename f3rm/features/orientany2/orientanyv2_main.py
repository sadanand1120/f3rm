import torch
import numpy as np
np.set_printoptions(precision=2, suppress=True)
from PIL import Image, ImageDraw
import math

from f3rm.features.orientany2.vision_tower import VGGT_OriAny_Ref
from f3rm.features.orientany2.app_utils_absori import background_preprocess, inf_single_absori
from f3rm.features.orientany.orientany_main import OrientAny


class OrientAnyV2:
    def __init__(self, ckpt_path='rotmod_realrotaug_best.pt', device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.ckpt_path = ckpt_path
        self._load_model()

    def _load_model(self):
        if self.device.type == "cuda" and torch.cuda.is_available():
            device_idx = self.device.index if self.device.index is not None else torch.cuda.current_device()
            major_cc = torch.cuda.get_device_capability(device_idx)[0]
            mark_dtype = torch.bfloat16 if major_cc >= 8 else torch.float16
        else:
            # CPU path: keep weights/activations in fp32 for compatibility.
            mark_dtype = torch.float32

        self.model = VGGT_OriAny_Ref(out_dim=900, dtype=mark_dtype, nopretrain=True)
        self.model.load_state_dict(torch.load(self.ckpt_path, map_location='cpu', weights_only=True))
        # Only the aggregator is used in forward() for my usecase; drop unused VGGT heads to save GPU memory and transfer time
        del self.model.vggt.camera_head, self.model.vggt.point_head
        del self.model.vggt.depth_head, self.model.vggt.track_head
        self.model.eval()
        self.model = self.model.to(self.device)

    def get_model_outputs(self, image):
        """
        azimuth(phi/az): 0-360, angle from x-axis to y-axis (about +z-axis)
        polar(theta_elev/el): -90-90, elevation angle from xy-plane
        roll(delta/ro): rotation about the viewing axis (-180 to 180)
        """
        with torch.no_grad():
            ans_dict = inf_single_absori(self.model, image)

        # Map V2 parameters to V1 naming convention
        # phi (azimuth) = az, theta_elev (elevation) = el, delta (roll) = ro
        phi = float(ans_dict.get('az_pred', 0))
        theta_elev = float(ans_dict.get('el_pred', 0))
        delta = float(ans_dict.get('ro_pred', 0))
        alpha = int(ans_dict.get('alpha_pred', 1))

        return {
            'phi': phi,  # azimuth (0-360)
            'theta_elev': theta_elev,  # elevation (-90-90)
            'delta': delta,  # roll (-180-180)
            'alpha': alpha,  # symmetry (0,1,2,4)
        }

    @staticmethod
    def draw_axes_on_image(image, phi, theta_elev, delta, alpha=1, radius=16, axes_len=2):
        """Draw axes on image using projection math with symmetry handling"""
        # Create a copy of the image to draw on
        result_img = image.copy()
        draw = ImageDraw.Draw(result_img)

        # Colors for X, Y, Z axes
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue

        # Determine which axes to draw and how many symmetry instances
        if alpha == 0:
            # Only draw Z axis
            axes_to_draw = [2]
            azimuth_offsets = [0]
        elif alpha == 4:
            # Draw ONLY X and Z axes for all sets
            axes_to_draw = [0, 2]  # X, Z axes
            azimuth_offsets = [i * (360.0 / 4) for i in range(4)]
        else:
            # Draw all three axes for each symmetry instance
            axes_to_draw = [0, 1, 2]  # X, Y, Z axes
            # Generate azimuth offsets for symmetry: 0, 360/alpha, 2*360/alpha, etc.
            azimuth_offsets = [i * (360.0 / alpha) for i in range(alpha)]

        for az_offset in azimuth_offsets:
            # Apply symmetry rotation around Z axis
            current_phi = (phi + az_offset) % 360

            # Get rotation matrix for this orientation
            R_objw2cam = OrientAny.get_R_objw2cam(current_phi, theta_elev, delta)

            # Project axes to image plane
            pcs_pts = OrientAny.project_axes_to_image(R_objw2cam, radius, axes_len, image.width, image.height)

            if pcs_pts is not None:
                # Draw the axes
                for idx in axes_to_draw:
                    if len(pcs_pts) > idx + 1:
                        start_point = (int(pcs_pts[0][0]), int(pcs_pts[0][1]))
                        end_point = (int(pcs_pts[idx + 1][0]), int(pcs_pts[idx + 1][1]))
                        draw.line([start_point, end_point], fill=colors[idx], width=3)

        return result_img

    @staticmethod
    def preprocess_remove_bkg(input_image, do_remove_background):
        """Preprocess image with background removal"""
        if do_remove_background:
            input_image = background_preprocess(input_image, True)
        elif not input_image.mode == 'RGBA':
            return input_image.convert('RGB')
        else:
            # if RGBA -> user has provided a background alpha mask so use it as-is
            input_image = background_preprocess(input_image, False)
        return input_image


if __name__ == "__main__":
    image_path = "/robodata/smodak/repos/f3rm/f3rm/features/orientany2/bottle.jpg"
    orient_any = OrientAnyV2("/robodata/smodak/repos/f3rm/f3rm/features/orientany2/rotmod_realrotaug_best.pt")
    origin_img = Image.open(image_path).convert('RGB')
    rm_bkg_img = orient_any.preprocess_remove_bkg(origin_img, do_remove_background=True)
    outs = orient_any.get_model_outputs(rm_bkg_img)
    R_objw2cam = OrientAny.get_R_objw2cam(outs['phi'], outs['theta_elev'], outs['delta'])
    result_img = orient_any.draw_axes_on_image(origin_img, outs['phi'], outs['theta_elev'], outs['delta'], outs['alpha'], radius=16, axes_len=2)
    print(f"Azimuth: {round(outs['phi'], 2)}°")
    print(f"Elevation: {round(outs['theta_elev'], 2)}°")
    print(f"Rotation: {round(outs['delta'], 2)}°")
    print(f"Symmetry: {outs['alpha']}")
    result_img.save("output_v2.png")

    debug_angles = OrientAny._angles_from_R(R_objw2cam)
    print(f"Debug angles: phi: {debug_angles['phi']}, theta_elev: {debug_angles['theta_elev']}, delta: {debug_angles['delta']}")
