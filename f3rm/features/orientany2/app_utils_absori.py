import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms as TF
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid

def background_preprocess(input_image, do_remove_background):
    if input_image is None:
        return None
    if do_remove_background:
        # NOTE: rembg is heavy and slow, so import it only when needed
        import rembg
        rembg_session = rembg.new_session()
        # Remove background
        do_remove = True
        if input_image.mode == "RGBA" and input_image.getextrema()[3][0] < 255:
            do_remove = False
        do_remove = do_remove or False  # force=False in this context
        if do_remove:
            input_image = rembg.remove(input_image, session=rembg_session)

    # Resize foreground
    image = np.array(input_image)
    assert image.shape[-1] == 4
    alpha = np.where(image[..., 3] > 0)
    y1, y2, x1, x2 = (
        alpha[0].min(),
        alpha[0].max(),
        alpha[1].min(),
        alpha[1].max(),
    )
    # crop the foreground
    fg = image[y1:y2, x1:x2]
    # pad to square
    size = max(fg.shape[0], fg.shape[1])
    ph0, pw0 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
    ph1, pw1 = size - fg.shape[0] - ph0, size - fg.shape[1] - pw0
    new_image = np.pad(
        fg,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )

    # compute padding according to the ratio
    new_size = int(new_image.shape[0] / 0.85)
    # pad to size, double side
    ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
    ph1, pw1 = new_size - size - ph0, new_size - size - pw0
    new_image = np.pad(
        new_image,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )
    input_image = Image.fromarray(new_image)

    return input_image

def val_fit_alpha(distribute):
    fit_alphas = []
    for y_noise in distribute:
        x = np.linspace(0, 2 * np.pi, 360)
        y_noise /= trapezoid(y_noise, x) + 1e-8

        initial_guess = [x[np.argmax(y_noise)], 1]

        # support 1,2,4
        alphas = [1.0, 2.0, 4.0]
        saved_params = []
        saved_r_squared = []

        for alpha in alphas:
            try:
                # Inline von_mises_pdf_alpha_numpy
                normalization = 2 * np.pi
                pdf_func = lambda x_val, mu, kappa: np.exp(kappa * np.cos(alpha * (x_val - mu))) / normalization

                params, covariance = curve_fit(pdf_func, x, y_noise, p0=initial_guess)

                residuals = y_noise - pdf_func(x, *params)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y_noise - np.mean(y_noise))**2)
                r_squared = 1 - (ss_res / (ss_tot+1e-8))

                saved_params.append(params)
                saved_r_squared.append(r_squared)
                if r_squared > 0.8:
                    break
            except:
                saved_params.append((0.,0.))
                saved_r_squared.append(0.)

        max_index = np.argmax(saved_r_squared)
        alpha = alphas[max_index]
        mu_fit, kappa_fit = saved_params[max_index]
        r_squared = saved_r_squared[max_index]

        if alpha == 1. and kappa_fit>=0.6 and r_squared>=0.45:
            pass
        elif alpha == 2. and kappa_fit>=0.45 and r_squared>=0.45:
            pass
        elif alpha == 4. and kappa_fit>=0.25 and r_squared>=0.45:
            pass
        else:
            alpha=0.
        fit_alphas.append(alpha)
    return torch.tensor(fit_alphas)

@torch.no_grad()
def inf_single_absori(model, image):
    """Run absolute orientation inference on a single image."""
    device = model.get_device()

    # Preprocess single image inline
    # Validate mode
    mode = "pad"
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    # If there's an alpha channel, blend onto white background:
    if image.mode == "RGBA":
        # Create white background
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        # Alpha composite onto the white background
        image = Image.alpha_composite(background, image)

    # Now convert to "RGB" (this step assigns white for transparent areas)
    image = image.convert("RGB")

    width, height = image.size

    if mode == "pad":
        # Make the largest dimension 518px while maintaining aspect ratio
        if width >= height:
            new_width = 518
            new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
        else:
            new_height = 518
            new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
    else:  # mode == "crop"
        # Original behavior: set width to 518px
        new_width = 518
        # Calculate height maintaining aspect ratio, divisible by 14
        new_height = round(height * (new_width / width) / 14) * 14

    # Resize with new dimensions (width, height)
    to_tensor = TF.ToTensor()
    try:
        image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
        image_tensor = to_tensor(image)  # Convert to tensor (0, 1)
    except Exception as e:
        print(e)
        print(width, height)
        print(new_width, new_height)
        assert False

    # Center crop height if it's larger than 518 (only in crop mode)
    if mode == "crop" and new_height > 518:
        start_y = (new_height - 518) // 2
        image_tensor = image_tensor[:, start_y : start_y + 518, :]

    # For pad mode, pad to make a square of 518 x 518
    if mode == "pad":
        h_padding = 518 - image_tensor.shape[1]
        w_padding = 518 - image_tensor.shape[2]

        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left

            # Pad with white (value=1.0)
            image_tensor = torch.nn.functional.pad(
                image_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
            )

    # Preprocess single image - returns (1, C, H, W)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    # Add batch dimension: (1, 1, C, H, W)
    batch_input = image_tensor.unsqueeze(0)

    # Forward pass through model
    pose_enc = model(batch_input)  # Shape: (B, S, D) = (1, 1, 900)

    # Flatten to (B*S, D) = (1, 900)
    pose_enc = pose_enc.view(-1, pose_enc.shape[-1])

    # Decode pose encoding into angles
    angle_az_pred = torch.argmax(pose_enc[:, 0:360], dim=-1).item()        # 0-359 degrees
    angle_el_pred = torch.argmax(pose_enc[:, 360:540], dim=-1).item() - 90  # -90 to 90 degrees
    angle_ro_pred = torch.argmax(pose_enc[:, 540:900], dim=-1).item() - 180 # -180 to 180 degrees

    # Alpha prediction (symmetry detection)
    distribute = F.sigmoid(pose_enc[:, 0:360]).cpu().float().numpy()
    alpha_pred = val_fit_alpha(distribute)[0].item()

    return {
        'az_pred': angle_az_pred,
        'el_pred': angle_el_pred,
        'ro_pred': angle_ro_pred,
        'alpha_pred': alpha_pred
    }