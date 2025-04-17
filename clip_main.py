import torch
import matplotlib.pyplot as plt
from clip.clip_util import load
from clipdino_utils import run_pca, viz_pca3, preprocess
from clip.clip_util import tokenize


@torch.inference_mode()
def compute_similarity_text2vis(img_patch_descriptors, text_embeddings):
    """
    img_patch_descriptors: (**, dim) where ** could be (num_patches = h x w), (h, w), (num_imgs, h, w) etc
    text_embeddings: (num_texts, dim)
    Returns:
        - sims: (**, num_texts) similarity scores softmaxed to [0, 1] for each text
    """
    # Normalize the image patch descriptors and text embeddings
    img_patch_descriptors /= img_patch_descriptors.norm(dim=-1, keepdim=True)
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
    # Compute similarities
    sims = img_patch_descriptors @ text_embeddings.T
    # Apply softmax to get probabilities
    flattened = sims.reshape(-1, sims.shape[-1])
    flat_softmax = torch.softmax(flattened, dim=0)
    sims = flat_softmax.reshape_as(sims)
    return sims


with torch.inference_mode():
    # image_path = "/home/dynamo/AMRL_Research/repos/f3rm/f3rm/scripts/images/frame_1.png"
    image_path = "/home/dynamo/AMRL_Research/repos/robot_calib/chairs.png"
    from PIL import Image
    pil_img = Image.open(image_path).convert("RGB")
    pil_img = pil_img.resize((336, 336), Image.LANCZOS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _ = load(name="ViT-L/14@336px", device=device)
    patch_size = model.visual.patch_size   # 14
    pil_img, _, cropped_h, cropped_w = preprocess(
        image_path=pil_img,
        load_size=336,  # model.visual.input_resolution = 336
        patch_size=patch_size,
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
        allow_crop=False
    )
    num_patches = (pil_img.shape[2] // patch_size, pil_img.shape[3] // patch_size)
    pil_img = pil_img.to(device)
    # import ipdb; ipdb.set_trace()
    descriptors = model.get_patch_encodings(pil_img)
    descriptors = descriptors.cpu().squeeze()
    projected_tokens = run_pca(descriptors, n_components=3)
    img = viz_pca3(projected_tokens, num_patches, cropped_w, cropped_h)
    plt.imshow(img)
    plt.show()

    # Compute similarity between text and image patches
    text_queries = ["teddy bear", "mug"]
    text_tokens = tokenize(text_queries).to(device)   # (num_texts, 77)
    text_embds = model.encode_text(text_tokens)  # (num_texts, dim)
    text_embds = text_embds.cpu()
    patch_d = descriptors.reshape(*num_patches, descriptors.shape[1])  # (h, w, dim)
    sims = compute_similarity_text2vis(patch_d, text_embds)
    for i, text_query in enumerate(text_queries):
        sim = sims[..., i]
        plt.imshow(sim)
        plt.axis("off")
        plt.title(text_query)
        plt.show()
