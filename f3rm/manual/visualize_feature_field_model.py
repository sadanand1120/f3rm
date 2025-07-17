"""
Minimal script to visualize FeatureFieldModel architecture using torchview.
Saves the SVG graph to 'feature_field_model_arch.svg'.
Set DO_LOAD = True to load a real model as in opt.py/interface.py, else use minimal config.
"""
from torchview import draw_graph
from f3rm.model import FeatureFieldModel, FeatureFieldModelConfig
import torch
import os
from nerfstudio.cameras.rays import RayBundle

# Set this flag to True to load a real model, False for minimal config
DO_LOAD = False

if DO_LOAD:
    # --- Load model as in opt.py/interface.py ---
    from f3rm.minimal.interface import NERFinterface
    # You may want to change this path to your actual config
    config_path = "outputs/ahgroom_colmap/f3rm/2025-04-14_190026/config.yml"
    device = torch.device("cuda")
    nerfint = NERFinterface(config_path, device)
    model = nerfint.model  # This is the loaded FeatureFieldModel
else:
    # --- Use correct hyperparameters from f3rm_config.py/model.py ---
    # These match the defaults in f3rm_config.py FeatureFieldModelConfig
    config = FeatureFieldModelConfig(
        feat_loss_weight=1e-3,
        feat_use_pe=True,
        feat_pe_n_freq=6,
        feat_num_levels=12,
        feat_log2_hashmap_size=19,
        feat_start_res=16,
        feat_max_res=128,
        feat_features_per_level=8,
        feat_hidden_dim=64,
        feat_num_layers=2,
    )
    # Use CLIP features and 512 dim as in your config
    model = FeatureFieldModel(config=config, metadata={'feature_type': 'CLIP', 'feature_dim': 512}, device=torch.device('cuda'))

# --- Create a dummy RayBundle for input ---
batch_size = 8  # Number of rays in the bundle
origins = torch.zeros((batch_size, 3), device=torch.device('cuda'))
directions = torch.zeros((batch_size, 3), device=torch.device('cuda'))
directions[:, 2] = 1.0  # All rays point in +z
pixel_area = torch.ones((batch_size, 1), device=torch.device('cuda'))
camera_indices = torch.zeros((batch_size, 1), dtype=torch.long, device=torch.device('cuda'))
nears = torch.ones((batch_size, 1), device=torch.device('cuda')) * 0.1
fars = torch.ones((batch_size, 1), device=torch.device('cuda')) * 1.0

ray_bundle = RayBundle(
    origins=origins,
    directions=directions,
    pixel_area=pixel_area,
    camera_indices=camera_indices,
    nears=nears,
    fars=fars,
)

# Visualize the model architecture using the dummy RayBundle
# torchview expects example_inputs for non-standard input types
# device='meta' disables memory allocation, but disables real forward pass
# For real input tracing, use device='cpu' (may use RAM)
graph = draw_graph(model, input_data=(ray_bundle,), device='cuda')

# Save the SVG to disk (convert Digraph to SVG string)
svg_str = graph.visual_graph.pipe(format='svg').decode()
with open('feature_field_model_arch.svg', 'w') as f:
    f.write(svg_str)

print("Saved model architecture SVG to feature_field_model_arch.svg")
