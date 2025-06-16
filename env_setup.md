conda create -n f3rm python=3.9 -y
conda activate f3rm
conda install libffi==3.3 -y
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch && pip3 install numpy==1.26.3
cd /robodata/smodak/repos/f3rm
pip install -e .
cd /robodata/smodak/repos/RoboPoint
pip install -e .
cd /robodata/smodak/repos/f3rm/Hierarchical-Localization
pip install -e .
cd /robodata/smodak/repos/f3rm
pip3 install torchtyping==0.1.5 && pip3 install typeguard==4.4.2
pip3 install open3d==0.16.0 && pip3 install timm==0.6.7 && pip3 install open_clip_torch
ns-install-cli
conda deactivate && conda activate f3rm
ns-train --help
pip install -e ".[robot]" && pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip3 install torchtyping==0.1.5 && pip3 install typeguard==4.4.2
f3rm-optimize --help
wandb login
pip3 install GPUtil psutil nvidia-ml-py
cp /robodata/smodak/repos/f3rm/nerfstudio_changes/record3d_utils.py /opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/process_data/
cp /robodata/smodak/repos/f3rm/nerfstudio_changes/hloc_utils.py /opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/process_data/
cp /robodata/smodak/repos/f3rm/nerfstudio_changes/colmap_utils.py /opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/process_data/
cp /robodata/smodak/repos/f3rm/nerfstudio_changes/base_model.py /opt/miniconda3/envs/f3rm/lib/python3.9/site-packages/nerfstudio/models/