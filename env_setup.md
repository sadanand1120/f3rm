<!-- f3rm env -->
conda create -n f3rm python=3.11 -y
conda activate f3rm
cd /robodata/smodak/repos/f3rm
<!-- conda install libffi==3.3 -y; INSTEAD: add export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7 to bashrc -->
pip3 install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 xformers --index-url https://download.pytorch.org/whl/cu128 && pip3 install -U pip setuptools==80.10.2 wheel ninja cmake && pip3 install -v --no-build-isolation "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch" && pip3 install numpy==1.26.4
cd /robodata/smodak/repos/f3rm && pip install -e . && cd /robodata/smodak/repos/f3rm/Hierarchical-Localization && pip install -e . && cd /robodata/smodak/repos/sam2 && pip install -e . && cd /robodata/smodak/repos/Orient-Anything && pip3 install -r requirements.txt && cd /robodata/smodak/repos/sam3 && pip install -e . && pip install -e ".[notebooks]" && cd /robodata/smodak/repos/Orient-Anything-V2 && pip install -r requirements.txt
cd /robodata/smodak/repos/f3rm
pip3 install torchtyping==0.1.5 && pip3 install typeguard==4.4.2 && pip3 install pycolmap==3.11.1
pip3 install open3d==0.18.0 && pip3 install timm==0.6.7 && pip3 install open_clip_torch spatialmath-python aiohttp openai
ns-install-cli
conda deactivate && conda activate f3rm
ns-train --help
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip3 install torchtyping==0.1.5 && pip3 install typeguard==4.4.2 && pip3 install pycolmap==3.11.1
pip3 install -U "gradio==5.50.0" "gradio_client==1.14.0"
cp /robodata/smodak/repos/f3rm/nerfstudio_changes/record3d_utils.py /opt/miniconda3/envs/f3rm/lib/python3.11/site-packages/nerfstudio/process_data/
cp /robodata/smodak/repos/f3rm/nerfstudio_changes/hloc_utils.py /opt/miniconda3/envs/f3rm/lib/python3.11/site-packages/nerfstudio/process_data/
cp /robodata/smodak/repos/f3rm/nerfstudio_changes/colmap_utils.py /opt/miniconda3/envs/f3rm/lib/python3.11/site-packages/nerfstudio/process_data/
cp /robodata/smodak/repos/f3rm/nerfstudio_changes/process_data_utils.py /opt/miniconda3/envs/f3rm/lib/python3.11/site-packages/nerfstudio/process_data/

<!-- vllm env -->
conda create -n vllm python=3.10 -y
conda activate vllm
pip3 install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128
pip3 install vllm==0.10.0 hf-transfer transformers==4.54.1 openai==1.90.0 --no-build-isolation --extra-index-url https://download.pytorch.org/whl/cu128

<!-- sam3d-objects env -->
conda create -n sam3d-objects python=3.11 -y
conda activate sam3d-objects
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
conda env update -f sam3d-objects-single.yml
conda deactivate && conda activate sam3d-objects
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
cd ../kaolin/
pip install -r tools/build_requirements.txt -r tools/viz_requirements.txt -r tools/requirements.txt
python3 setup.py develop
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3%2Bcu128torch2.8-cp311-cp311-linux_x86_64.whl
cd ../sam-3d-objects
pip install -e .
cd /robodata/smodak/repos/sam2 && pip install -e .