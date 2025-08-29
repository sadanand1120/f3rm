#!/bin/bash

mkdir -p f3rm/features/orientany/ckpts
cd f3rm/features/orientany/ckpts

# # ViT-S (384 input, 722 output)
# wget https://huggingface.co/Viglong/Orient-Anything/resolve/main/cropsmallEx03/dino_weight.pt   -O cropsmallEx03_dino_weight.pt

# # ViT-B (768 input, 902 output)
# wget https://huggingface.co/Viglong/Orient-Anything/resolve/main/base100p/dino_weight.pt        -O base100p_dino_weight.pt
# # ViT-B (768 input, 902 output)
# wget https://huggingface.co/Viglong/Orient-Anything/resolve/main/base100p2/dino_weight.pt       -O base100p2_dino_weight.pt
# # ViT-B (768 input, 902 output)
# wget https://huggingface.co/Viglong/Orient-Anything/resolve/main/base25p/dino_weight.pt         -O base25p_dino_weight.pt
# # ViT-B (768 input, 902 output)
# wget https://huggingface.co/Viglong/Orient-Anything/resolve/main/base50p/dino_weight.pt         -O base50p_dino_weight.pt
# # ViT-B (768 input, 902 output)
# wget https://huggingface.co/Viglong/Orient-Anything/resolve/main/base75p/dino_weight.pt         -O base75p_dino_weight.pt
# # ViT-B (768 input, 902 output)
# wget https://huggingface.co/Viglong/Orient-Anything/resolve/main/base75p2/dino_weight.pt        -O base75p2_dino_weight.pt
# # ViT-B (768 input, 722 output)
# wget https://huggingface.co/Viglong/Orient-Anything/resolve/main/cropbaseEx03/dino_weight.pt    -O cropbaseEx03_dino_weight.pt

# # ViT-L (1024 input, 602 output)
# wget https://huggingface.co/Viglong/Orient-Anything/resolve/main/celarge/dino_weight.pt         -O celarge_dino_weight.pt
# # ViT-L (1024 input, 722 output)
# wget https://huggingface.co/Viglong/Orient-Anything/resolve/main/croplargeEX03/dino_weight.pt   -O croplargeEX03_dino_weight.pt
# # ViT-L (1024 input, 722 output)
# wget https://huggingface.co/Viglong/Orient-Anything/resolve/main/croplargeEX2/dino_weight.pt    -O croplargeEX2_dino_weight.pt
# # ViT-L (1024 input, 902 output)
# wget https://huggingface.co/Viglong/Orient-Anything/resolve/main/mixreallarge/dino_weight.pt    -O mixreallarge_dino_weight.pt
# ViT-L (1024 input, 902 output)
wget https://huggingface.co/Viglong/Orient-Anything/resolve/main/ronormsigma1/dino_weight.pt    -O ronormsigma1_dino_weight.pt
