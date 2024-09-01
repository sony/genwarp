# GenWarp: Single Image to Novel Views with Semantic-Preserving Generative Warping

[![Project Site](https://img.shields.io/badge/Project-Web-green)](https://genwarp-nvs.github.io/) &nbsp;
[![Spaces](https://img.shields.io/badge/Spaces-Demo-yellow?logo=huggingface)](https://huggingface.co/spaces/Sony/genwarp) &nbsp; 
[![Github](https://img.shields.io/badge/Github-Repo-orange?logo=github)](https://github.com/sony/genwarp/) &nbsp; 
[![Models](https://img.shields.io/badge/Models-checkpoints-blue?logo=huggingface)](https://huggingface.co/Sony/genwarp) &nbsp; 
[![arXiv](https://img.shields.io/badge/arXiv-2405.17251-red?logo=arxiv)](https://arxiv.org/abs/2405.17251)

[Introduction](#introduction)
| [Demo](#demo)
| [Examples](#examples)
| [How to use](#how-to-use)
| [Citation](#citation)
| [Acknowledgements](#acknowledgements)

![concept image](https://github.com/user-attachments/assets/2c89bd9c-fa9e-40af-bc27-f00d3e12de3a)

## Introduction

This repository is an official implementation for the paper "[GenWarp: Single Image to Novel Views with Semantic-Preserving Generative Warping](https://genwarp-nvs.github.io/)". Genwarp can generate novel view images from a single input conditioned on camera poses. In this repository, we offer the codes for inference of the model. For detailed information, please refer to the [paper](https://arxiv.org/abs/2405.17251).

![Framework](https://github.com/user-attachments/assets/b89d00cf-ea19-4354-b23d-07ccc0ee0f62)

## Demo

Here is a quick preview of GenWarp in action. Try it out by yourself at [Spaces](https://huggingface.co/spaces/Sony/genwarp) or run it locally on your machine. See [How to use](#how-to-use) section for more details. (Left) 3D scene reconstructed from the input image and the estimated depth. (Middle) Warped image. (Right) Generated image.

<video autoplay loop src="https://github.com/user-attachments/assets/f8cc4bcd-a30e-4c7d-8b9d-ef8820c5ac4a" width="1592" height="522"></video>

## Examples

Our model can handle images from various domains including indoor/outdoor scenes, and even illustrations with challenging camera viewpoint changes.

You can find examples on our [project page](https://genwarp-nvs.github.io/) and on our [paper](https://arxiv.org/abs/2405.17251). Or even better, you can try your favourite images on the live demo at [Spaces](https://huggingface.co/spaces/Sony/genwarp).

![Examples](https://github.com/user-attachments/assets/4490519b-db75-4034-a329-6c62c2b6875b)

Generated novel views can be used for 3D reconstruction. In the example below, we reconstructed a 3D scene via [InstantSplat](https://instantsplat.github.io/). We generated the video using [this implementation](https://github.com/ONground-Korea/unofficial-Instantsplat).

<video autoplay loop src="https://github.com/user-attachments/assets/b3362776-815c-426f-bf39-d04722eb8a6f" width="852" height="480"></video>

## How to use

### Environment

We tested our codes on Ubuntu 20.04 with nVidia A100 GPU. If you're using other machines like Windows, consider using Docker. You can either add packages to your python environment or use Docker to build an python environment. Commands below are all expected to run in the root directory of the repository.

#### Use Docker to build an environment

> [!NOTE]
> You may want to change username and uid variables written in DockerFile. Please check DockerFile before running the commands below.

``` shell
docker build . -t genwarp:latest
docker run --gpus=all -it -v $(pwd):/workspace/genwarp -w /workspace/genwarp genwarp
```

Inside the docker container, you can install packages as below.

#### Add dependencies to your python environment

We tested the environment with python `>=3.10` and CUDA `=11.8`. To add mandatory dependencies run the command below.

``` shell
pip install -r requirements.txt
```

To run developmental codes such as the example provided in jupyter notebook and the live demo implemented by gradio, add extra dependencies via the command below.

``` shell
pip install -r requirements_dev.txt
```

### Download pretrained models

GenWarp uses pretrained models which consist of both our finetuned models and publicly available third-party ones. Download all the models to `checkpoints` directory or anywhere of your choice. You can do it manually or by the [download_models.sh](scripts/download_models.sh) script.

#### Download script

``` shell
./scripts/download_models.sh ./checkpoints
```

#### Manual download

> [!NOTE]
> Models and checkpoints provided below may be distributed under different licenses. Users are required to check licenses carefully on their behalf.

1. Our finetuned models:
    - For details about each model, check out the [model card](https://huggingface.co/Sony/genwarp).
    - [multi-dataset model 1](https://huggingface.co/Sony/genwarp)
      - download all files into `checkpoints/multi1`
    - [multi-dataset model 2](https://huggingface.co/Sony/genwarp)
      - download all files into `checkpoints/multi2`
2. Pretrained models:
    - [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
      - download `config.json` and `diffusion_pytorch_model.safetensors` to `checkpoints/sd-vae-ft-mse`
    - [sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers)
      - download `image_encoder/config.json` and `image_encoder/pytorch_model.bin` to `checkpoints/image_encoder`

The final `checkpoints` directory must look like this:

```
genwarp
└── checkpoints
    ├── image_encoder
    │   ├── config.json
    │   └── pytorch_model.bin
    ├── multi1
    │   ├── config.json
    │   ├── denoising_unet.pth
    │   ├── pose_guider.pth
    │   └── reference_unet.pth
    ├── multi2
    │   ├── config.json
    │   ├── denoising_unet.pth
    │   ├── pose_guider.pth
    │   └── reference_unet.pth
    └── sd-vae-ft-mse
        ├── config.json
        └── diffusion_pytorch_model.safetensors
```

### Inference

#### (Recommended) Install MDE module

The model requires depth maps to generate novel views. To this end, users can install one of Monocular Depth Estimation (MDE) models publicly available. We used and therefore recommend ZoeDepth.

``` shell
git clone https://github.com/isl-org/ZoeDepth.git extern/ZoeDepth
```

To use ZoeDepth, please install `requirements_dev.txt` for additional packages.

#### API

**Initialisation**

Import GenWarp class and instantiate it with a config. Set the path to the checkpoints directory to `pretrained_model_path` and select the model version in `checkpoint_name`. For more options, check out [GenWarp.py](genwarp/GenWarp.py)

``` python
from genwarp import GenWarp

genwarp_cfg = dict(
    pretrained_model_path='./checkpoints',
    checkpoint_name='multi1',
    half_precision_weights=True
)
genwarp_nvs = GenWarp(cfg=genwarp_full_cfg)

# Load MDE model.
depth_estimator = torch.hub.load(
    './extern/ZoeDepth',
    'ZoeD_N',
    source='local',
    pretrained=True,
    trust_repo=True
).to('cuda')
```

**Prepare inputs**

Load the input image and estimate the corresponding depth map. Create camera matrices for the intrinsic and extrinsic parameters. [ops.py](genwarp/ops.py) has helper functions to create matrices.

``` python
from PIL import Image
from torchvision.transforms.functional import to_tensor

src_image = to_tensor(Image.open(image_file).convert('RGB'))[None].cuda()
src_depth = depth_estimator.infer(src_image)
```

``` python
import torch
from genwarp.ops import camera_lookat, get_projection_matrix

proj_mtx = get_projection_matrix(
    fovy=fovy,
    aspect_wh=1.,
    near=near,
    far=far
)

src_view_mtx = camera_lookat(
    torch.tensor([[0., 0., 0.]]),  # From (0, 0, 0)
    torch.tensor([[-1., 0., 0.]]), # Cast rays to -x
    torch.tensor([[0., 0., 1.]])   # z-up
)

tar_view_mtx = camera_lookat(
    torch.tensor([[-0.1, 2., 1.]]), # Camera eye position
    torch.tensor([[-5., 0., 0.]]),  # Looking at.
    z_up  # z-up
)

rel_view_mtx = (
    tar_view_mtx @ torch.linalg.inv(src_view_mtx.float())
).to(src_image)
```

**Warping**

Call the main function of GenWarp. And check the result.

``` python
renders = genwarp_nvs(
    src_image=src_image,
    src_depth=src_depth,
    rel_view_mtx=rel_view_mtx,
    src_proj_mtx=proj_mtx,
    tar_proj_mtx=proj_mtx
)

# Outputs.
renders['synthesized']     # Generated image.
renders['warped']          # Depth based warping image (for comparison).
renders['mask']            # Mask image (mask=1 where visible pixels).
renders['correspondence']  # Correspondence map.
```

### Example notebook

We provide a complete example in [genwarp_inference.ipynb](examples/genwarp_inference.ipynb)

To access a Jupyter Notebook running in a docker container, you may need to use the host's network. For further details, please refer to the manual of Docker.

``` shell
docker run --gpus=all -it --net host -v $(pwd):/workspace/genwarp -w /workspace/genwarp genwarp
```

Install `requirements_dev.txt` for additional packages  to run the Jupyter Notebook.

### Gradio live demo

An interactive live demo is also available. Start gradio demo by running the command below, and goto [http://127.0.0.1:7860/](http://127.0.0.1:7860/)
If you are running it on the server, be sure to forward the port 7860.

Or you can just visit [Spaces](https://huggingface.co/spaces/Sony/genwarp) hosted by Hugging Face to try it now.

```shell
python app.py
```

## Citation

``` bibtex
@misc{seo2024genwarpsingleimagenovel,
  title={GenWarp: Single Image to Novel Views with Semantic-Preserving Generative Warping}, 
  author={Junyoung Seo and Kazumi Fukuda and Takashi Shibuya and Takuya Narihira and Naoki Murata and Shoukang Hu and Chieh-Hsin Lai and Seungryong Kim and Yuki Mitsufuji},
  year={2024},
  eprint={2405.17251},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2405.17251}, 
}
```

## Acknowledgements

Our codes are based on [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone) and other repositories it is based on. We thank the authors of relevant repositories and papers.
