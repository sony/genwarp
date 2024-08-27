from os.path import basename, splitext, join
from io import BytesIO
import tempfile

import gradio as gr
from gradio_model3dgscamera import Model3DGSCamera
import numpy as np
from scipy.spatial import KDTree
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, to_pil_image
from einops import rearrange, repeat
from torch import Tensor
from jaxtyping import Float

from extern.ZoeDepth.zoedepth.utils.misc import colorize

from genwarp import GenWarp
from genwarp.ops import (
    camera_lookat, get_projection_matrix, get_viewport_matrix
)

IMAGE_SIZE = 512
NEAR, FAR = 0.01, 100
FOVY = np.deg2rad(55)

# Crop the image to the shorter side.
def crop(img: Image) -> Image:
    W, H = img.size
    if W < H:
        left, right = 0, W
        top, bottom = np.ceil((H - W) / 2.), np.floor((H - W) / 2.) + W
    else:
        left, right = np.ceil((W - H) / 2.), np.floor((W - H) / 2.) + H
        top, bottom = 0, H
    return img.crop((left, top, right, bottom))

def unproject(depth):
    fovy_deg = 55
    H, W = depth.shape[2:4]

    mean_depth = depth.mean(dim=(2, 3)).squeeze()

    viewport_mtx = get_viewport_matrix(
        IMAGE_SIZE, IMAGE_SIZE,
        batch_size=1
    ).to(depth)

    # Projection matrix.
    fovy = torch.ones(1) * FOVY
    proj_mtx = get_projection_matrix(
        fovy=fovy,
        aspect_wh=1.,
        near=NEAR,
        far=FAR
    ).to(depth)

    view_mtx = camera_lookat(
        torch.tensor([[0., 0., 0.]]),
        torch.tensor([[0., 0., 1.]]),
        torch.tensor([[0., -1., 0.]])
    ).to(depth)

    scr_mtx = (viewport_mtx @ proj_mtx).to(depth)

    grid = torch.stack(torch.meshgrid(
        torch.arange(W), torch.arange(H), indexing='xy'), dim=-1
    ).to(depth)[None]  # BHW2

    screen = F.pad(grid, (0, 1), 'constant', 0)
    screen = F.pad(screen, (0, 1), 'constant', 1)
    screen_flat = rearrange(screen, 'b h w c -> b (h w) c')

    eye = screen_flat @ torch.linalg.inv_ex(
        scr_mtx.float()
    )[0].mT.to(depth)
    eye = eye * rearrange(depth, 'b c h w -> b (h w) c')
    eye[..., 3] = 1

    points = eye @ torch.linalg.inv_ex(view_mtx.float())[0].mT.to(depth)
    points = points[0, :, :3]

    # Translate to the origin.
    points[..., 2] -= mean_depth
    camera_pos = (0, 0, -mean_depth)
    view_mtx = camera_lookat(
        torch.tensor([[0., 0., -mean_depth]]),
        torch.tensor([[0., 0., 0.]]),
        torch.tensor([[0., -1., 0.]])
    ).to(depth)

    return points, camera_pos, view_mtx, proj_mtx

def calc_dist2(points: np.ndarray):
    dists, _ = KDTree(points).query(points, k=4)
    mean_dists = (dists[:, 1:] ** 2).mean(1)
    return mean_dists

def save_as_splat(
    filepath: str,
    xyz: np.ndarray,
    rgb: np.ndarray
):
    # To gaussian splat
    inv_sigmoid = lambda x: np.log(x / (1 - x))
    dist2 = np.clip(calc_dist2(xyz), a_min=0.0000001, a_max=None)
    scales = np.repeat(np.log(np.sqrt(dist2))[..., np.newaxis], 3, axis=1)
    rots = np.zeros((xyz.shape[0], 4))
    rots[:, 0] = 1
    opacities = inv_sigmoid(0.1 * np.ones((xyz.shape[0], 1)))

    sorted_indices = np.argsort((
        -np.exp(np.sum(scales, axis=-1, keepdims=True))
        / (1 + np.exp(-opacities))
    ).squeeze())

    buffer = BytesIO()
    for idx in sorted_indices:
        position = xyz[idx]
        scale = np.exp(scales[idx]).astype(np.float32)
        rot = rots[idx].astype(np.float32)
        color = np.concatenate(
            (rgb[idx], 1 / (1 + np.exp(-opacities[idx]))),
            axis=-1
        )
        buffer.write(position.tobytes())
        buffer.write(scale.tobytes())
        buffer.write((color * 255).clip(0, 255).astype(np.uint8).tobytes())
        buffer.write(
            ((rot / np.linalg.norm(rot)) * 128 + 128)
            .clip(0, 255)
            .astype(np.uint8)
            .tobytes()
        )

    with open(filepath, "wb") as f:
        f.write(buffer.getvalue())

def view_from_rt(position, rotation):
    t = np.array(position)
    euler = np.array(rotation)

    cx = np.cos(euler[0])
    sx = np.sin(euler[0])
    cy = np.cos(euler[1])
    sy = np.sin(euler[1])
    cz = np.cos(euler[2])
    sz = np.sin(euler[2])
    R = np.array([
        cy * cz + sy * sx * sz,
        -cy * sz + sy * sx * cz,
        sy * cx,
        cx * sz,
        cx * cz,
        -sx,
        -sy * cz + cy * sx * sz,
        sy * sz + cy * sx * cz,
        cy * cx
    ])
    view_mtx = np.array([
        [R[0], R[1], R[2], 0],
        [R[3], R[4], R[5], 0],
        [R[6], R[7], R[8], 0],
        [
            -t[0] * R[0] - t[1] * R[3] - t[2] * R[6],
            -t[0] * R[1] - t[1] * R[4] - t[2] * R[7],
            -t[0] * R[2] - t[1] * R[5] - t[2] * R[8],
            1
        ]
    ]).T

    B = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    return B @ view_mtx

# ZoeDepth
if 'mde' not in globals():
    mde = torch.hub.load(
        './extern/ZoeDepth',
        'ZoeD_N',
        source='local',
        pretrained=True,
        trust_repo=True
    ).to('cuda')

# GenWarp
if 'genwarp_nvs' not in globals():
    genwarp_cfg = dict(
        pretrained_model_path='checkpoints',
        checkpoint_name='multi1',
        half_precision_weights=True
    )
    genwarp_nvs = GenWarp(cfg=genwarp_cfg)


with tempfile.TemporaryDirectory() as tmpdir:
    with gr.Blocks(
        title='GenWarp Demo',
        css='img {display: inline;}'
    ) as demo:
        # Internal states.
        src_image = gr.State()
        src_depth = gr.State()
        proj_mtx = gr.State()
        src_view_mtx = gr.State()

        # Blocks.
        gr.Markdown(
            """
            # GenWarp: Single Image to Novel Views with Semantic-Preserving Generative Warping
            [![Project Site](https://img.shields.io/badge/Project-Web-green)](https://genwarp-nvs.github.io/) &nbsp;
            [![Spaces](https://img.shields.io/badge/Spaces-Demo-yellow?logo=huggingface)](https://huggingface.co/spaces/Sony/GenWarp) &nbsp;
            [![Github](https://img.shields.io/badge/Github-Repo-orange?logo=github)](https://github.com/sony/genwarp/) &nbsp;
            [![Models](https://img.shields.io/badge/Models-checkpoints-blue?logo=huggingface)](https://huggingface.co/Sony/genwarp) &nbsp;
            [![arXiv](https://img.shields.io/badge/arXiv-2405.17251-red?logo=arxiv)](https://arxiv.org/abs/2405.17251)

            ## Introduction
            This is an official demo for the paper "[GenWarp: Single Image to Novel Views with Semantic-Preserving Generative Warping](https://genwarp-nvs.github.io/)". Genwarp can generate novel view images from a single input conditioned on camera poses. In this demo, we offer a basic use of inference of the model. For detailed information, please refer the [paper](https://arxiv.org/abs/2405.17251).

            ## How to Use
            1. Upload a reference image to "Reference Input"
                - You can also select a image from "Examples"
            2. Move the camera to your desired view in "Unprojected 3DGS" 3D viewer
            3. Hit "Generate a novel view" button and check the result

            """
        )
        file = gr.File(label='Reference Input', file_types=['image'])
        examples = gr.Examples(
            examples=['./assets/pexels-heyho-5998120_19mm.jpg',
                    './assets/pexels-itsterrymag-12639296_24mm.jpg'],
            inputs=file
        )
        with gr.Row():
            image_widget = gr.Image(
                label='Reference View', type='filepath',
                interactive=False
            )
            depth_widget = gr.Image(label='Estimated Depth', type='pil')
            viewer = Model3DGSCamera(
                label = 'Unprojected 3DGS',
                width=IMAGE_SIZE,
                height=IMAGE_SIZE,
                camera_width=IMAGE_SIZE,
                camera_height=IMAGE_SIZE,
                camera_fx=IMAGE_SIZE / (np.tan(FOVY / 2.)) / 2.,
                camera_fy=IMAGE_SIZE / (np.tan(FOVY / 2.)) / 2.,
                camera_near=NEAR,
                camera_far=FAR
            )
        button = gr.Button('Generate a novel view', size='lg', variant='primary')
        with gr.Row():
            warped_widget = gr.Image(
                label='Warped Image', type='pil', interactive=False
            )
            gen_widget = gr.Image(
                label='Generated View', type='pil', interactive=False
            )

        # Callbacks
        def cb_mde(image_file: str):
            image = to_tensor(crop(Image.open(
                image_file
            ).convert('RGB')).resize((IMAGE_SIZE, IMAGE_SIZE)))[None].cuda()
            depth = mde.infer(image)
            depth_image = to_pil_image(colorize(depth[0]))
            return to_pil_image(image[0]), depth_image, image, depth

        def cb_3d(image, depth, image_file):
            xyz, camera_pos, view_mtx, proj_mtx = unproject(depth)
            rgb = rearrange(image, 'b c h w -> b (h w) c')[0]
            splat_file = join(tmpdir, f'./{splitext(basename(image_file))[0]}.splat')
            save_as_splat(splat_file, xyz.cpu().detach().numpy(), rgb.cpu().detach().numpy())
            return (splat_file, camera_pos, None), view_mtx, proj_mtx

        def cb_generate(viewer, image, depth, src_view_mtx, proj_mtx):
            src_camera_pos = viewer[1]
            src_camera_rot = viewer[2]
            tar_view_mtx = view_from_rt(src_camera_pos, src_camera_rot)
            tar_view_mtx = torch.from_numpy(tar_view_mtx).to(image)
            rel_view_mtx = (
                tar_view_mtx @ torch.linalg.inv(src_view_mtx.to(image))
            ).to(image)

            # GenWarp.
            renders = genwarp_nvs(
                src_image=image.half(),
                src_depth=depth.half(),
                rel_view_mtx=rel_view_mtx.half(),
                src_proj_mtx=proj_mtx.half(),
                tar_proj_mtx=proj_mtx.half()
            )

            warped = renders['warped']
            synthesized = renders['synthesized']
            warped_pil = to_pil_image(warped[0])
            synthesized_pil = to_pil_image(synthesized[0])

            return warped_pil, synthesized_pil

        # Events
        file.change(
            fn=cb_mde,
            inputs=file,
            outputs=[image_widget, depth_widget, src_image, src_depth]
        ).then(
            fn=cb_3d,
            inputs=[src_image, src_depth, image_widget],
            outputs=[viewer, src_view_mtx, proj_mtx])
        button.click(
            fn=cb_generate,
            inputs=[viewer, src_image, src_depth, src_view_mtx, proj_mtx],
            outputs=[warped_widget, gen_widget])

    demo.launch()
