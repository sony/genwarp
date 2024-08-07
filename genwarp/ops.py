from typing import Dict
from jaxtyping import Float

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from einops import rearrange
from splatting import splatting_function

def sph2cart(
    azi: Float[Tensor, 'B'],
    ele: Float[Tensor, 'B'],
    r: Float[Tensor, 'B']
) -> Float[Tensor, 'B 3']:
    # z-up, y-right, x-back
    rcos = r * torch.cos(ele)
    pos_cart = torch.stack([
        rcos * torch.cos(azi),
        rcos * torch.sin(azi),
        r * torch.sin(ele)
    ], dim=1)

    return pos_cart

def get_viewport_matrix(
    width: int,
    height: int,
    batch_size: int=1,
    device: torch.device=None,
) -> Float[Tensor, 'B 4 4']:
    N = torch.tensor(
        [[width/2, 0, 0, width/2],
        [0, height/2, 0, height/2],
        [0, 0, 1/2, 1/2],
        [0, 0, 0, 1]],
        dtype=torch.float32,
        device=device
    )[None].repeat(batch_size, 1, 1)
    return N

def get_projection_matrix(
    fovy: Float[Tensor, 'B'],
    aspect_wh: float,
    near: float,
    far: float
) -> Float[Tensor, 'B 4 4']:
    batch_size = fovy.shape[0]
    proj_mtx = torch.zeros(batch_size, 4, 4, dtype=torch.float32)
    proj_mtx[:, 0, 0] = 1.0 / (torch.tan(fovy / 2.0) * aspect_wh)
    proj_mtx[:, 1, 1] = -1.0 / torch.tan(
        fovy / 2.0
    )  # add a negative sign here as the y axis is flipped in nvdiffrast output
    proj_mtx[:, 2, 2] = -(far + near) / (far - near)
    proj_mtx[:, 2, 3] = -2.0 * far * near / (far - near)
    proj_mtx[:, 3, 2] = -1.0
    return proj_mtx

def camera_lookat(
    eye: Float[Tensor, 'B 3'],
    target: Float[Tensor, 'B 3'],
    up: Float[Tensor, 'B 3']
) -> Float[Tensor, 'B 4 4']:
    B = eye.shape[0]
    f = F.normalize(eye - target)
    l = F.normalize(torch.linalg.cross(up, f))
    u = F.normalize(torch.linalg.cross(f, l))

    R = torch.stack((l, u, f), dim=1)  # B 3 3
    M_R = torch.eye(4, dtype=torch.float32)[None].repeat((B, 1, 1))
    M_R[..., :3, :3] = R

    T = - eye
    M_T = torch.eye(4, dtype=torch.float32)[None].repeat((B, 1, 1))
    M_T[..., :3, 3] = T

    return (M_R @ M_T).to(dtype=torch.float32)

def focal_length_to_fov(
    focal_length: float,
    censor_length: float = 24.
) -> float:
    return 2 * np.arctan(censor_length / focal_length / 2.)

def forward_warper(
    image: Float[Tensor, 'B C H W'],
    screen: Float[Tensor, 'B (H W) 2'],
    pcd: Float[Tensor, 'B (H W) 4'],
    mvp_mtx: Float[Tensor, 'B 4 4'],
    viewport_mtx: Float[Tensor, 'B 4 4'],
    alpha: float = 0.5
) -> Dict[str, Tensor]:
    H, W = image.shape[2:4]

    # Projection.
    points_c = pcd @ mvp_mtx.mT
    points_ndc = points_c / points_c[..., 3:4]
    # To screen.
    coords_new = points_ndc @ viewport_mtx.mT

    # Masking invalid pixels.
    invalid = coords_new[..., 2] <= 0
    coords_new[invalid] = -1000000 if coords_new.dtype == torch.float32 else -1e+4

    # Calculate flow and importance for splatting.
    new_z = points_c[..., 2:3]
    flow = coords_new[..., :2] - screen[..., :2]
    ## Importance.
    importance = alpha / new_z
    importance -= importance.amin((1, 2), keepdim=True)
    importance /= importance.amax((1, 2), keepdim=True) + 1e-6
    importance = importance * 10 - 10
    ## Rearrange.
    importance = rearrange(importance, 'b (h w) c -> b c h w', h=H, w=W)
    flow = rearrange(flow, 'b (h w) c -> b c h w', h=H, w=W)

    # Splatting.
    warped = splatting_function('softmax', image, flow, importance, eps=1e-6)
    ## mask is 1 where there is no splat
    mask = (warped == 0.).all(dim=1, keepdim=True).to(image.dtype)
    flow2 = rearrange(coords_new[..., :2], 'b (h w) c -> b c h w', h=H, w=W)

    output = dict(
        warped=warped,
        mask=mask,
        correspondence=flow2
    )

    return output