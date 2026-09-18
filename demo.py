from __future__ import print_function, division

import argparse
import logging
import os

import imageio
import numpy as np
import torch
from PIL import Image
import trimesh

from core.models import (
    build_model,
    load_model_weights,
    model_label,
    normalize_model_size,
    normalize_version,
    require_checkpoint,
    resolve_checkpoint,
)
from core.utils.utils import InputPadder
from Utils import *


def save_stereo_gif(img0, img1, output_gif):
    frames = [Image.fromarray(img0), Image.fromarray(img1)]
    frames[0].save(
        output_gif,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=[500, 500],
        loop=0,
        disposal=2,
        optimize=False,
    )


def load_model(args, device):
    version = normalize_version(args.version)
    model_size = normalize_model_size(version, args.model_size)
    label = model_label(version, model_size)
    ckpt_path = resolve_checkpoint(version, args.restore_ckpt, model_size=model_size)
    model = build_model(version, fnet_pretrained=False, model_size=model_size, max_disp=args.max_disp)

    if ckpt_path is not None:
        require_checkpoint(ckpt_path)
        logging.info("Loading %s checkpoint from %s", label, ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location=device)
        load_model_weights(model, checkpoint, strict=True)
        logging.info("Done loading checkpoint")

    return model.to(device).eval(), version, label


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="las1", help="model version: las1/v1 or las2/v2")
    parser.add_argument("--model_size", "--model-size", default=None, help="LAS2 model size: s, m, l, or h")
    parser.add_argument("--stereo_file", default="./assets/Explorer_HD2K_SN28883284_20-42-06.png", type=str)
    parser.add_argument("--restore_ckpt", default=None, type=str, help="checkpoint path; use 'none' to skip loading")
    parser.add_argument("--out_dir", default="./output-vis/20-42-06", type=str, help="directory to save results")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--scale", default=1, type=float, help="downsize the image by scale, must be <=1")
    parser.add_argument("--max_disp", type=int, default=192, help="maximum disparity")
    parser.add_argument("--z_far", default=3, type=float, help="max depth to clip in point cloud")
    parser.add_argument("--get_pc", type=int, default=1, help="save point cloud output")
    parser.add_argument(
        "--remove_invisible",
        default=1,
        type=int,
        help="remove non-overlapping observations between left and right images from point cloud",
    )
    parser.add_argument("--denoise_cloud", type=int, default=0, help="whether to denoise the point cloud")
    parser.add_argument("--denoise_nb_points", type=int, default=10, help="number of points for radius outlier removal")
    parser.add_argument("--denoise_radius", type=float, default=0.05, help="radius for outlier removal")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s")

    K_left = np.array([
        [1068.6300, 0.0, 1097.8500],
        [0.0, 1068.7900, 632.3400],
        [0.0, 0.0, 1.0],
    ])
    baseline = 0.1201340

    torch.autograd.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model, version, label = load_model(args, device)
    logging.info("Running %s on %s", label, device)

    img = imageio.imread(args.stereo_file)[..., :3]
    assert args.scale <= 1, "scale must be <=1"
    if args.scale != 1:
        img = cv2.resize(img, fx=args.scale, fy=args.scale, dsize=None)

    H, W = img.shape[:2]
    if W % 2 != 0:
        img = img[:, :-1, :]
        W -= 1

    img0 = img[:, : W // 2, :]
    img1 = img[:, W // 2 :, :]
    save_stereo_gif(img0, img1, os.path.join(args.out_dir, "img.gif"))

    img0_ori = img0.copy()
    logging.info("Left image: %s", img0.shape)

    img0_t = torch.as_tensor(img0, device=device).float()[None].permute(0, 3, 1, 2)
    img1_t = torch.as_tensor(img1, device=device).float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(img0_t.shape, divis_by=32)
    img0_t, img1_t = padder.pad(img0_t, img1_t)

    with torch.no_grad():
        disp = model(img0_t, img1_t, max_disp=args.max_disp, test_mode=True)
    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H, W // 2)

    disp_vis = vis_disparity(disp)
    imageio.imwrite(os.path.join(args.out_dir, "vis.png"), np.concatenate([img0_ori, disp_vis], axis=1))
    np.save(os.path.join(args.out_dir, "disp.npy"), disp)
    logging.info("Disparity output saved to %s", args.out_dir)

    if args.remove_invisible:
        _, xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing="ij")
        disp[xx - disp < 0] = np.inf

    if args.get_pc:
        K_scaled = K_left.copy()
        K_scaled[:2] *= args.scale
        depth = K_scaled[0, 0] * baseline / disp
        xyz_map = depth2xyzmap(depth, K_scaled)
        pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), img0_ori.reshape(-1, 3))
        points = np.asarray(pcd.points)
        keep_mask = (points[:, 2] > 0) & np.isfinite(points[:, 2]) & (points[:, 2] <= args.z_far)
        pcd = pcd.select_by_index(np.arange(len(points))[keep_mask])
        o3d.io.write_point_cloud(os.path.join(args.out_dir, "cloud.ply"), pcd)
        logging.info("Point cloud saved to %s", args.out_dir)

        if args.denoise_cloud:
            logging.info("Denoising point cloud...")
            _, ind = pcd.remove_radius_outlier(nb_points=args.denoise_nb_points, radius=args.denoise_radius)
            pcd = pcd.select_by_index(ind)
            o3d.io.write_point_cloud(os.path.join(args.out_dir, "cloud_denoise.ply"), pcd)

        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if len(pcd.colors) > 0 else None
        if colors is not None and colors.dtype != np.uint8:
            colors = (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)

        cloud = trimesh.points.PointCloud(vertices=points, colors=colors)
        scene = trimesh.Scene()
        scene.add_geometry(cloud)
        glb_name = "cloud_denoise.glb" if args.denoise_cloud else "cloud.glb"
        glb_path = os.path.join(args.out_dir, glb_name)
        scene.export(glb_path)
        logging.info("GLB saved to %s", glb_path)
