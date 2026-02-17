import random
import time
from typing import Dict, Any
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
import numpy as np
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from gaussian_splatting.scene.gaussian_model import GaussianModel
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_mapping
from utils.camera_utils import Camera
import cv2
import math
from dataclasses import dataclass
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import sys
from gaussian_splatting.utils.sh_utils import eval_sh

from utils.slam_utils import get_loss_tracking, get_median_depth, get_loss_tracking_rgb, get_loss_tracking_rgbd

"""
Analytical Jacobian Verification for MonoGS Pose Optimization

Key corrections applied:
1. Fixed sign in covariance gradient: ∂α/∂Σ has POSITIVE coefficient 0.5, not -0.5
   - For α = opacity * exp(-0.5 * Δ^T Σ^{-1} Δ)
   - ∂α/∂Σ = α * 0.5 * Σ^{-1} (ΔΔ^T) Σ^{-1}

2. CRITICAL: Fixed viewing direction for spherical harmonics
   - Viewing directions MUST be computed in WORLD coordinates
   - Camera center must be extracted from w2c: c = -R^T @ t
   - viewdir = (camera_center_world - gaussian_world) / ||...||
   - Previously was incorrectly using camera frame coordinates

3. Ensured proper coordinate transformations:
   - World → Camera: xyz_cam = w2c @ xyz_world
   - Camera → Normalized image plane: uses Jacobian of projection
   - Normalized → Pixel space: uses intrinsic matrix K

4. Verified gradient flattening order matches Jacobian structure:
   - Covariance gradients flattened as [Σ_00, Σ_01, Σ_10, Σ_11]
   - Matches dcovI_dTcw output from analytical Jacobian

5. Added extensive debugging to track gradient flow through chain rule
"""



def get_render_settings(
    w: int,
    h: int,
    intrinsics: np.ndarray,
    w2c: torch.Tensor,
    near = 0.01,
    far = 100,
    sh_degree = 0
) -> Dict[str, Any]:
    """
    Create render settings for Gaussian splatting rasterization.
    
    Args:
        w: Image width
        h: Image height
        intrinsics: 3x3 camera intrinsics matrix (numpy array or similar)
        w2c: 4x4 world-to-camera transformation tensor
        near: Near clipping plane
        far: Far clipping plane
        sh_degree: Spherical harmonics degree
        
    Returns:
        Dictionary containing raster settings
    """
    
    # Extract camera intrinsics
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    # Convert w2c to tensor and move to GPU
    w2c_tensor = w2c.transpose(0, 1)
    
    # Compute camera center
    w2c_inv = torch.inverse(w2c_tensor)
    cam_center = w2c_inv[0:3, 3]
    
    # Get view matrix (transpose of w2c)
    viewmatrix = w2c_tensor.transpose(0, 1)
    
    # Create OpenGL projection matrix
    opengl_proj = torch.zeros(4, 4, dtype=torch.float32, device='cuda')
    
    opengl_proj[0, 0] = 2 * fx / w
    opengl_proj[0, 2] = -(w - 2 * cx) / w
    opengl_proj[1, 1] = 2 * fy / h
    opengl_proj[1, 2] = -(h - 2 * cy) / h
    opengl_proj[2, 2] = far / (far - near)
    opengl_proj[2, 3] = -(far * near) / (far - near)
    opengl_proj[3, 2] = 1.0
    
    opengl_proj = opengl_proj.transpose(0, 1)
    
    # Compute full projection matrix
    full_proj_matrix = torch.matmul(
        viewmatrix.unsqueeze(0),
        opengl_proj.unsqueeze(0)
    ).squeeze(0)

    print(" get_render_settings: full_proj_matrix", full_proj_matrix)
    print(" get_render_settings: viewmatrix", viewmatrix)
    print(" get_render_settings: projmatrix", opengl_proj)
    
    # Create raster settings dictionary
    raster_settings = {
        'image_height': h,
        'image_width': w,
        'tanfovx': w / (2 * fx),
        'tanfovy': h / (2 * fy),
        'bg': torch.zeros(3, dtype=torch.float32, device='cuda'),
        'scale_modifier': 1.0,
        'viewmatrix': viewmatrix,
        'full_proj_matrix': full_proj_matrix,
        'projmatrix': opengl_proj,      # pure projection matrix P^T (for Jacobian entries)
        'sh_degree': sh_degree,
        'campos': cam_center,
        'prefiltered': False,
        'debug': False
    }
    
    return raster_settings


@dataclass
class pipeline_params:
    convert_SHs_python =  False
    compute_cov3D_python =  False


def compute_loss(gaussian_model, color, depth, color_gt, depth_gt, mask, compute_depth_loss=True):
    """
    Compute loss for Gaussian Splatting optimization.
    
    Args:
        color: Rendered RGB image [3, H, W]
        depth: Rendered depth map [1, H, W]
        color_gt: Ground truth RGB image [3, H, W]
        depth_gt: Ground truth depth map [H, W] or [1, H, W]
        mask: Valid pixel mask [H, W]
        compute_depth_loss: Whether to include depth loss
    
    Returns:
        Total loss (scalar tensor)
    """
    # Compute L1 color loss with mask
    l1_color = torch.nn.functional.l1_loss(
        color * mask.unsqueeze(0), 
        color_gt * mask.unsqueeze(0)
    )
    
    # Compute isotropic loss (regularization on Gaussian scales)
    scales = gaussian_model.get_scaling
    isotropic_loss = torch.abs(scales - scales.mean(dim=1, keepdim=True)).mean()
    
    if compute_depth_loss:
        # Create depth mask: valid depth AND valid pixels
        if depth_gt.dim() == 2:
            depth_mask = (depth_gt > 0.0) & mask
        else:
            depth_mask = (depth_gt.squeeze(0) > 0.0) & mask
        
        # Compute L1 depth loss only on valid pixels
        depth_valid = depth.squeeze(0)[depth_mask]
        depth_gt_valid = depth_gt.squeeze(0)[depth_mask] if depth_gt.dim() == 3 else depth_gt[depth_mask]
        
        l1_depth = torch.nn.functional.l1_loss(depth_valid, depth_gt_valid)
        loss = l1_color + l1_depth + 10.0 * isotropic_loss
    else:
        loss = l1_color + 10.0 * isotropic_loss
    
    return loss


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))



# Display the Rendered images 

def DisplayRenderedImage(viewpoint, render_image):

	col_img = viewpoint.original_image.permute(1, 2, 0).cpu().detach().numpy()
	col_img = (col_img * 255).clip(0, 255).astype('uint8')
	col_img = cv2.cvtColor(col_img, cv2.COLOR_RGB2BGR)

	cv2.imwrite("./Jacob_test_result/viewpoint_orig_image.png", col_img)

	# Show original (very dark)
	img_display = render_image.permute(1, 2, 0).cpu().detach().numpy()
	fig, axes = plt.subplots(1, 3, figsize=(15, 5))

	# CORRECT way to save with cv2.imwrite:
	# 1. Convert from float [0,1] to uint8 [0,255]
	# 2. Convert from RGB to BGR
	img_to_save = (img_display * 255).clip(0, 255).astype('uint8')
	img_to_save_bgr = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
	cv2.imwrite("./Jacob_test_result/rendered_noisy_img.png", img_to_save_bgr)

	axes[0].imshow(img_display)
	axes[0].set_title('Original Render (Very Dark)')
	axes[0].axis('off')

	# Show with contrast enhancement
	img_enhanced = np.clip(img_display * 5, 0, 1)  # 5x brightness
	axes[1].imshow(img_enhanced)
	axes[1].set_title('5x Brightness')
	axes[1].axis('off')

	# Show depth map
	depth_display = render_depth.squeeze().cpu().detach().numpy()
	axes[2].imshow(depth_display, cmap='viridis')
	axes[2].set_title('Depth Map')
	axes[2].axis('off')

	plt.tight_layout()
	plt.savefig('./Jacob_test_result/debug_visualization.png', dpi=150, bbox_inches='tight')
	plt.show()

	print(f"\nImage statistics:")
	print(f"  Non-zero pixels: {(img_display > 0).sum()} / {img_display.size}")
	print(f"  Pixels > 0.1: {(img_display > 0.1).sum()}")
	print(f"  Pixels > 0.5: {(img_display > 0.5).sum()}")
	print(f"\nSaved corrected image to: ./Jacob_test_result/rendered_img_corrected.png")


def VisualizeGaussiansInWorldFrame(gaussian_model):
	# Get the 3D positions of all Gaussians
	xyz = gaussian_model.get_xyz.cpu().detach().numpy()

	# Create interactive 3D scatter plot
	fig = go.Figure(data=[go.Scatter3d(
	    x=xyz[:, 0],
	    y=xyz[:, 1],
	    z=xyz[:, 2],
	    mode='markers',
	    marker=dict(
	        size=2,
	        color=xyz[:, 2],  # Color by Z coordinate
	        colorscale='Viridis',
	        showscale=True,
	        colorbar=dict(title="Z"),
	        opacity=0.8
	    ),
	    text=[f'Gaussian {i}' for i in range(len(xyz))],
	    hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
	)])

	fig.update_layout(
	    title=f'3D Gaussian Centers ({len(xyz)} Gaussians)',
	    scene=dict(
	        xaxis_title='X',
	        yaxis_title='Y',
	        zaxis_title='Z',
	        aspectmode='data'
	    ),
	    width=900,
	    height=700,
	    hovermode='closest'
	)

	fig.show()


def VisualizeGaussiansInCameraFrame(xyz_cam_np):
	# Visualize Gaussians in camera coordinates


	# Create interactive 3D scatter plot in camera coordinates
	fig = go.Figure(data=[go.Scatter3d(
	    x=xyz_cam_np[:, 0],
	    y=xyz_cam_np[:, 1],
	    z=xyz_cam_np[:, 2],
	    mode='markers',
	    marker=dict(
	        size=2,
	        color=xyz_cam_np[:, 2],  # Color by depth (Z in camera coords)
	        colorscale='Plasma',
	        showscale=True,
	        colorbar=dict(title="Depth (Z)"),
	        opacity=0.8
	    ),
	    text=[f'Gaussian {i}<br>Depth: {xyz_cam_np[i,2]:.3f}m' for i in range(len(xyz_cam_np))],
	    hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
	)])

	# Add camera frame axes
	axis_length = 0.5
	fig.add_trace(go.Scatter3d(
	    x=[0, axis_length], y=[0, 0], z=[0, 0],
	    mode='lines+text',
	    line=dict(color='red', width=4),
	    text=['', 'X'],
	    textposition='top center',
	    name='X-axis',
	    showlegend=True
	))
	fig.add_trace(go.Scatter3d(
	    x=[0, 0], y=[0, axis_length], z=[0, 0],
	    mode='lines+text',
	    line=dict(color='green', width=4),
	    text=['', 'Y'],
	    textposition='top center',
	    name='Y-axis',
	    showlegend=True
	))
	fig.add_trace(go.Scatter3d(
	    x=[0, 0], y=[0, 0], z=[0, axis_length],
	    mode='lines+text',
	    line=dict(color='blue', width=4),
	    text=['', 'Z'],
	    textposition='top center',
	    name='Z-axis (depth)',
	    showlegend=True
	))

	fig.update_layout(
	    title=f'3D Gaussians in Camera Coordinates ({len(xyz_cam_np)} Gaussians)',
	    scene=dict(
	        xaxis_title='X (right)',
	        yaxis_title='Y (down)',
	        zaxis_title='Z (forward/depth)',
	        aspectmode='data'
	    ),
	    width=900,
	    height=700,
	    hovermode='closest'
	)

	fig.show()


## Helper for evaluating SH and computing colors

def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions using hardcoded SH polynomials.
    
    Args:
        deg: int, degree of SH (0-3)
        sh: (N, (deg+1)^2, 3) SH coefficients
        dirs: (N, 3) unit direction vectors (viewing directions)
    
    Returns:
        colors: (N, 3) RGB colors
    """
    assert deg <= 3, "Only support SH up to degree 3"
    
    N = dirs.shape[0]
    result = np.zeros((N, 3))
    
    # Extract direction components
    x = dirs[:, 0:1]  # (N, 1) - keep dimension for proper broadcasting
    y = dirs[:, 1:2]  # (N, 1)
    z = dirs[:, 2:3]  # (N, 1)
    
    # Constants for SH evaluation
    C0 = 0.28209479177387814  # 1 / (2 * sqrt(pi))
    
    # Degree 0 (DC component)
    result += C0 * sh[:, 0, :]  # (N, 3)
    
    if deg > 0:
        # Degree 1
        C1 = 0.4886025119029199  # sqrt(3) / (2 * sqrt(pi))
        
        result += -C1 * y * sh[:, 1, :]  # -Y_1^{-1}
        result += C1 * z * sh[:, 2, :]   # Y_1^0
        result += -C1 * x * sh[:, 3, :]  # -Y_1^1
    
    if deg > 1:
        # Degree 2
        C2_0 = 1.0925484305920792   # sqrt(15) / (2 * sqrt(pi))
        C2_1 = -1.0925484305920792  # -sqrt(15) / (2 * sqrt(pi))
        C2_2 = 0.31539156525252005   # sqrt(5) / (4 * sqrt(pi))
        C2_3 = -1.0925484305920792  # -sqrt(15) / (2 * sqrt(pi))
        C2_4 = 0.5462742152960396    # sqrt(15) / (4 * sqrt(pi))
        
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        
        result += C2_0 * xy * sh[:, 4, :]                 # Y_2^{-2}
        result += C2_1 * yz * sh[:, 5, :]                 # Y_2^{-1}
        result += C2_2 * (2.0 * zz - xx - yy) * sh[:, 6, :] # Y_2^0
        result += C2_3 * xz * sh[:, 7, :]                 # Y_2^1
        result += C2_4 * (xx - yy) * sh[:, 8, :]          # Y_2^2
    
    if deg > 2:
        # Degree 3
        C3_0 = -0.5900435899266435   # -sqrt(70) / (8 * sqrt(pi))
        C3_1 = 2.890611442640554     # sqrt(105) / (2 * sqrt(pi))
        C3_2 = -0.4570457994644658   # -sqrt(42) / (8 * sqrt(pi))
        C3_3 = 0.3731763325901154    # sqrt(7) / (4 * sqrt(pi))
        C3_4 = -0.4570457994644658   # -sqrt(42) / (8 * sqrt(pi))
        C3_5 = 1.445305721320277     # sqrt(105) / (4 * sqrt(pi))
        C3_6 = -0.5900435899266435   # -sqrt(70) / (8 * sqrt(pi))
        
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        
        result += C3_0 * y * (3 * xx - yy) * sh[:, 9, :]           # Y_3^{-3}
        result += C3_1 * xy * z * sh[:, 10, :]                     # Y_3^{-2}
        result += C3_2 * y * (4 * zz - xx - yy) * sh[:, 11, :]    # Y_3^{-1}
        result += C3_3 * z * (2 * zz - 3 * xx - 3 * yy) * sh[:, 12, :] # Y_3^0
        result += C3_4 * x * (4 * zz - xx - yy) * sh[:, 13, :]    # Y_3^1
        result += C3_5 * z * (xx - yy) * sh[:, 14, :]              # Y_3^2
        result += C3_6 * x * (xx - 3 * yy) * sh[:, 15, :]          # Y_3^3
    
    return result


def dnormvdv(v, dL_dnorm_v):
    """Backprop through vector normalization: norm_v = v / ||v||
    
    Args:
        v: (3,) unnormalized vector (dir_orig)
        dL_dnorm_v: (3,) gradient w.r.t. normalized vector
    
    Returns:
        dL_dv: (3,) gradient w.r.t. unnormalized vector
    """
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return np.zeros_like(v)
    v_hat = v / norm
    # d(v/||v||)/dv = (I - v_hat @ v_hat^T) / ||v||
    return (dL_dnorm_v - np.dot(dL_dnorm_v, v_hat) * v_hat) / norm


def compute_sh_backward_single(dir_normalized, sh_coeffs, dL_dRGB, deg=3):
    """Compute SH backward for a single Gaussian: dL/d(dir) via SH polynomial.
    
    Mirrors the CUDA computeColorFromSH backward pass.
    
    Args:
        dir_normalized: (3,) normalized viewing direction
        sh_coeffs: (16, 3) SH coefficients for this Gaussian
        dL_dRGB: (3,) gradient of loss w.r.t. this Gaussian's color (with clamping applied)
        deg: int - SH degree
    
    Returns:
        dL_ddir: (3,) gradient of loss w.r.t. normalized direction
    """
    x, y, z = dir_normalized
    
    # SH constants (must match forward pass exactly)
    SH_C1 = 0.4886025119029199
    SH_C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
             -1.0925484305920792, 0.5462742152960396]
    SH_C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
             0.3731763325901154, -0.4570457994644658, 1.445305721320277,
             -0.5900435899266435]
    
    sh = sh_coeffs  # (16, 3)
    
    # Accumulate dRGB/d(x,y,z) - each is a (3,) vector for the 3 color channels
    dRGBdx = np.zeros(3, dtype=np.float64)
    dRGBdy = np.zeros(3, dtype=np.float64)
    dRGBdz = np.zeros(3, dtype=np.float64)
    
    if deg > 0:
        dRGBdx += -SH_C1 * sh[3]
        dRGBdy += -SH_C1 * sh[1]
        dRGBdz += SH_C1 * sh[2]
        
        if deg > 1:
            xx, yy, zz = x*x, y*y, z*z
            xy, yz, xz = x*y, y*z, x*z
            
            dRGBdx += (SH_C2[0] * y * sh[4] + SH_C2[2] * 2.0 * (-x) * sh[6] +
                       SH_C2[3] * z * sh[7] + SH_C2[4] * 2.0 * x * sh[8])
            dRGBdy += (SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] +
                       SH_C2[2] * 2.0 * (-y) * sh[6] + SH_C2[4] * 2.0 * (-y) * sh[8])
            dRGBdz += (SH_C2[1] * y * sh[5] + SH_C2[2] * 2.0 * 2.0 * z * sh[6] +
                       SH_C2[3] * x * sh[7])
            
            if deg > 2:
                dRGBdx += (
                    SH_C3[0] * sh[9] * 3.0 * 2.0 * xy +
                    SH_C3[1] * sh[10] * yz +
                    SH_C3[2] * sh[11] * (-2.0 * xy) +
                    SH_C3[3] * sh[12] * (-3.0 * 2.0 * xz) +
                    SH_C3[4] * sh[13] * (-3.0 * xx + 4.0 * zz - yy) +
                    SH_C3[5] * sh[14] * 2.0 * xz +
                    SH_C3[6] * sh[15] * 3.0 * (xx - yy))
                
                dRGBdy += (
                    SH_C3[0] * sh[9] * 3.0 * (xx - yy) +
                    SH_C3[1] * sh[10] * xz +
                    SH_C3[2] * sh[11] * (-3.0 * yy + 4.0 * zz - xx) +
                    SH_C3[3] * sh[12] * (-3.0 * 2.0 * yz) +
                    SH_C3[4] * sh[13] * (-2.0 * xy) +
                    SH_C3[5] * sh[14] * (-2.0 * yz) +
                    SH_C3[6] * sh[15] * (-3.0 * 2.0 * xy))
                
                dRGBdz += (
                    SH_C3[1] * sh[10] * xy +
                    SH_C3[2] * sh[11] * 4.0 * 2.0 * yz +
                    SH_C3[3] * sh[12] * 3.0 * (2.0 * zz - xx - yy) +
                    SH_C3[4] * sh[13] * 4.0 * 2.0 * xz +
                    SH_C3[5] * sh[14] * (xx - yy))
    
    # dL/d(dir) = [dRGBdx · dL_dRGB, dRGBdy · dL_dRGB, dRGBdz · dL_dRGB]
    dL_ddir = np.array([
        np.dot(dRGBdx, dL_dRGB),
        np.dot(dRGBdy, dL_dRGB),
        np.dot(dRGBdz, dL_dRGB)
    ])
    
    return dL_ddir


def compute_colors_from_sh(coeffs, viewdirs, deg=3):
    """
    Compute RGB colors from SH coefficients and viewing directions.
    
    Args:
        features_dc: (N, 1, 3) - DC component
        features_rest: (N, 15, 3) - Higher order SH coefficients
        viewdirs: (N, 3) - Viewing directions (unit vectors)
        deg: int - SH degree (0-3)
    
    Returns:
        colors: (N, 3) - RGB colors in range [0, 1]
    """
    N = coeffs.shape[0]
    
    # Combine DC and rest into single array
    sh_coeffs = coeffs  # (N, 16, 3)
    
    # Normalize viewing directions
    # viewdirs_normalized = viewdirs / (np.linalg.norm(viewdirs, axis=1, keepdims=True) + 1e-8)
    
    # Evaluate SH
    colors = eval_sh(deg, sh_coeffs, viewdirs)  # (N, 3)
    
    # Add 0.5 and clamp to [0, 1]
    # 3DGS uses this offset for better color range
    colors = colors + 0.5
    # colors = np.clip(colors, 0.0, 1.0)

    colors = np.maximum(colors, 0.0)
    
    return colors


def compute_viewing_directions(gaussian_positions, camera_position):
    """
    Compute viewing direction from each Gaussian to camera.
    IMPORTANT: Both inputs must be in the SAME coordinate frame (typically world frame).
    
    Args:
        gaussian_positions: (N, 3) - Gaussian centers (world coords for SH evaluation)
        camera_position: (3,) - Camera position (world coords)
    
    Returns:
        viewdirs: (N, 3) - Normalized viewing directions (Gaussian -> Camera)
    """
    # Direction from Gaussian to camera
    viewdirs = gaussian_positions - camera_position[None, :] # (N, 3)
    
    # Normalize
    norms = np.linalg.norm(viewdirs, axis=1, keepdims=True) + 1e-8
    viewdirs = viewdirs / norms
    
    return viewdirs


def pi_func(v):
    '''
    projection function
    v is in homogenous coords
    '''
    # I34 = np.eye(3,4)
    # v_proj=np.divide(v,v[2])
    pi = (np.matmul(np.eye(3,4), v))/v[2]
    # print(v_proj)
    return pi 

def Jacobian_pi_func(v):

    a = np.array([[1, 0, -v[0]/v[2]],
                  [0, 1, -v[1]/v[2]],
                  [0, 0,  0]])
    a = a/v[2]
    return a

def hat_operator(v):
	'''
	Hat operator
	'''
	return np.array([[0    ,-v[2],  v[1]],
                     [v[2] ,    0, -v[0]],
                     [-v[1], v[0],    0]])

def Get_dcovI_dJ(R,cov_3D,mu_c):

    x = mu_c[0]
    y = mu_c[1]
    z = mu_c[2]
    A , B , C , D ,E ,F = cov_3D[0,0], cov_3D[0,1], cov_3D[0,2], cov_3D[1,1], cov_3D[1,2], cov_3D[2,2]
    R_00 ,R_01 ,R_02 ,R_10 ,R_11 ,R_12 ,R_20 ,R_21 ,R_22 = R[0,0], R[0,1], R[0,2], R[1,0], R[1,1], R[1,2], R[2,0], R[2,1], R[2,2]
    
    
    temp = np.array([[R_00*(A*(R_00/z - R_20*x/z**2) + B*(R_01/z - R_21*x/z**2) + C*(R_02/z - R_22*x/z**2)) + R_01*(B*(R_00/z - R_20*x/z**2) + D*(R_01/z - R_21*x/z**2) + E*(R_02/z - R_22*x/z**2)) + R_02*(C*(R_00/z - R_20*x/z**2) + E*(R_01/z - R_21*x/z**2) + F*(R_02/z - R_22*x/z**2)) - x*(R_20*(A*R_00 + B*R_01 + C*R_02) + R_21*(B*R_00 + D*R_01 + E*R_02) + R_22*(C*R_00 + E*R_01 + F*R_02))/z**2 + (R_00*(A*R_00 + B*R_01 + C*R_02) + R_01*(B*R_00 + D*R_01 + E*R_02) + R_02*(C*R_00 + E*R_01 + F*R_02))/z, R_10*(A*(R_00/z - R_20*x/z**2) + B*(R_01/z - R_21*x/z**2) + C*(R_02/z - R_22*x/z**2)) + R_11*(B*(R_00/z - R_20*x/z**2) + D*(R_01/z - R_21*x/z**2) + E*(R_02/z - R_22*x/z**2)) + R_12*(C*(R_00/z - R_20*x/z**2) + E*(R_01/z - R_21*x/z**2) + F*(R_02/z - R_22*x/z**2)) - x*(R_20*(A*R_10 + B*R_11 + C*R_12) + R_21*(B*R_10 + D*R_11 + E*R_12) + R_22*(C*R_10 + E*R_11 + F*R_12))/z**2 + (R_00*(A*R_10 + B*R_11 + C*R_12) + R_01*(B*R_10 + D*R_11 + E*R_12) + R_02*(C*R_10 + E*R_11 + F*R_12))/z, R_20*(A*(R_00/z - R_20*x/z**2) + B*(R_01/z - R_21*x/z**2) + C*(R_02/z - R_22*x/z**2)) + R_21*(B*(R_00/z - R_20*x/z**2) + D*(R_01/z - R_21*x/z**2) + E*(R_02/z - R_22*x/z**2)) + R_22*(C*(R_00/z - R_20*x/z**2) + E*(R_01/z - R_21*x/z**2) + F*(R_02/z - R_22*x/z**2)) - x*(R_20*(A*R_20 + B*R_21 + C*R_22) + R_21*(B*R_20 + D*R_21 + E*R_22) + R_22*(C*R_20 + E*R_21 + F*R_22))/z**2 + (R_00*(A*R_20 + B*R_21 + C*R_22) + R_01*(B*R_20 + D*R_21 + E*R_22) + R_02*(C*R_20 + E*R_21 + F*R_22))/z, 0, 0, 0], 
                     [-y*(R_20*(A*R_00 + B*R_01 + C*R_02) + R_21*(B*R_00 + D*R_01 + E*R_02) + R_22*(C*R_00 + E*R_01 + F*R_02))/z**2 + (R_10*(A*R_00 + B*R_01 + C*R_02) + R_11*(B*R_00 + D*R_01 + E*R_02) + R_12*(C*R_00 + E*R_01 + F*R_02))/z, -y*(R_20*(A*R_10 + B*R_11 + C*R_12) + R_21*(B*R_10 + D*R_11 + E*R_12) + R_22*(C*R_10 + E*R_11 + F*R_12))/z**2 + (R_10*(A*R_10 + B*R_11 + C*R_12) + R_11*(B*R_10 + D*R_11 + E*R_12) + R_12*(C*R_10 + E*R_11 + F*R_12))/z, -y*(R_20*(A*R_20 + B*R_21 + C*R_22) + R_21*(B*R_20 + D*R_21 + E*R_22) + R_22*(C*R_20 + E*R_21 + F*R_22))/z**2 + (R_10*(A*R_20 + B*R_21 + C*R_22) + R_11*(B*R_20 + D*R_21 + E*R_22) + R_12*(C*R_20 + E*R_21 + F*R_22))/z, R_00*(A*(R_00/z - R_20*x/z**2) + B*(R_01/z - R_21*x/z**2) + C*(R_02/z - R_22*x/z**2)) + R_01*(B*(R_00/z - R_20*x/z**2) + D*(R_01/z - R_21*x/z**2) + E*(R_02/z - R_22*x/z**2)) + R_02*(C*(R_00/z - R_20*x/z**2) + E*(R_01/z - R_21*x/z**2) + F*(R_02/z - R_22*x/z**2)), R_10*(A*(R_00/z - R_20*x/z**2) + B*(R_01/z - R_21*x/z**2) + C*(R_02/z - R_22*x/z**2)) + R_11*(B*(R_00/z - R_20*x/z**2) + D*(R_01/z - R_21*x/z**2) + E*(R_02/z - R_22*x/z**2)) + R_12*(C*(R_00/z - R_20*x/z**2) + E*(R_01/z - R_21*x/z**2) + F*(R_02/z - R_22*x/z**2)), R_20*(A*(R_00/z - R_20*x/z**2) + B*(R_01/z - R_21*x/z**2) + C*(R_02/z - R_22*x/z**2)) + R_21*(B*(R_00/z - R_20*x/z**2) + D*(R_01/z - R_21*x/z**2) + E*(R_02/z - R_22*x/z**2)) + R_22*(C*(R_00/z - R_20*x/z**2) + E*(R_01/z - R_21*x/z**2) + F*(R_02/z - R_22*x/z**2))], 
                     [R_00*(A*(R_10/z - R_20*y/z**2) + B*(R_11/z - R_21*y/z**2) + C*(R_12/z - R_22*y/z**2)) + R_01*(B*(R_10/z - R_20*y/z**2) + D*(R_11/z - R_21*y/z**2) + E*(R_12/z - R_22*y/z**2)) + R_02*(C*(R_10/z - R_20*y/z**2) + E*(R_11/z - R_21*y/z**2) + F*(R_12/z - R_22*y/z**2)), R_10*(A*(R_10/z - R_20*y/z**2) + B*(R_11/z - R_21*y/z**2) + C*(R_12/z - R_22*y/z**2)) + R_11*(B*(R_10/z - R_20*y/z**2) + D*(R_11/z - R_21*y/z**2) + E*(R_12/z - R_22*y/z**2)) + R_12*(C*(R_10/z - R_20*y/z**2) + E*(R_11/z - R_21*y/z**2) + F*(R_12/z - R_22*y/z**2)), R_20*(A*(R_10/z - R_20*y/z**2) + B*(R_11/z - R_21*y/z**2) + C*(R_12/z - R_22*y/z**2)) + R_21*(B*(R_10/z - R_20*y/z**2) + D*(R_11/z - R_21*y/z**2) + E*(R_12/z - R_22*y/z**2)) + R_22*(C*(R_10/z - R_20*y/z**2) + E*(R_11/z - R_21*y/z**2) + F*(R_12/z - R_22*y/z**2)), -x*(R_20*(A*R_00 + B*R_01 + C*R_02) + R_21*(B*R_00 + D*R_01 + E*R_02) + R_22*(C*R_00 + E*R_01 + F*R_02))/z**2 + (R_00*(A*R_00 + B*R_01 + C*R_02) + R_01*(B*R_00 + D*R_01 + E*R_02) + R_02*(C*R_00 + E*R_01 + F*R_02))/z, -x*(R_20*(A*R_10 + B*R_11 + C*R_12) + R_21*(B*R_10 + D*R_11 + E*R_12) + R_22*(C*R_10 + E*R_11 + F*R_12))/z**2 + (R_00*(A*R_10 + B*R_11 + C*R_12) + R_01*(B*R_10 + D*R_11 + E*R_12) + R_02*(C*R_10 + E*R_11 + F*R_12))/z, -x*(R_20*(A*R_20 + B*R_21 + C*R_22) + R_21*(B*R_20 + D*R_21 + E*R_22) + R_22*(C*R_20 + E*R_21 + F*R_22))/z**2 + (R_00*(A*R_20 + B*R_21 + C*R_22) + R_01*(B*R_20 + D*R_21 + E*R_22) + R_02*(C*R_20 + E*R_21 + F*R_22))/z], 
                     [0, 0, 0, R_00*(A*(R_10/z - R_20*y/z**2) + B*(R_11/z - R_21*y/z**2) + C*(R_12/z - R_22*y/z**2)) + R_01*(B*(R_10/z - R_20*y/z**2) + D*(R_11/z - R_21*y/z**2) + E*(R_12/z - R_22*y/z**2)) + R_02*(C*(R_10/z - R_20*y/z**2) + E*(R_11/z - R_21*y/z**2) + F*(R_12/z - R_22*y/z**2)) - y*(R_20*(A*R_00 + B*R_01 + C*R_02) + R_21*(B*R_00 + D*R_01 + E*R_02) + R_22*(C*R_00 + E*R_01 + F*R_02))/z**2 + (R_10*(A*R_00 + B*R_01 + C*R_02) + R_11*(B*R_00 + D*R_01 + E*R_02) + R_12*(C*R_00 + E*R_01 + F*R_02))/z, R_10*(A*(R_10/z - R_20*y/z**2) + B*(R_11/z - R_21*y/z**2) + C*(R_12/z - R_22*y/z**2)) + R_11*(B*(R_10/z - R_20*y/z**2) + D*(R_11/z - R_21*y/z**2) + E*(R_12/z - R_22*y/z**2)) + R_12*(C*(R_10/z - R_20*y/z**2) + E*(R_11/z - R_21*y/z**2) + F*(R_12/z - R_22*y/z**2)) - y*(R_20*(A*R_10 + B*R_11 + C*R_12) + R_21*(B*R_10 + D*R_11 + E*R_12) + R_22*(C*R_10 + E*R_11 + F*R_12))/z**2 + (R_10*(A*R_10 + B*R_11 + C*R_12) + R_11*(B*R_10 + D*R_11 + E*R_12) + R_12*(C*R_10 + E*R_11 + F*R_12))/z, R_20*(A*(R_10/z - R_20*y/z**2) + B*(R_11/z - R_21*y/z**2) + C*(R_12/z - R_22*y/z**2)) + R_21*(B*(R_10/z - R_20*y/z**2) + D*(R_11/z - R_21*y/z**2) + E*(R_12/z - R_22*y/z**2)) + R_22*(C*(R_10/z - R_20*y/z**2) + E*(R_11/z - R_21*y/z**2) + F*(R_12/z - R_22*y/z**2)) - y*(R_20*(A*R_20 + B*R_21 + C*R_22) + R_21*(B*R_20 + D*R_21 + E*R_22) + R_22*(C*R_20 + E*R_21 + F*R_22))/z**2 + (R_10*(A*R_20 + B*R_21 + C*R_22) + R_11*(B*R_20 + D*R_21 + E*R_22) + R_12*(C*R_20 + E*R_21 + F*R_22))/z]])
    return temp

def GetAnalyticalJcobian(T_cw, mu_w, cov_3D, fx, fy):
    '''
    T_cw : Camera pose in world frame
    mu_w : Mean of 3D gaussian
    '''
    R = T_cw[0:3,0:3]
    
    mu_c = np.matmul(T_cw,mu_w)
    # print("mu_c:\n", mu_c)
    
    a = 1/mu_c[2]
    b = -mu_c[0]/(mu_c[2]**2)
    c = -mu_c[1]/(mu_c[2]**2)
    
    dmuI_dmuC = np.array([[a , 0 , b],
                          [0,  a , c]])
    # print("dmuI_dmuC: \n", dmuI_dmuC)
    
    dmuC_dTcw = np.hstack((np.eye(3),-hat_operator(mu_c)))
    # print("dmuC_dTcw: \n", dmuC_dTcw)
    
    dmuI_dTcw = np.matmul(dmuI_dmuC,dmuC_dTcw) # Equation 3
    # print("dmuI_dTcw: \n", dmuI_dTcw)
    
    # 2nd term of eqution 4 
    
    dW_dTcw = np.zeros([9,6])
    dW_dTcw[0:3,3:6] = -hat_operator(R[:,0])
    dW_dTcw[3:6,3:6] = -hat_operator(R[:,1])
    dW_dTcw[6:9,3:6] = -hat_operator(R[:,2])
    
    # print("dW_dTcw: \n",dW_dTcw)
    
    # dmuI_dW
    dcovI_dW = np.array([[a*(cov_3D[0,0]*R[0,0]*a + cov_3D[0,0]*(R[0,0]*a + R[2,0]*b) + cov_3D[0,1]*R[0,1]*a + cov_3D[0,1]*(R[0,1]*a + R[2,1]*b) + cov_3D[0,2]*R[0,2]*a + cov_3D[0,2]*(R[0,2]*a + R[2,2]*b)) + b*(cov_3D[0,0]*R[2,0]*a + cov_3D[0,1]*R[2,1]*a + cov_3D[0,2]*R[2,2]*a), 0, a*(cov_3D[0,0]*R[0,0]*b + cov_3D[0,1]*R[0,1]*b + cov_3D[0,2]*R[0,2]*b) + b*(cov_3D[0,0]*R[2,0]*b + cov_3D[0,0]*(R[0,0]*a + R[2,0]*b) + cov_3D[0,1]*R[2,1]*b + cov_3D[0,1]*(R[0,1]*a + R[2,1]*b) + cov_3D[0,2]*R[2,2]*b + cov_3D[0,2]*(R[0,2]*a + R[2,2]*b)), a*(cov_3D[0,1]*R[0,0]*a + cov_3D[0,1]*(R[0,0]*a + R[2,0]*b) + cov_3D[1,1]*R[0,1]*a + cov_3D[1,1]*(R[0,1]*a + R[2,1]*b) + cov_3D[1,2]*R[0,2]*a + cov_3D[1,2]*(R[0,2]*a + R[2,2]*b)) + b*(cov_3D[0,1]*R[2,0]*a + cov_3D[1,1]*R[2,1]*a + cov_3D[1,2]*R[2,2]*a), 0, a*(cov_3D[0,1]*R[0,0]*b + cov_3D[1,1]*R[0,1]*b + cov_3D[1,2]*R[0,2]*b) + b*(cov_3D[0,1]*R[2,0]*b + cov_3D[0,1]*(R[0,0]*a + R[2,0]*b) + cov_3D[1,1]*R[2,1]*b + cov_3D[1,1]*(R[0,1]*a + R[2,1]*b) + cov_3D[1,2]*R[2,2]*b + cov_3D[1,2]*(R[0,2]*a + R[2,2]*b)), a*(cov_3D[0,2]*R[0,0]*a + cov_3D[0,2]*(R[0,0]*a + R[2,0]*b) + cov_3D[1,2]*R[0,1]*a + cov_3D[1,2]*(R[0,1]*a + R[2,1]*b) + cov_3D[2,2]*R[0,2]*a + cov_3D[2,2]*(R[0,2]*a + R[2,2]*b)) + b*(cov_3D[0,2]*R[2,0]*a + cov_3D[1,2]*R[2,1]*a + cov_3D[2,2]*R[2,2]*a), 0, a*(cov_3D[0,2]*R[0,0]*b + cov_3D[1,2]*R[0,1]*b + cov_3D[2,2]*R[0,2]*b) + b*(cov_3D[0,2]*R[2,0]*b + cov_3D[0,2]*(R[0,0]*a + R[2,0]*b) + cov_3D[1,2]*R[2,1]*b + cov_3D[1,2]*(R[0,1]*a + R[2,1]*b) + cov_3D[2,2]*R[2,2]*b + cov_3D[2,2]*(R[0,2]*a + R[2,2]*b))],
                         [a*(cov_3D[0,0]*R[1,0]*a + cov_3D[0,1]*R[1,1]*a + cov_3D[0,2]*R[1,2]*a) + c*(cov_3D[0,0]*R[2,0]*a + cov_3D[0,1]*R[2,1]*a + cov_3D[0,2]*R[2,2]*a), a*(cov_3D[0,0]*(R[0,0]*a + R[2,0]*b) + cov_3D[0,1]*(R[0,1]*a + R[2,1]*b) + cov_3D[0,2]*(R[0,2]*a + R[2,2]*b)), a*(cov_3D[0,0]*R[1,0]*b + cov_3D[0,1]*R[1,1]*b + cov_3D[0,2]*R[1,2]*b) + c*(cov_3D[0,0]*R[2,0]*b + cov_3D[0,0]*(R[0,0]*a + R[2,0]*b) + cov_3D[0,1]*R[2,1]*b + cov_3D[0,1]*(R[0,1]*a + R[2,1]*b) + cov_3D[0,2]*R[2,2]*b + cov_3D[0,2]*(R[0,2]*a + R[2,2]*b)), a*(cov_3D[0,1]*R[1,0]*a + cov_3D[1,1]*R[1,1]*a + cov_3D[1,2]*R[1,2]*a) + c*(cov_3D[0,1]*R[2,0]*a + cov_3D[1,1]*R[2,1]*a + cov_3D[1,2]*R[2,2]*a), a*(cov_3D[0,1]*(R[0,0]*a + R[2,0]*b) + cov_3D[1,1]*(R[0,1]*a + R[2,1]*b) + cov_3D[1,2]*(R[0,2]*a + R[2,2]*b)), a*(cov_3D[0,1]*R[1,0]*b + cov_3D[1,1]*R[1,1]*b + cov_3D[1,2]*R[1,2]*b) + c*(cov_3D[0,1]*R[2,0]*b + cov_3D[0,1]*(R[0,0]*a + R[2,0]*b) + cov_3D[1,1]*R[2,1]*b + cov_3D[1,1]*(R[0,1]*a + R[2,1]*b) + cov_3D[1,2]*R[2,2]*b + cov_3D[1,2]*(R[0,2]*a + R[2,2]*b)), a*(cov_3D[0,2]*R[1,0]*a + cov_3D[1,2]*R[1,1]*a + cov_3D[2,2]*R[1,2]*a) + c*(cov_3D[0,2]*R[2,0]*a + cov_3D[1,2]*R[2,1]*a + cov_3D[2,2]*R[2,2]*a), a*(cov_3D[0,2]*(R[0,0]*a + R[2,0]*b) + cov_3D[1,2]*(R[0,1]*a + R[2,1]*b) + cov_3D[2,2]*(R[0,2]*a + R[2,2]*b)), a*(cov_3D[0,2]*R[1,0]*b + cov_3D[1,2]*R[1,1]*b + cov_3D[2,2]*R[1,2]*b) + c*(cov_3D[0,2]*R[2,0]*b + cov_3D[0,2]*(R[0,0]*a + R[2,0]*b) + cov_3D[1,2]*R[2,1]*b + cov_3D[1,2]*(R[0,1]*a + R[2,1]*b) + cov_3D[2,2]*R[2,2]*b + cov_3D[2,2]*(R[0,2]*a + R[2,2]*b))],
                         [a*(cov_3D[0,0]*(R[1,0]*a + R[2,0]*c) + cov_3D[0,1]*(R[1,1]*a + R[2,1]*c) + cov_3D[0,2]*(R[1,2]*a + R[2,2]*c)), a*(cov_3D[0,0]*R[0,0]*a + cov_3D[0,1]*R[0,1]*a + cov_3D[0,2]*R[0,2]*a) + b*(cov_3D[0,0]*R[2,0]*a + cov_3D[0,1]*R[2,1]*a + cov_3D[0,2]*R[2,2]*a), a*(cov_3D[0,0]*R[0,0]*c + cov_3D[0,1]*R[0,1]*c + cov_3D[0,2]*R[0,2]*c) + b*(cov_3D[0,0]*R[2,0]*c + cov_3D[0,0]*(R[1,0]*a + R[2,0]*c) + cov_3D[0,1]*R[2,1]*c + cov_3D[0,1]*(R[1,1]*a + R[2,1]*c) + cov_3D[0,2]*R[2,2]*c + cov_3D[0,2]*(R[1,2]*a + R[2,2]*c)), a*(cov_3D[0,1]*(R[1,0]*a + R[2,0]*c) + cov_3D[1,1]*(R[1,1]*a + R[2,1]*c) + cov_3D[1,2]*(R[1,2]*a + R[2,2]*c)), a*(cov_3D[0,1]*R[0,0]*a + cov_3D[1,1]*R[0,1]*a + cov_3D[1,2]*R[0,2]*a) + b*(cov_3D[0,1]*R[2,0]*a + cov_3D[1,1]*R[2,1]*a + cov_3D[1,2]*R[2,2]*a), a*(cov_3D[0,1]*R[0,0]*c + cov_3D[1,1]*R[0,1]*c + cov_3D[1,2]*R[0,2]*c) + b*(cov_3D[0,1]*R[2,0]*c + cov_3D[0,1]*(R[1,0]*a + R[2,0]*c) + cov_3D[1,1]*R[2,1]*c + cov_3D[1,1]*(R[1,1]*a + R[2,1]*c) + cov_3D[1,2]*R[2,2]*c + cov_3D[1,2]*(R[1,2]*a + R[2,2]*c)), a*(cov_3D[0,2]*(R[1,0]*a + R[2,0]*c) + cov_3D[1,2]*(R[1,1]*a + R[2,1]*c) + cov_3D[2,2]*(R[1,2]*a + R[2,2]*c)), a*(cov_3D[0,2]*R[0,0]*a + cov_3D[1,2]*R[0,1]*a + cov_3D[2,2]*R[0,2]*a) + b*(cov_3D[0,2]*R[2,0]*a + cov_3D[1,2]*R[2,1]*a + cov_3D[2,2]*R[2,2]*a), a*(cov_3D[0,2]*R[0,0]*c + cov_3D[1,2]*R[0,1]*c + cov_3D[2,2]*R[0,2]*c) + b*(cov_3D[0,2]*R[2,0]*c + cov_3D[0,2]*(R[1,0]*a + R[2,0]*c) + cov_3D[1,2]*R[2,1]*c + cov_3D[1,2]*(R[1,1]*a + R[2,1]*c) + cov_3D[2,2]*R[2,2]*c + cov_3D[2,2]*(R[1,2]*a + R[2,2]*c))],
                         [0, a*(cov_3D[0,0]*R[1,0]*a + cov_3D[0,0]*(R[1,0]*a + R[2,0]*c) + cov_3D[0,1]*R[1,1]*a + cov_3D[0,1]*(R[1,1]*a + R[2,1]*c) + cov_3D[0,2]*R[1,2]*a + cov_3D[0,2]*(R[1,2]*a + R[2,2]*c)) + c*(cov_3D[0,0]*R[2,0]*a + cov_3D[0,1]*R[2,1]*a + cov_3D[0,2]*R[2,2]*a), a*(cov_3D[0,0]*R[1,0]*c + cov_3D[0,1]*R[1,1]*c + cov_3D[0,2]*R[1,2]*c) + c*(cov_3D[0,0]*R[2,0]*c + cov_3D[0,0]*(R[1,0]*a + R[2,0]*c) + cov_3D[0,1]*R[2,1]*c + cov_3D[0,1]*(R[1,1]*a + R[2,1]*c) + cov_3D[0,2]*R[2,2]*c + cov_3D[0,2]*(R[1,2]*a + R[2,2]*c)), 0, a*(cov_3D[0,1]*R[1,0]*a + cov_3D[0,1]*(R[1,0]*a + R[2,0]*c) + cov_3D[1,1]*R[1,1]*a + cov_3D[1,1]*(R[1,1]*a + R[2,1]*c) + cov_3D[1,2]*R[1,2]*a + cov_3D[1,2]*(R[1,2]*a + R[2,2]*c)) + c*(cov_3D[0,1]*R[2,0]*a + cov_3D[1,1]*R[2,1]*a + cov_3D[1,2]*R[2,2]*a), a*(cov_3D[0,1]*R[1,0]*c + cov_3D[1,1]*R[1,1]*c + cov_3D[1,2]*R[1,2]*c) + c*(cov_3D[0,1]*R[2,0]*c + cov_3D[0,1]*(R[1,0]*a + R[2,0]*c) + cov_3D[1,1]*R[2,1]*c + cov_3D[1,1]*(R[1,1]*a + R[2,1]*c) + cov_3D[1,2]*R[2,2]*c + cov_3D[1,2]*(R[1,2]*a + R[2,2]*c)), 0, a*(cov_3D[0,2]*R[1,0]*a + cov_3D[0,2]*(R[1,0]*a + R[2,0]*c) + cov_3D[1,2]*R[1,1]*a + cov_3D[1,2]*(R[1,1]*a + R[2,1]*c) + cov_3D[2,2]*R[1,2]*a + cov_3D[2,2]*(R[1,2]*a + R[2,2]*c)) + c*(cov_3D[0,2]*R[2,0]*a + cov_3D[1,2]*R[2,1]*a + cov_3D[2,2]*R[2,2]*a), a*(cov_3D[0,2]*R[1,0]*c + cov_3D[1,2]*R[1,1]*c + cov_3D[2,2]*R[1,2]*c) + c*(cov_3D[0,2]*R[2,0]*c + cov_3D[0,2]*(R[1,0]*a + R[2,0]*c) + cov_3D[1,2]*R[2,1]*c + cov_3D[1,2]*(R[1,1]*a + R[2,1]*c) + cov_3D[2,2]*R[2,2]*c + cov_3D[2,2]*(R[1,2]*a + R[2,2]*c))]])
    # print("dcovI_dW: \n", dcovI_dW)
    
    second_term = np.matmul(dcovI_dW,dW_dTcw)
    # print(second_term)
    
    
    # First term of eqution 4
    # dcovI_dJ = np.array([[R[0,0]*(cov_3D[0,0]*(R[0,0]/mu_c[2] - R[2,0]*mu_c[0]/mu_c[2]**2) + cov_3D[0,1]*(R[0,1]/mu_c[2] - R[2,1]*mu_c[0]/mu_c[2]**2) + cov_3D[0,2]*(R[0,2]/mu_c[2] - R[2,2]*mu_c[0]/mu_c[2]**2)) + R[0,1]*(cov_3D[0,1]*(R[0,0]/mu_c[2] - R[2,0]*mu_c[0]/mu_c[2]**2) + cov_3D[1,1]*(R[0,1]/mu_c[2] - R[2,1]*mu_c[0]/mu_c[2]**2) + cov_3D[1,2]*(R[0,2]/mu_c[2] - R[2,2]*mu_c[0]/mu_c[2]**2)) + R[0,2]*(cov_3D[0,2]*(R[0,0]/mu_c[2] - R[2,0]*mu_c[0]/mu_c[2]**2) + cov_3D[1,2]*(R[0,1]/mu_c[2] - R[2,1]*mu_c[0]/mu_c[2]**2) + cov_3D[2,2]*(R[0,2]/mu_c[2] - R[2,2]*mu_c[0]/mu_c[2]**2)) - mu_c[0]*(R[2,0]*(cov_3D[0,0]*R[0,0] + cov_3D[0,1]*R[0,1] + cov_3D[0,2]*R[0,2]) + R[2,1]*(cov_3D[0,1]*R[0,0] + cov_3D[1,1]*R[0,1] + cov_3D[1,2]*R[0,2]) + R[2,2]*(cov_3D[0,2]*R[0,0] + cov_3D[1,2]*R[0,1] + cov_3D[2,2]*R[0,2]))/mu_c[2]**2 + (R[0,0]*(cov_3D[0,0]*R[0,0] + cov_3D[0,1]*R[0,1] + cov_3D[0,2]*R[0,2]) + R[0,1]*(cov_3D[0,1]*R[0,0] + cov_3D[1,1]*R[0,1] + cov_3D[1,2]*R[0,2]) + R[0,2]*(cov_3D[0,2]*R[0,0] + cov_3D[1,2]*R[0,1] + cov_3D[2,2]*R[0,2]))/mu_c[2], 0, R[2,0]*(cov_3D[0,0]*(R[0,0]/mu_c[2] - R[2,0]*mu_c[0]/mu_c[2]**2) + cov_3D[0,1]*(R[0,1]/mu_c[2] - R[2,1]*mu_c[0]/mu_c[2]**2) + cov_3D[0,2]*(R[0,2]/mu_c[2] - R[2,2]*mu_c[0]/mu_c[2]**2)) + R[2,1]*(cov_3D[0,1]*(R[0,0]/mu_c[2] - R[2,0]*mu_c[0]/mu_c[2]**2) + cov_3D[1,1]*(R[0,1]/mu_c[2] - R[2,1]*mu_c[0]/mu_c[2]**2) + cov_3D[1,2]*(R[0,2]/mu_c[2] - R[2,2]*mu_c[0]/mu_c[2]**2)) + R[2,2]*(cov_3D[0,2]*(R[0,0]/mu_c[2] - R[2,0]*mu_c[0]/mu_c[2]**2) + cov_3D[1,2]*(R[0,1]/mu_c[2] - R[2,1]*mu_c[0]/mu_c[2]**2) + cov_3D[2,2]*(R[0,2]/mu_c[2] - R[2,2]*mu_c[0]/mu_c[2]**2)) - mu_c[0]*(R[2,0]*(cov_3D[0,0]*R[2,0] + cov_3D[0,1]*R[2,1] + cov_3D[0,2]*R[2,2]) + R[2,1]*(cov_3D[0,1]*R[2,0] + cov_3D[1,1]*R[2,1] + cov_3D[1,2]*R[2,2]) + R[2,2]*(cov_3D[0,2]*R[2,0] + cov_3D[1,2]*R[2,1] + cov_3D[2,2]*R[2,2]))/mu_c[2]**2 + (R[0,0]*(cov_3D[0,0]*R[2,0] + cov_3D[0,1]*R[2,1] + cov_3D[0,2]*R[2,2]) + R[0,1]*(cov_3D[0,1]*R[2,0] + cov_3D[1,1]*R[2,1] + cov_3D[1,2]*R[2,2]) + R[0,2]*(cov_3D[0,2]*R[2,0] + cov_3D[1,2]*R[2,1] + cov_3D[2,2]*R[2,2]))/mu_c[2], 0, R[0,0]*(cov_3D[0,0]*(R[0,0]/mu_c[2] - R[2,0]*mu_c[0]/mu_c[2]**2) + cov_3D[0,1]*(R[0,1]/mu_c[2] - R[2,1]*mu_c[0]/mu_c[2]**2) + cov_3D[0,2]*(R[0,2]/mu_c[2] - R[2,2]*mu_c[0]/mu_c[2]**2)) + R[0,1]*(cov_3D[0,1]*(R[0,0]/mu_c[2] - R[2,0]*mu_c[0]/mu_c[2]**2) + cov_3D[1,1]*(R[0,1]/mu_c[2] - R[2,1]*mu_c[0]/mu_c[2]**2) + cov_3D[1,2]*(R[0,2]/mu_c[2] - R[2,2]*mu_c[0]/mu_c[2]**2)) + R[0,2]*(cov_3D[0,2]*(R[0,0]/mu_c[2] - R[2,0]*mu_c[0]/mu_c[2]**2) + cov_3D[1,2]*(R[0,1]/mu_c[2] - R[2,1]*mu_c[0]/mu_c[2]**2) + cov_3D[2,2]*(R[0,2]/mu_c[2] - R[2,2]*mu_c[0]/mu_c[2]**2)) - mu_c[0]*(R[2,0]*(cov_3D[0,0]*R[0,0] + cov_3D[0,1]*R[0,1] + cov_3D[0,2]*R[0,2]) + R[2,1]*(cov_3D[0,1]*R[0,0] + cov_3D[1,1]*R[0,1] + cov_3D[1,2]*R[0,2]) + R[2,2]*(cov_3D[0,2]*R[0,0] + cov_3D[1,2]*R[0,1] + cov_3D[2,2]*R[0,2]))/mu_c[2]**2 + (R[0,0]*(cov_3D[0,0]*R[0,0] + cov_3D[0,1]*R[0,1] + cov_3D[0,2]*R[0,2]) + R[0,1]*(cov_3D[0,1]*R[0,0] + cov_3D[1,1]*R[0,1] + cov_3D[1,2]*R[0,2]) + R[0,2]*(cov_3D[0,2]*R[0,0] + cov_3D[1,2]*R[0,1] + cov_3D[2,2]*R[0,2]))/mu_c[2], 0],
    #   [R[1,0]*(cov_3D[0,0]*(R[0,0]/mu_c[2] - R[2,0]*mu_c[0]/mu_c[2]**2) + cov_3D[0,1]*(R[0,1]/mu_c[2] - R[2,1]*mu_c[0]/mu_c[2]**2) + cov_3D[0,2]*(R[0,2]/mu_c[2] - R[2,2]*mu_c[0]/mu_c[2]**2)) + R[1,1]*(cov_3D[0,1]*(R[0,0]/mu_c[2] - R[2,0]*mu_c[0]/mu_c[2]**2) + cov_3D[1,1]*(R[0,1]/mu_c[2] - R[2,1]*mu_c[0]/mu_c[2]**2) + cov_3D[1,2]*(R[0,2]/mu_c[2] - R[2,2]*mu_c[0]/mu_c[2]**2)) + R[1,2]*(cov_3D[0,2]*(R[0,0]/mu_c[2] - R[2,0]*mu_c[0]/mu_c[2]**2) + cov_3D[1,2]*(R[0,1]/mu_c[2] - R[2,1]*mu_c[0]/mu_c[2]**2) + cov_3D[2,2]*(R[0,2]/mu_c[2] - R[2,2]*mu_c[0]/mu_c[2]**2)) - mu_c[1]*(R[2,0]*(cov_3D[0,0]*R[0,0] + cov_3D[0,1]*R[0,1] + cov_3D[0,2]*R[0,2]) + R[2,1]*(cov_3D[0,1]*R[0,0] + cov_3D[1,1]*R[0,1] + cov_3D[1,2]*R[0,2]) + R[2,2]*(cov_3D[0,2]*R[0,0] + cov_3D[1,2]*R[0,1] + cov_3D[2,2]*R[0,2]))/mu_c[2]**2 + (R[1,0]*(cov_3D[0,0]*R[0,0] + cov_3D[0,1]*R[0,1] + cov_3D[0,2]*R[0,2]) + R[1,1]*(cov_3D[0,1]*R[0,0] + cov_3D[1,1]*R[0,1] + cov_3D[1,2]*R[0,2]) + R[1,2]*(cov_3D[0,2]*R[0,0] + cov_3D[1,2]*R[0,1] + cov_3D[2,2]*R[0,2]))/mu_c[2], 0, -mu_c[1]*(R[2,0]*(cov_3D[0,0]*R[2,0] + cov_3D[0,1]*R[2,1] + cov_3D[0,2]*R[2,2]) + R[2,1]*(cov_3D[0,1]*R[2,0] + cov_3D[1,1]*R[2,1] + cov_3D[1,2]*R[2,2]) + R[2,2]*(cov_3D[0,2]*R[2,0] + cov_3D[1,2]*R[2,1] + cov_3D[2,2]*R[2,2]))/mu_c[2]**2 + (R[1,0]*(cov_3D[0,0]*R[2,0] + cov_3D[0,1]*R[2,1] + cov_3D[0,2]*R[2,2]) + R[1,1]*(cov_3D[0,1]*R[2,0] + cov_3D[1,1]*R[2,1] + cov_3D[1,2]*R[2,2]) + R[1,2]*(cov_3D[0,2]*R[2,0] + cov_3D[1,2]*R[2,1] + cov_3D[2,2]*R[2,2]))/mu_c[2], 0, R[1,0]*(cov_3D[0,0]*(R[0,0]/mu_c[2] - R[2,0]*mu_c[0]/mu_c[2]**2) + cov_3D[0,1]*(R[0,1]/mu_c[2] - R[2,1]*mu_c[0]/mu_c[2]**2) + cov_3D[0,2]*(R[0,2]/mu_c[2] - R[2,2]*mu_c[0]/mu_c[2]**2)) + R[1,1]*(cov_3D[0,1]*(R[0,0]/mu_c[2] - R[2,0]*mu_c[0]/mu_c[2]**2) + cov_3D[1,1]*(R[0,1]/mu_c[2] - R[2,1]*mu_c[0]/mu_c[2]**2) + cov_3D[1,2]*(R[0,2]/mu_c[2] - R[2,2]*mu_c[0]/mu_c[2]**2)) + R[1,2]*(cov_3D[0,2]*(R[0,0]/mu_c[2] - R[2,0]*mu_c[0]/mu_c[2]**2) + cov_3D[1,2]*(R[0,1]/mu_c[2] - R[2,1]*mu_c[0]/mu_c[2]**2) + cov_3D[2,2]*(R[0,2]/mu_c[2] - R[2,2]*mu_c[0]/mu_c[2]**2)) - mu_c[1]*(R[2,0]*(cov_3D[0,0]*R[0,0] + cov_3D[0,1]*R[0,1] + cov_3D[0,2]*R[0,2]) + R[2,1]*(cov_3D[0,1]*R[0,0] + cov_3D[1,1]*R[0,1] + cov_3D[1,2]*R[0,2]) + R[2,2]*(cov_3D[0,2]*R[0,0] + cov_3D[1,2]*R[0,1] + cov_3D[2,2]*R[0,2]))/mu_c[2]**2 + (R[1,0]*(cov_3D[0,0]*R[0,0] + cov_3D[0,1]*R[0,1] + cov_3D[0,2]*R[0,2]) + R[1,1]*(cov_3D[0,1]*R[0,0] + cov_3D[1,1]*R[0,1] + cov_3D[1,2]*R[0,2]) + R[1,2]*(cov_3D[0,2]*R[0,0] + cov_3D[1,2]*R[0,1] + cov_3D[2,2]*R[0,2]))/mu_c[2], R[2,0]*(cov_3D[0,0]*(R[0,0]/mu_c[2] - R[2,0]*mu_c[0]/mu_c[2]**2) + cov_3D[0,1]*(R[0,1]/mu_c[2] - R[2,1]*mu_c[0]/mu_c[2]**2) + cov_3D[0,2]*(R[0,2]/mu_c[2] - R[2,2]*mu_c[0]/mu_c[2]**2)) + R[2,1]*(cov_3D[0,1]*(R[0,0]/mu_c[2] - R[2,0]*mu_c[0]/mu_c[2]**2) + cov_3D[1,1]*(R[0,1]/mu_c[2] - R[2,1]*mu_c[0]/mu_c[2]**2) + cov_3D[1,2]*(R[0,2]/mu_c[2] - R[2,2]*mu_c[0]/mu_c[2]**2)) + R[2,2]*(cov_3D[0,2]*(R[0,0]/mu_c[2] - R[2,0]*mu_c[0]/mu_c[2]**2) + cov_3D[1,2]*(R[0,1]/mu_c[2] - R[2,1]*mu_c[0]/mu_c[2]**2) + cov_3D[2,2]*(R[0,2]/mu_c[2] - R[2,2]*mu_c[0]/mu_c[2]**2))], 
    #   [R[0,0]*(cov_3D[0,0]*(R[1,0]/mu_c[2] - R[2,0]*mu_c[1]/mu_c[2]**2) + cov_3D[0,1]*(R[1,1]/mu_c[2] - R[2,1]*mu_c[1]/mu_c[2]**2) + cov_3D[0,2]*(R[1,2]/mu_c[2] - R[2,2]*mu_c[1]/mu_c[2]**2)) + R[0,1]*(cov_3D[0,1]*(R[1,0]/mu_c[2] - R[2,0]*mu_c[1]/mu_c[2]**2) + cov_3D[1,1]*(R[1,1]/mu_c[2] - R[2,1]*mu_c[1]/mu_c[2]**2) + cov_3D[1,2]*(R[1,2]/mu_c[2] - R[2,2]*mu_c[1]/mu_c[2]**2)) + R[0,2]*(cov_3D[0,2]*(R[1,0]/mu_c[2] - R[2,0]*mu_c[1]/mu_c[2]**2) + cov_3D[1,2]*(R[1,1]/mu_c[2] - R[2,1]*mu_c[1]/mu_c[2]**2) + cov_3D[2,2]*(R[1,2]/mu_c[2] - R[2,2]*mu_c[1]/mu_c[2]**2)) - mu_c[0]*(R[2,0]*(cov_3D[0,0]*R[1,0] + cov_3D[0,1]*R[1,1] + cov_3D[0,2]*R[1,2]) + R[2,1]*(cov_3D[0,1]*R[1,0] + cov_3D[1,1]*R[1,1] + cov_3D[1,2]*R[1,2]) + R[2,2]*(cov_3D[0,2]*R[1,0] + cov_3D[1,2]*R[1,1] + cov_3D[2,2]*R[1,2]))/mu_c[2]**2 + (R[0,0]*(cov_3D[0,0]*R[1,0] + cov_3D[0,1]*R[1,1] + cov_3D[0,2]*R[1,2]) + R[0,1]*(cov_3D[0,1]*R[1,0] + cov_3D[1,1]*R[1,1] + cov_3D[1,2]*R[1,2]) + R[0,2]*(cov_3D[0,2]*R[1,0] + cov_3D[1,2]*R[1,1] + cov_3D[2,2]*R[1,2]))/mu_c[2], 0, R[2,0]*(cov_3D[0,0]*(R[1,0]/mu_c[2] - R[2,0]*mu_c[1]/mu_c[2]**2) + cov_3D[0,1]*(R[1,1]/mu_c[2] - R[2,1]*mu_c[1]/mu_c[2]**2) + cov_3D[0,2]*(R[1,2]/mu_c[2] - R[2,2]*mu_c[1]/mu_c[2]**2)) + R[2,1]*(cov_3D[0,1]*(R[1,0]/mu_c[2] - R[2,0]*mu_c[1]/mu_c[2]**2) + cov_3D[1,1]*(R[1,1]/mu_c[2] - R[2,1]*mu_c[1]/mu_c[2]**2) + cov_3D[1,2]*(R[1,2]/mu_c[2] - R[2,2]*mu_c[1]/mu_c[2]**2)) + R[2,2]*(cov_3D[0,2]*(R[1,0]/mu_c[2] - R[2,0]*mu_c[1]/mu_c[2]**2) + cov_3D[1,2]*(R[1,1]/mu_c[2] - R[2,1]*mu_c[1]/mu_c[2]**2) + cov_3D[2,2]*(R[1,2]/mu_c[2] - R[2,2]*mu_c[1]/mu_c[2]**2)), 0, R[0,0]*(cov_3D[0,0]*(R[1,0]/mu_c[2] - R[2,0]*mu_c[1]/mu_c[2]**2) + cov_3D[0,1]*(R[1,1]/mu_c[2] - R[2,1]*mu_c[1]/mu_c[2]**2) + cov_3D[0,2]*(R[1,2]/mu_c[2] - R[2,2]*mu_c[1]/mu_c[2]**2)) + R[0,1]*(cov_3D[0,1]*(R[1,0]/mu_c[2] - R[2,0]*mu_c[1]/mu_c[2]**2) + cov_3D[1,1]*(R[1,1]/mu_c[2] - R[2,1]*mu_c[1]/mu_c[2]**2) + cov_3D[1,2]*(R[1,2]/mu_c[2] - R[2,2]*mu_c[1]/mu_c[2]**2)) + R[0,2]*(cov_3D[0,2]*(R[1,0]/mu_c[2] - R[2,0]*mu_c[1]/mu_c[2]**2) + cov_3D[1,2]*(R[1,1]/mu_c[2] - R[2,1]*mu_c[1]/mu_c[2]**2) + cov_3D[2,2]*(R[1,2]/mu_c[2] - R[2,2]*mu_c[1]/mu_c[2]**2)) - mu_c[0]*(R[2,0]*(cov_3D[0,0]*R[1,0] + cov_3D[0,1]*R[1,1] + cov_3D[0,2]*R[1,2]) + R[2,1]*(cov_3D[0,1]*R[1,0] + cov_3D[1,1]*R[1,1] + cov_3D[1,2]*R[1,2]) + R[2,2]*(cov_3D[0,2]*R[1,0] + cov_3D[1,2]*R[1,1] + cov_3D[2,2]*R[1,2]))/mu_c[2]**2 + (R[0,0]*(cov_3D[0,0]*R[1,0] + cov_3D[0,1]*R[1,1] + cov_3D[0,2]*R[1,2]) + R[0,1]*(cov_3D[0,1]*R[1,0] + cov_3D[1,1]*R[1,1] + cov_3D[1,2]*R[1,2]) + R[0,2]*(cov_3D[0,2]*R[1,0] + cov_3D[1,2]*R[1,1] + cov_3D[2,2]*R[1,2]))/mu_c[2], -mu_c[0]*(R[2,0]*(cov_3D[0,0]*R[2,0] + cov_3D[0,1]*R[2,1] + cov_3D[0,2]*R[2,2]) + R[2,1]*(cov_3D[0,1]*R[2,0] + cov_3D[1,1]*R[2,1] + cov_3D[1,2]*R[2,2]) + R[2,2]*(cov_3D[0,2]*R[2,0] + cov_3D[1,2]*R[2,1] + cov_3D[2,2]*R[2,2]))/mu_c[2]**2 + (R[0,0]*(cov_3D[0,0]*R[2,0] + cov_3D[0,1]*R[2,1] + cov_3D[0,2]*R[2,2]) + R[0,1]*(cov_3D[0,1]*R[2,0] + cov_3D[1,1]*R[2,1] + cov_3D[1,2]*R[2,2]) + R[0,2]*(cov_3D[0,2]*R[2,0] + cov_3D[1,2]*R[2,1] + cov_3D[2,2]*R[2,2]))/mu_c[2]], 
    #   [R[1,0]*(cov_3D[0,0]*(R[1,0]/mu_c[2] - R[2,0]*mu_c[1]/mu_c[2]**2) + cov_3D[0,1]*(R[1,1]/mu_c[2] - R[2,1]*mu_c[1]/mu_c[2]**2) + cov_3D[0,2]*(R[1,2]/mu_c[2] - R[2,2]*mu_c[1]/mu_c[2]**2)) + R[1,1]*(cov_3D[0,1]*(R[1,0]/mu_c[2] - R[2,0]*mu_c[1]/mu_c[2]**2) + cov_3D[1,1]*(R[1,1]/mu_c[2] - R[2,1]*mu_c[1]/mu_c[2]**2) + cov_3D[1,2]*(R[1,2]/mu_c[2] - R[2,2]*mu_c[1]/mu_c[2]**2)) + R[1,2]*(cov_3D[0,2]*(R[1,0]/mu_c[2] - R[2,0]*mu_c[1]/mu_c[2]**2) + cov_3D[1,2]*(R[1,1]/mu_c[2] - R[2,1]*mu_c[1]/mu_c[2]**2) + cov_3D[2,2]*(R[1,2]/mu_c[2] - R[2,2]*mu_c[1]/mu_c[2]**2)) - mu_c[1]*(R[2,0]*(cov_3D[0,0]*R[1,0] + cov_3D[0,1]*R[1,1] + cov_3D[0,2]*R[1,2]) + R[2,1]*(cov_3D[0,1]*R[1,0] + cov_3D[1,1]*R[1,1] + cov_3D[1,2]*R[1,2]) + R[2,2]*(cov_3D[0,2]*R[1,0] + cov_3D[1,2]*R[1,1] + cov_3D[2,2]*R[1,2]))/mu_c[2]**2 + (R[1,0]*(cov_3D[0,0]*R[1,0] + cov_3D[0,1]*R[1,1] + cov_3D[0,2]*R[1,2]) + R[1,1]*(cov_3D[0,1]*R[1,0] + cov_3D[1,1]*R[1,1] + cov_3D[1,2]*R[1,2]) + R[1,2]*(cov_3D[0,2]*R[1,0] + cov_3D[1,2]*R[1,1] + cov_3D[2,2]*R[1,2]))/mu_c[2], 0, 0, 0, R[1,0]*(cov_3D[0,0]*(R[1,0]/mu_c[2] - R[2,0]*mu_c[1]/mu_c[2]**2) + cov_3D[0,1]*(R[1,1]/mu_c[2] - R[2,1]*mu_c[1]/mu_c[2]**2) + cov_3D[0,2]*(R[1,2]/mu_c[2] - R[2,2]*mu_c[1]/mu_c[2]**2)) + R[1,1]*(cov_3D[0,1]*(R[1,0]/mu_c[2] - R[2,0]*mu_c[1]/mu_c[2]**2) + cov_3D[1,1]*(R[1,1]/mu_c[2] - R[2,1]*mu_c[1]/mu_c[2]**2) + cov_3D[1,2]*(R[1,2]/mu_c[2] - R[2,2]*mu_c[1]/mu_c[2]**2)) + R[1,2]*(cov_3D[0,2]*(R[1,0]/mu_c[2] - R[2,0]*mu_c[1]/mu_c[2]**2) + cov_3D[1,2]*(R[1,1]/mu_c[2] - R[2,1]*mu_c[1]/mu_c[2]**2) + cov_3D[2,2]*(R[1,2]/mu_c[2] - R[2,2]*mu_c[1]/mu_c[2]**2)) - mu_c[1]*(R[2,0]*(cov_3D[0,0]*R[1,0] + cov_3D[0,1]*R[1,1] + cov_3D[0,2]*R[1,2]) + R[2,1]*(cov_3D[0,1]*R[1,0] + cov_3D[1,1]*R[1,1] + cov_3D[1,2]*R[1,2]) + R[2,2]*(cov_3D[0,2]*R[1,0] + cov_3D[1,2]*R[1,1] + cov_3D[2,2]*R[1,2]))/mu_c[2]**2 + (R[1,0]*(cov_3D[0,0]*R[1,0] + cov_3D[0,1]*R[1,1] + cov_3D[0,2]*R[1,2]) + R[1,1]*(cov_3D[0,1]*R[1,0] + cov_3D[1,1]*R[1,1] + cov_3D[1,2]*R[1,2]) + R[1,2]*(cov_3D[0,2]*R[1,0] + cov_3D[1,2]*R[1,1] + cov_3D[2,2]*R[1,2]))/mu_c[2], R[2,0]*(cov_3D[0,0]*(R[1,0]/mu_c[2] - R[2,0]*mu_c[1]/mu_c[2]**2) + cov_3D[0,1]*(R[1,1]/mu_c[2] - R[2,1]*mu_c[1]/mu_c[2]**2) + cov_3D[0,2]*(R[1,2]/mu_c[2] - R[2,2]*mu_c[1]/mu_c[2]**2)) + R[2,1]*(cov_3D[0,1]*(R[1,0]/mu_c[2] - R[2,0]*mu_c[1]/mu_c[2]**2) + cov_3D[1,1]*(R[1,1]/mu_c[2] - R[2,1]*mu_c[1]/mu_c[2]**2) + cov_3D[1,2]*(R[1,2]/mu_c[2] - R[2,2]*mu_c[1]/mu_c[2]**2)) + R[2,2]*(cov_3D[0,2]*(R[1,0]/mu_c[2] - R[2,0]*mu_c[1]/mu_c[2]**2) + cov_3D[1,2]*(R[1,1]/mu_c[2] - R[2,1]*mu_c[1]/mu_c[2]**2) + cov_3D[2,2]*(R[1,2]/mu_c[2] - R[2,2]*mu_c[1]/mu_c[2]**2)) - mu_c[1]*(R[2,0]*(cov_3D[0,0]*R[2,0] + cov_3D[0,1]*R[2,1] + cov_3D[0,2]*R[2,2]) + R[2,1]*(cov_3D[0,1]*R[2,0] + cov_3D[1,1]*R[2,1] + cov_3D[1,2]*R[2,2]) + R[2,2]*(cov_3D[0,2]*R[2,0] + cov_3D[1,2]*R[2,1] + cov_3D[2,2]*R[2,2]))/mu_c[2]**2 + (R[1,0]*(cov_3D[0,0]*R[2,0] + cov_3D[0,1]*R[2,1] + cov_3D[0,2]*R[2,2]) + R[1,1]*(cov_3D[0,1]*R[2,0] + cov_3D[1,1]*R[2,1] + cov_3D[1,2]*R[2,2]) + R[1,2]*(cov_3D[0,2]*R[2,0] + cov_3D[1,2]*R[2,1] + cov_3D[2,2]*R[2,2]))/mu_c[2]]])

    dcovI_dJ = Get_dcovI_dJ(R, cov_3D, mu_c)
    
    # print("dcovI_dJ: ", dcovI_dJ)
    
    dJ_dmu_c = np.array([[0, 0, -1/mu_c[2]**2],
                         [0, 0, 0],
                         [-1/mu_c[2]**2, 0, 2*mu_c[0]/mu_c[2]**3],
                         [0, 0, 0], 
                         [0, 0, -1/mu_c[2]**2], 
                         [0, -1/mu_c[2]**2, 2*mu_c[1]/mu_c[2]**3]])
    # print("dJ_dmu_c: \n", dJ_dmu_c)
    
    
    First_term = np.matmul(dcovI_dJ,np.matmul(dJ_dmu_c,dmuC_dTcw))
    
    # print('First_term: \n',First_term)
    # print('seond_term: \n',second_term)
    
    dcovI_dTcw = First_term+second_term

    return dmuI_dTcw, dcovI_dTcw

def compute_analytical_jacobians_all_gaussians(mu_W_all_homo, gaussian_3D_covs, T_cw, fx, fy, W,H):
    """
    Compute analytical Jacobians for all N Gaussians.
    
    Args:
        mu_W_all_homo: (N, 4) - positions of all Gaussians in world coordinates, homogenous
        Sigma_W_all: (N, 3, 3) - covariances of all Gaussians
        T_cw: (4, 4) - camera pose
    
    Returns:
        dmu_I_dT_all: (N, 2, 6) - ∂μ_I/∂T_CW for all Gaussians
        dcov_I_dT_all: (N, 4, 6) - ∂Σ_I/∂T_CW for all Gaussians (flattened)
    """
    N = mu_W_all_homo.shape[0]
    
    dmu_I_dT_all = np.zeros((N, 2, 6))
    dcov_I_dT_all = np.zeros((N, 4, 6))

        # ndc2Pix Jacobian: d(pixel)/d(ndc) = diag(W*0.5, H*0.5)
    ndc_to_pixel_jacobian = np.array([[2*fx/W, 0],
                                      [0, 2*fy/H]], dtype=np.float64)
    
    # ndc_to_pixel_jacobian = np.array([[fx, 0],
    #                                 [0, fy]], dtype=np.float64)
    
    for i in range(N):
        # Convert to homogeneous coordinates
        mu_w_hom = mu_W_all_homo[i]

        cov_3D = gaussian_3D_covs[i]  # [1,6] in (xx,yy,zz,xy,xz,yz) format
        cov_3D_matrix = np.array([[cov_3D[0], cov_3D[1], cov_3D[2]],
                                  [cov_3D[1], cov_3D[3], cov_3D[4]],
                                  [cov_3D[2], cov_3D[4], cov_3D[5]]])  #
        
        # Compute Jacobians for this Gaussian
        dmuI_dTcw, dcovI_dTcw = GetAnalyticalJcobian(T_cw, mu_w_hom, cov_3D_matrix, fx, fy)
        
        dmu_I_dT_all[i] = ndc_to_pixel_jacobian @ dmuI_dTcw
        # dmu_I_dT_all[i] = dmuI_dTcw


        # dcovI_dTcw is (4, 6) in normalized coords: [Σ_00, Σ_01, Σ_10, Σ_11]
        # Transform to pixel space: Σ_pixel = K @ Σ_norm @ K^T, K = diag(fx, fy)
        # Using vec form: vec(Σ_pixel) = (K ⊗ K) @ vec(Σ_norm)
        # (K ⊗ K) = diag(fx², fx*fy, fy*fx, fy²) for K = diag(fx, fy)
        print("dcovI_dTcw", dcovI_dTcw)
        dcovI_dTcw[0,:] *= fx * fx      # Σ_00: row 0 of K times col 0 of K^T
        dcovI_dTcw[1,:] *= fx * fy      # Σ_01: row 0 of K times col 1 of K^T
        dcovI_dTcw[2,:] *= fy * fx      # Σ_10: row 1 of K times col 0 of K^T
        dcovI_dTcw[3,:] *= fy * fy      # Σ_11: row 1 of K times col 1 of K^T
        dcov_I_dT_all[i] = dcovI_dTcw

        print("dcovI_dTcw (pixel space): ", dcovI_dTcw)
        print('\n')
    
    return dmu_I_dT_all, dcov_I_dT_all

# reorder the Gaussians based on depth (Z in camera coordinates)

def OrderGaussiansByDepth(gaussians):
    # gaussians: [N, 3] 
    # returns: list of tuples (index, gaussian) ordered by depth (Z)
    gaussians_with_idx = [(i, g) for i, g in enumerate(gaussians)]
    gaussians_with_idx.sort(key=lambda x: x[1][2])  # sort by Z value
    return gaussians_with_idx


def compute_cov2d(mean_3D, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix, debug=False):
    """
    Compute 2D covariance matrix from 3D Gaussian parameters.
    
    Args:
        mean: 3D mean position (numpy array of shape (3,))
        focal_x: focal length in x direction
        focal_y: focal length in y direction
        tan_fovx: tangent of half field of view in x
        tan_fovy: tangent of half field of view in y
        cov3D: 3D covariance (numpy array of shape (6,) - upper triangular)
        viewmatrix: 4x4 view matrix in column-major format (numpy array of shape (16,))
    
    Returns:
        cov2D: 2D covariance (numpy array of shape (3,)) - [cov[0,0], cov[0,1], cov[1,1]]
    """


    # Transform point to view space
    view_matrix_transpose = viewmatrix.transpose()

    t = view_matrix_transpose @ np.array([mean_3D[0], mean_3D[1], mean_3D[2], 1.0])

    if debug:
        print("t: ", t)
    
    # Apply limiter to avoid extreme values
    limx = 1.3 * tan_fovx
    limy = 1.3 * tan_fovy
    txtz = t[0] / t[2]
    tytz = t[1] / t[2]
    t[0] = np.clip(txtz, -limx, limx) * t[2]
    t[1] = np.clip(tytz, -limy, limy) * t[2]
    
    # Jacobian of perspective projection
    J = np.array([
        [focal_x / t[2], 0.0, -(focal_x * t[0]) / (t[2] * t[2])],
        [0.0, focal_y / t[2], -(focal_y * t[1]) / (t[2] * t[2])],
        [0.0, 0.0, 0.0]
    ])

    
    # Extract rotation part of view matrix (column-major to row-major)
    W = np.array([
        [viewmatrix[0,0], viewmatrix[1,0], viewmatrix[2,0]],
        [viewmatrix[0,1], viewmatrix[1,1], viewmatrix[2,1]],
        [viewmatrix[0,2], viewmatrix[1,2], viewmatrix[2,2]]
    ])
    
    # Compute transformation matrix
    T = J @ W
    
    # Reconstruct 3D covariance matrix from upper triangular representation
    Vrk = np.array([
        [cov3D[0], cov3D[1], cov3D[2]],
        [cov3D[1], cov3D[3], cov3D[4]],
        [cov3D[2], cov3D[4], cov3D[5]]
    ])
    
    # Compute 2D covariance: cov = T^T * Vrk^T * T
    cov = T @ Vrk @ T.T
    
    # Apply low-pass filter: every Gaussian should be at least one pixel wide/high
    cov[0, 0] += 0.3
    cov[1, 1] += 0.3
    
    cov = cov[0:2, 0:2]

    if debug:
        print("J: \n", J)
        print("W: \n", W)
        print("T: \n", T)
        print("Vrk (3D cov): \n", Vrk)
        print("2D Covariance: \n", cov)

    # Return upper triangular part of 2x2 covariance (discard 3rd row and column)
    return cov, Vrk


def ndc2Pix(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5

def GetImagePlaneMeanAndCovs(gaussian_model, gaussians_sorted_by_depth, gaussian_3D_covs, xyz_cam, opacity_arg, xyz_world, camera_center_world, fx, fy, cx, cy, viewpoint, W, H, render_setting):
    """
    Project Gaussians to image plane and compute their 2D parameters.
    
    Args:
        xyz_cam: (N, 3) Gaussian positions in camera coordinates
        xyz_world: (N, 3) Gaussian positions in world coordinates  
        camera_center_world: (3,) Camera center in world coordinates
    """
    print(" Inside GetImagePlaneMeanAndCovs function")

    # Compute viewing directions in WORLD coordinates for SH evaluation
    viewdirs_normalized = compute_viewing_directions(xyz_world, camera_center_world)
    colors = compute_colors_from_sh(gaussian_model.get_features.cpu().detach().numpy(), viewdirs_normalized, deg=3)
    
    # Debug: Check viewing directions
    # print(f"First 3 viewing directions (world frame):")
    # for i in range(min(3, len(viewdirs_normalized))):
    #     print(f"  Gaussian {i}: viewdir = {viewdirs_normalized[i]}, norm = {np.linalg.norm(viewdirs_normalized[i]):.6f}")
    #     if i < len(xyz_world):
    #         gauss_world = xyz_world[i]
    #         expected_dir = camera_center_world - gauss_world
    #         expected_dir = expected_dir / (np.linalg.norm(expected_dir) + 1e-8)
    #         print(f"    Gaussian world pos: {gauss_world}, Camera: {camera_center_world}")
    #         print(f"    Expected viewdir: {expected_dir}, Match: {np.allclose(viewdirs_normalized[i], expected_dir)}")


    # Project depth sorted gaussians to image plane
    projected_mean_and_covs = []

    print(f"\n=== Gaussians Sorted by Depth (Camera Z) ===")
    for i in range(min(15, len(gaussians_sorted_by_depth))):
        print(gaussians_sorted_by_depth[i])
    print('\n')

    print(f"tanfovx {render_setting['tanfovx']}, tanfovy {render_setting['tanfovy']}")

    projmatrix = viewpoint.full_proj_transform.detach().cpu().numpy()
    print("Full Projection (viewproj) matrix: \n", projmatrix)
    projmatrix_transpose = projmatrix.transpose()
    print("Full Projection (viewproj) matrix transpose: \n", projmatrix_transpose)

    view_matrix = render_setting['viewmatrix'].detach().cpu().numpy()
    print("View matrix: \n", view_matrix)
    view_matrix_transpose = view_matrix.transpose()
    print("View matrix transpose: \n", view_matrix_transpose)
    print('\n')

    for elem in gaussians_sorted_by_depth:
        # print(elem)
        temp_dict = {}

        idx, _ = elem

        mu_3D = xyz_world[idx]  # in WORLD coordinates

        # Get the 2D mean in UV pixel coordinates
        # Project from 3D camera coordinates to 2D pixel coordinates
        # u = fx * (x/z) + cx
        # v = fy * (y/z) + cy
        print(f"idx:{idx} mu_3D (world): {mu_3D}")
        
        proj_hom = projmatrix_transpose @ np.array([mu_3D[0], mu_3D[1], mu_3D[2], 1.0])
        print(f"idx:{idx} Projected homogeneous coords: {proj_hom}")

        p_proj = proj_hom[0:3] / (proj_hom[3]+ 0.0000001)
        print(f"idx:{idx} Projected 3D coords after division by w: {p_proj}")

        point_image = np.array([ ndc2Pix(p_proj[0], W), ndc2Pix(p_proj[1], H) ])
        print(f"idx:{idx} Projected 2D pixel coords: {point_image}")

        # break
        temp_dict['proj_hom'] = proj_hom
        temp_dict['p_proj'] = p_proj

        temp_dict['idx'] = idx
        temp_dict['mean_2D'] = point_image
        temp_dict['mean_3D'] = mu_3D
        temp_dict['alpha'] = opacity_arg[idx][0]
        # temp_dict['depth'] = mu_3D[2]
        temp_dict['color'] = colors[idx]

        # Get the 2D covariance in PIXEL space

        # if(idx == 10):
        #     print("mean_3D: ",mu_3D)
        #     print("viewmatrix: ", render_setting['viewmatrix'].detach().cpu().numpy())
        #     print("focal_x: ", fx)
        #     print("focal_y: ", fy)
        #     print("tan_fovx: ", render_setting['tanfovx'])
        #     print("tan_fovy: ", render_setting['tanfovy'])

        cov_pixel, cov_3D = compute_cov2d(mu_3D, fx, fy, render_setting['tanfovx'], render_setting['tanfovy'], gaussian_3D_covs[idx], render_setting['viewmatrix'].detach().cpu().numpy())



        p_view =  view_matrix_transpose@np.array([mu_3D[0], mu_3D[1], mu_3D[2], 1.0])  #(view_mat @ proj_hom[0:3]) + np.array([view_matrix[3,0], view_matrix[3,1], view_matrix[3,2]])
        print(f"idx:{idx} Projected view coords: {p_view}")
        temp_dict['depth'] = p_view[2]


        
        temp_dict['cov_2D'] = cov_pixel
        temp_dict['cov_3D'] = cov_3D
        temp_dict['cov_3D_striped'] = gaussian_3D_covs[idx]

        print('\n')
        
        projected_mean_and_covs.append(temp_dict)

    print("Check first 15 projected Gaussians:")
    for i in range(min(15, len(projected_mean_and_covs))):
        print(projected_mean_and_covs[i])

    # Sanity check: Get an RGB image from the projected Gaussians
    # rendered_Image_from_Projected_Gaussians_vectorized(projected_mean_and_covs)

    return projected_mean_and_covs

def rendered_Image_from_Projected_Gaussians_vectorized(projected_mean_and_covs):
    ## Sanity check: Get an RGB image from the projected Gaussians (Vectorized & Chunked)
    H, W = 480, 640
    rendered_image = np.zeros((H, W, 3), dtype=np.float32)
    
    # Create pixel coordinate grid
    u_coords, v_coords = np.meshgrid(np.arange(W), np.arange(H))
    pixel_coords = np.stack([u_coords, v_coords], axis=-1).reshape(-1, 2)  # (H*W, 2)
    
    # Process in chunks to manage memory
    chunk_size = 10000
    num_pixels = H * W
    
    for chunk_idx in range(0, num_pixels, chunk_size):
        chunk_end = min(chunk_idx + chunk_size, num_pixels)
        chunk_pixels = pixel_coords[chunk_idx:chunk_end]  # (chunk_size, 2)
        chunk_colors = np.zeros((len(chunk_pixels), 3), dtype=np.float32)
        chunk_T = np.ones(len(chunk_pixels), dtype=np.float32)
        
        # Render each Gaussian for this chunk (vectorized)
        for gauss in projected_mean_and_covs:
            # Vectorized alpha computation
            delta = chunk_pixels - gauss['mean_2D']  # (chunk_size, 2)
            sigma_inv = np.linalg.inv(gauss['cov_2D'])
            
            # Compute Mahalanobis distance for all pixels at once
            power = -0.5 * np.sum(delta @ sigma_inv * delta, axis=1)  # (chunk_size,)
            gaussian_vals = np.exp(power)
            alphas = np.clip(gauss['alpha'] * gaussian_vals, 0.0, 1.0)  # (chunk_size,)
            
            # Accumulate color with alpha blending
            chunk_colors += gauss['color'] * (alphas * chunk_T)[:, None]
            chunk_T *= (1 - alphas)
        
        # Write chunk back to rendered image
        chunk_v = chunk_idx // W
        chunk_u_start = chunk_idx % W
        rendered_image.reshape(-1, 3)[chunk_idx:chunk_end] = chunk_colors
        
        if (chunk_idx // chunk_size) % 10 == 0:
            print(f"Rendered {chunk_end}/{num_pixels} pixels ({100*chunk_end/num_pixels:.1f}%)")
    
    # import matplotlib.pyplot as plt
    plt.imshow(np.clip(rendered_image, 0.0, 1.0))
    plt.title("Rendered Image from Projected Gaussians (Vectorized)")
    plt.show()


def compute_alpha_at_pixel(gaussian, pixel_pos):
    """
    Compute alpha value for a Gaussian at a pixel.
    
    Args:
        gaussian: Dict with 'mean_2D', 'Sigma_I', 'opacity'
        pixel_pos: (u, v) pixel coordinates
    
    Returns:
        alpha: Float
    """
    # print(gaussian['mean_2D'])
    # print(gaussian['cov_2D'])

    Delta = pixel_pos - gaussian['mean_2D']
    Sigma_inv = np.linalg.inv(gaussian['cov_2D'])
    
    # Mahalanobis distance
    power = -0.5 * Delta @ Sigma_inv @ Delta
    
    # Gaussian value
    gaussian_val = np.exp(power)
    
    # Alpha
    alpha = gaussian['alpha'] * gaussian_val
    
    return np.clip(alpha, 0.0, 1.0)


def compute_gradients_2D(image_projected_gaussians_sorted_by_depth, rendered_color, rendered_depth, 
                         gt_color, gt_depth, image_size):
    """
    Compute ∂L/∂μ_I and ∂L/∂Σ_I for all Gaussians.
    
    Args:
        image_projected_gaussians_sorted_by_depth: List of projected 2D Gaussians (with mu_I, Sigma_I, cov_I, alpha, etc.)
        rendered_color: (H, W, 3) rendered color image
        rendered_depth: (H, W) rendered depth image
        gt_color: (H, W, 3) ground truth color
        gt_depth: (H, W) ground truth depth
        image_size: (H, W)
    
    Returns:
        grad_mu_I: List of gradients ∂L/∂μ_I for each Gaussian (each is 2D vector)
        grad_Sigma_I: List of gradients ∂L/∂Σ_I for each Gaussian (each is 2x2 matrix)
    """
    H, W = image_size
    num_gaussians = len(image_projected_gaussians_sorted_by_depth)
    
    # Initialize gradients
    grad_mu_I = [np.zeros(2) for _ in range(num_gaussians)]
    grad_Sigma_I = [np.zeros((2, 2)) for _ in range(num_gaussians)]
    
    # Step 1: Compute pixel-level loss gradients
    # Using L1 loss here
    grad_color = np.sign(rendered_color - gt_color)  # (H, W, 3)
    grad_depth = np.sign(rendered_depth - gt_depth)  # (H, W)
    
    # Step 2: For each pixel, backpropagate through rendering
    for v in range(H):
        print(f"Processing row: {v+1}")
        for u in range(W):
            pixel_pos = np.array([u, v])
            
            # Get loss gradient at this pixel
            dL_dC = grad_color[v, u]  # (3,)
            dL_dD = grad_depth[v, u]  # scalar
            
            # Sort Gaussians by depth for this pixel
            # gaussians_with_idx = [(i, g) for i, g in enumerate(gaussians)]
            # gaussians_sorted = sorted(gaussians_with_idx, 
            #                          key=lambda x: x[1]['depth'])
            
            # Forward pass: compute alphas and transmittances
            alphas = []
            transmittances = []
            T = 1.0
            
            for gauss in image_projected_gaussians_sorted_by_depth:
                # print(gauss)
                alpha = compute_alpha_at_pixel(gauss, pixel_pos)
                alphas.append(alpha)
                transmittances.append(T)
                T *= (1 - alpha)
            
            # Backward pass: compute gradient w.r.t. alphas
            grad_alphas = [0.0] * len(image_projected_gaussians_sorted_by_depth)
            
            for idx, gauss in enumerate(image_projected_gaussians_sorted_by_depth):
                alpha_i = alphas[idx]
                T_i = transmittances[idx]
                
                # Compute accumulated contribution after i
                color_after = np.zeros(3)
                depth_after = 0.0
                
                for j in range(idx + 1, len(image_projected_gaussians_sorted_by_depth)):
                    alpha_j = alphas[j]
                    T_j = transmittances[j]
                    g_j = image_projected_gaussians_sorted_by_depth[j]
                    
                    color_after += g_j['color'] * alpha_j * T_j # Calculated in compute_colors_from_sh function
                    depth_after += g_j['depth'] * alpha_j * T_j
                
                # Gradient of rendered color w.r.t. alpha_i
                dC_dalpha_i = gauss['color'] * T_i
                if alpha_i < 0.999:  # Avoid division by zero
                    dC_dalpha_i -= color_after / (1 - alpha_i)
                
                # Gradient of rendered depth w.r.t. alpha_i
                dD_dalpha_i = gauss['depth'] * T_i
                if alpha_i < 0.999:
                    dD_dalpha_i -= depth_after / (1 - alpha_i)
                
                # Chain rule: ∂L/∂α_i
                dL_dalpha_i = np.dot(dL_dC, dC_dalpha_i) + dL_dD * dD_dalpha_i
                grad_alphas[idx] = dL_dalpha_i
            
            # Step 3: Backprop from alphas to μ_I and Σ_I
            for idx, gauss in enumerate(image_projected_gaussians_sorted_by_depth):
                dL_dalpha = grad_alphas[idx]
                alpha = alphas[idx]
                
                if abs(alpha) < 1e-8:  # Skip if alpha is too small
                    continue
                
                # Compute Δ = p - μ_I
                Delta = pixel_pos - gauss['mean_2D']  # (2,)
                
                # Compute Σ_I^{-1}
                Sigma_I_inv = np.linalg.inv(gauss['cov_2D'])
                
                # Gradient w.r.t. μ_I: ∂α/∂μ_I = α * Σ^{-1} @ Δ
                # Note: This should be positive because we're moving in direction to increase alpha
                dalpha_dmu = alpha * (Sigma_I_inv @ Delta)  # (2,)
                
                # Chain rule: ∂L/∂μ_I
                grad_mu_I[idx] += dL_dalpha * dalpha_dmu
                
                # Gradient w.r.t. Σ_I: ∂α/∂Σ_I = 0.5 * α * Σ^{-1} @ (ΔΔ^T) @ Σ^{-1}
                # Positive sign because derivative of exp(-0.5 * x^T Σ^{-1} x) w.r.t. Σ is positive
                dalpha_dSigma = 0.5 * alpha * (
                    Sigma_I_inv @ np.outer(Delta, Delta) @ Sigma_I_inv
                )  # (2, 2)
                
                # Chain rule: ∂L/∂Σ_I
                grad_Sigma_I[idx] += dL_dalpha * dalpha_dSigma
    
    return grad_mu_I, grad_Sigma_I



def compute_gradients_2D_vectorized_chunked(image_projected_gaussians_sorted_by_depth, rendered_color, rendered_depth, 
                                           gt_color, gt_depth, mask, image_size, chunk_size=1000):
    """
    Memory-efficient vectorized version: Process pixels in chunks to avoid memory overflow.
    
    Args:
        image_projected_gaussians_sorted_by_depth: List of projected 2D Gaussians
        rendered_color: (H, W, 3) rendered color image
        rendered_depth: (H, W) rendered depth image
        gt_color: (H, W, 3) ground truth color
        gt_depth: (H, W) ground truth depth
        mask: (H, W) boolean mask for valid pixels
        image_size: (H, W)
        chunk_size: Number of pixels to process at once (reduce if memory issues)
    
    Returns:
        grad_mu_I: (N, 2) gradients ∂L/∂μ_I for each Gaussian
        grad_Sigma_I: (N, 2, 2) gradients ∂L/∂Σ_I for each Gaussian
    """
    H, W = image_size
    N = len(image_projected_gaussians_sorted_by_depth)
    
    print(f"\n=== Gradient Computation Debug ===")
    print(f"Image size: {H}x{W}, Num Gaussians: {N}")
    
    # Extract Gaussian properties into arrays
    means_2D = np.array([g['mean_2D'] for g in image_projected_gaussians_sorted_by_depth], dtype=np.float32)  # (N, 2)
    covs_2D = np.array([g['cov_2D'] for g in image_projected_gaussians_sorted_by_depth], dtype=np.float32)    # (N, 2, 2)
    colors = np.array([g['color'] for g in image_projected_gaussians_sorted_by_depth], dtype=np.float32)      # (N, 3)
    depths = np.array([g['depth'] for g in image_projected_gaussians_sorted_by_depth], dtype=np.float32)      # (N,)
    alphas = np.array([g['alpha'] for g in image_projected_gaussians_sorted_by_depth], dtype=np.float32)      # (N,)

    print(f"Mean 2D range: ({means_2D[:, 0].min():.1f}, {means_2D[:, 0].max():.1f}), ({means_2D[:, 1].min():.1f}, {means_2D[:, 1].max():.1f})")
    print(f"Depth range: {depths.min():.3f} to {depths.max():.3f}")
    print(f"Alpha range: {alphas.min():.4f} to {alphas.max():.4f}")
    print(f"Color range: {colors.min():.4f} to {colors.max():.4f}")
    
    # Precompute inverse covariances
    covs_inv = np.linalg.inv(covs_2D)  # (N, 2, 2)
    
    # Compute pixel-level loss gradients
    color_diff = rendered_color - gt_color
    depth_diff = rendered_depth - gt_depth
    
    # Apply mask to match compute_loss function
    # Color: mask applied to both rendered and GT
    mask_3d = mask[..., None].astype(np.float32)  # (H, W, 1) for broadcasting
    grad_color = np.sign(color_diff).astype(np.float32) * mask_3d  # (H, W, 3)
    
    # Depth: only where depth_gt > 0 (matching compute_loss depth_mask)
    depth_valid_mask = (gt_depth > 0.0) & mask  # (H, W)
    grad_depth = np.sign(depth_diff).astype(np.float32) * depth_valid_mask.astype(np.float32)  # (H, W)
    
    print(f"\nLoss gradients:")
    print(f"Color diff range: {color_diff.min():.4f} to {color_diff.max():.4f}")
    print(f"Depth diff range: {depth_diff.min():.4f} to {depth_diff.max():.4f}")
    print(f"Masked color diffs: {np.count_nonzero(grad_color)} / {color_diff.size}")
    print(f"Valid depth diffs: {np.count_nonzero(grad_depth)} / {depth_diff.size}")
    
    # Create pixel grid - NOTE: mean_2D is (u, v) format
    v_coords, u_coords = np.mgrid[0:H, 0:W]
    pixel_positions = np.stack([u_coords, v_coords], axis=-1).astype(np.float32)  # (H, W, 2) in (u, v) format
    
    # Flatten
    pixels_flat = pixel_positions.reshape(-1, 2)  # (P, 2) where P = H*W
    grad_color_flat = grad_color.reshape(-1, 3)   # (P, 3)
    grad_depth_flat = grad_depth.reshape(-1)      # (P,)
    
    P = pixels_flat.shape[0]
    
    # Initialize gradient accumulation
    grad_mu_I = np.zeros((N, 2), dtype=np.float32)
    grad_Sigma_I = np.zeros((N, 2, 2), dtype=np.float32)
    grad_depth_per_gaussian = np.zeros((N,), dtype=np.float32)  # NEW: To accumulate depth gradients for debugging
    grad_color_per_gaussian = np.zeros((N, 3), dtype=np.float32)  # NEW: Per-Gaussian color gradients for SH backward
    
    # Process in chunks
    num_chunks = (P + chunk_size - 1) // chunk_size
    print(f"\nProcessing {num_chunks} chunks of size {chunk_size}")

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, P)
        
        # Get chunk of pixels
        pixels_chunk = pixels_flat[start_idx:end_idx]  # (C, 2)
        grad_color_chunk = grad_color_flat[start_idx:end_idx]  # (C, 3)
        grad_depth_chunk = grad_depth_flat[start_idx:end_idx]  # (C,)
        C = pixels_chunk.shape[0]
        
        # Compute Delta for this chunk
        Delta = pixels_chunk[:, None, :] - means_2D[None, :, :]  # (C, N, 2)
        
        # Compute alphas for this chunk
        temp = np.einsum('cni,nij->cnj', Delta, covs_inv)  # (C, N, 2)
        exponent = -0.5 * np.einsum('cni,cni->cn', temp, Delta)  # (C, N)
        gaussian_vals = np.exp(exponent)  # (C, N) - just the Gaussian function
        alphas_all = alphas[None, :] * gaussian_vals  # (C, N) - multiply by opacity
        alphas_all = np.clip(alphas_all, 0.0, 1.0)  # Clip to valid range
        
        # Debug first chunk
        if chunk_idx == 0:
            print(f"\nFirst chunk statistics:")
            print(f"  Gaussian vals range: {gaussian_vals.min():.6f} to {gaussian_vals.max():.6f}")
            print(f"  Alphas range: {alphas_all.min():.6f} to {alphas_all.max():.6f}")
            print(f"  Non-zero alphas: {np.count_nonzero(alphas_all > 1e-6)}")
        
        # Compute transmittances
        one_minus_alpha = 1 - alphas_all
        transmittances = np.cumprod(
            np.concatenate([np.ones((C, 1), dtype=np.float32), one_minus_alpha[:, :-1]], axis=1),
            axis=1
        )  # (C, N)
        
        # Compute contributions
        alpha_T = alphas_all * transmittances  # (C, N)
        color_contrib = colors[None, :, :] * alpha_T[:, :, None]  # (C, N, 3)
        depth_contrib = depths[None, :] * alpha_T  # (C, N)

        grad_depth_chunk_per_gaussian = alpha_T * grad_depth_chunk[:, None]  # (C, N)
        grad_depth_per_gaussian += grad_depth_chunk_per_gaussian.sum(axis=0)  # (N,)

        # NEW: accumulate ∂L/∂c_i = Σ_pixels (α_i * T_i) * ∂L/∂C(p)
        # alpha_T is (C, N), grad_color_chunk is (C, 3)
        grad_color_chunk_per_gaussian = alpha_T[:, :, None] * grad_color_chunk[:, None, :]  # (C, N, 3)
        grad_color_per_gaussian += grad_color_chunk_per_gaussian.sum(axis=0)  # (N, 3)
        
        # Reverse cumsum for "sum after i"
        color_after = np.flip(np.cumsum(np.flip(color_contrib, axis=1), axis=1), axis=1)
        depth_after = np.flip(np.cumsum(np.flip(depth_contrib, axis=1), axis=1), axis=1)
        color_after = np.concatenate([color_after[:, 1:, :], np.zeros((C, 1, 3), dtype=np.float32)], axis=1)
        depth_after = np.concatenate([depth_after[:, 1:], np.zeros((C, 1), dtype=np.float32)], axis=1)
        
        # Gradient of rendered values w.r.t. alpha
        dC_dalpha = colors[None, :, :] * transmittances[:, :, None]  # (C, N, 3)
        dD_dalpha = depths[None, :] * transmittances  # (C, N)
        
        # Avoid division by zero
        safe_denom = np.where(alphas_all < 0.999, 1 - alphas_all, 1.0)
        dC_dalpha -= color_after / safe_denom[:, :, None]
        dD_dalpha -= depth_after / safe_denom
        
        # Chain rule: dL/dalpha
        dL_dalpha = np.einsum('ci,cni->cn', grad_color_chunk, dC_dalpha) + grad_depth_chunk[:, None] * dD_dalpha
        
        if chunk_idx == 0:
            print(f"  dL_dalpha range: {dL_dalpha.min():.6f} to {dL_dalpha.max():.6f}")
            print(f"  Non-zero dL_dalpha: {np.count_nonzero(np.abs(dL_dalpha) > 1e-6)}")
        
        # Backprop to mu_I
        # ∂α/∂μ_I = α * Σ^{-1} @ Δ
        dalpha_dmu = alphas_all[:, :, None] * np.einsum('nij,cnj->cni', covs_inv, Delta)  # (C, N, 2)
        grad_mu_chunk = dL_dalpha[:, :, None] * dalpha_dmu  # (C, N, 2)
        grad_mu_I += grad_mu_chunk.sum(axis=0)  # Accumulate
        
        # Backprop to Sigma_I
        # For α = opacity * exp(-0.5 * Δ^T Σ^{-1} Δ), we have:
        # ∂α/∂Σ = α * 0.5 * Σ^{-1} (ΔΔ^T) Σ^{-1}
        # This is the derivative w.r.t. Σ (not Σ^{-1})
        Delta_outer = np.einsum('cni,cnj->cnij', Delta, Delta)  # (C, N, 2, 2)
        temp1 = np.einsum('nij,cnjk->cnik', covs_inv, Delta_outer)  # (C, N, 2, 2)
        dalpha_dSigma = 0.5 * alphas_all[:, :, None, None] * np.einsum('cnij,njk->cnik', temp1, covs_inv)
        grad_Sigma_chunk = dL_dalpha[:, :, None, None] * dalpha_dSigma  # (C, N, 2, 2)
        grad_Sigma_I += grad_Sigma_chunk.sum(axis=0)  # Accumulate
        
        if (chunk_idx + 1) % 100 == 0:
            print(f"Processed {chunk_idx + 1}/{num_chunks} chunks ({100*(chunk_idx+1)/num_chunks:.1f}%)")
    
    print(f"\n=== Final Gradients ===")
    print(f"grad_mu_I range: {grad_mu_I.min():.6f} to {grad_mu_I.max():.6f}")
    print(f"grad_Sigma_I range: {grad_Sigma_I.min():.6f} to {grad_Sigma_I.max():.6f}")
    print(f"Non-zero grad_mu_I: {np.count_nonzero(np.abs(grad_mu_I) > 1e-6)}")
    print(f"Non-zero grad_Sigma_I: {np.count_nonzero(np.abs(grad_Sigma_I) > 1e-6)}")
    print(f"grad_depth_per_gaussian range: {grad_depth_per_gaussian.min():.6f} to {grad_depth_per_gaussian.max():.6f}")
    print(f"Non-zero grad_depth_per_gaussian: {np.count_nonzero(np.abs(grad_depth_per_gaussian) > 1e-6)}")
    print(f"grad_color_per_gaussian range: {grad_color_per_gaussian.min():.6f} to {grad_color_per_gaussian.max():.6f}")
    print(f"Non-zero grad_color_per_gaussian: {np.count_nonzero(np.abs(grad_color_per_gaussian) > 1e-6)}")
    
    return grad_mu_I, grad_Sigma_I, grad_depth_per_gaussian, grad_color_per_gaussian
    

if __name__ == "__main__":
	
    gaussian_model = GaussianModel(3)
    gaussian_model.load_tensors("./optimized_params_small.pt")

    dataset_root = "../gs_obj_opt/data/NOCS/val/00337"
    image_path = dataset_root + "/0008_color.png"
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img)
    img_tensor = img_tensor.permute(2, 0, 1).to(torch.device("cuda"))


    depth_path = dataset_root + "/0008_depth.png"
    depth_img_ = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth_channels = cv2.split(depth_img_)

    # compute the depth values, 2nd channel * 256 + 3rd channel
    depth_channel_1 = depth_channels[1].astype(np.float32)
    depth_channel_2 = depth_channels[2].astype(np.float32)
    depth_img = depth_channel_1 * 256 + depth_channel_2
    depth_img = depth_img.astype(np.float32) / 1000.0
    depth_tensor = torch.from_numpy(depth_img)
    depth_tensor = depth_tensor.unsqueeze(0).to(torch.device("cuda"))

    mask_path = dataset_root + "/0008_mask.png"
    mask_all = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    instance_id = 1
    channel = 2

    # extract the channel
    mask_channels = cv2.split(mask_all)
    mask = (mask_channels[channel] == instance_id)

    # convert the mask image to torch.Tensor
    mask_tensor = torch.from_numpy(mask)
    mask_tensor = mask_tensor.to(torch.device("cuda"))
    print(f"mask_tensor shape: {mask_tensor.shape}")

    masked_color = img_tensor * mask_tensor.unsqueeze(0)

    pose_data_root = "./Jacob_test_result/"
    w2c_gt_path = pose_data_root + "w2c_gt.txt"
    T_noise_path = pose_data_root + "T_noise.txt"
    w2c_gt = np.loadtxt(w2c_gt_path, dtype=np.float32)
    T_noise = np.loadtxt(T_noise_path, dtype=np.float32)

    print(f"w2c_gt: {w2c_gt}")
    print(f"T_noise: {T_noise}")

    cam_intrinsics = np.array([
        [577.5, 0, 319.5],
        [0, 577.5, 239.5],
        [0, 0, 1]
    ], dtype=np.float32)

    fx = cam_intrinsics[0, 0]
    fy = cam_intrinsics[1, 1]
    cx = cam_intrinsics[0, 2]
    cy = cam_intrinsics[1, 2]


    w = 640
    h = 480

    image_size = (h, w)

    # w2c is the noisy camera pose
    w2c = w2c_gt @ T_noise

    print(f"w2c: {w2c}")
    w2c_ = torch.from_numpy(w2c).to(torch.device("cuda"))
    w2c_ = w2c_.transpose(0,1) 
    print("w2c_ on cuda device: \n", w2c_)

    render_setting = get_render_settings(w, h, cam_intrinsics, w2c_)
    fovx = focal2fov(fx, w)
    fovy = focal2fov(fy, h)

    # Store PURE projection (not viewproj) as Camera.projection_matrix
    # The renderer uses full_proj_transform (viewproj) for projmatrix,
    # and projection_matrix (pure projection) for projmatrix_raw (Jacobian entries)
    projmatrix = render_setting['projmatrix']
    viewpoint = Camera(0, img_tensor, depth_tensor, w2c_gt, projmatrix, fx, fy, cx, cy, fovx, fovy, h, w, render_setting['viewmatrix'].T)

    projmatrix = viewpoint.projection_matrix

    print("Projection Matrix from viewpoint:")
    print(projmatrix.detach().cpu().numpy())
    print(" Projection Matrix transposed:")
    print(projmatrix.transpose(0,1).detach().cpu().numpy())

    print(projmatrix.transpose(0,1))

    # Render the RGB and Depth image

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    pipline = pipeline_params

    render_pkg = render(
                    viewpoint, gaussian_model, pipline, background
                )
    render_image, render_depth, render_opacity = (
        render_pkg["render"],
        render_pkg["depth"],
        render_pkg["opacity"],
    )

    # Display Rendered Image
    # DisplayRenderedImage(viewpoint, render_image)

    # Visualize Gaussians In World Frame
    # VisualizeGaussiansInWorldFrame(gaussian_model)


    # Transform Gaussians from world coordinates to camera coordinates
    xyz_world = gaussian_model.get_xyz # [N, 3] in world coordinates

    # Add homogeneous coordinate (convert to [N, 4])
    xyz_world_homo = torch.cat([xyz_world, torch.ones(xyz_world.shape[0], 1, device='cuda')], dim=1).cpu().detach().numpy()   # [N, 4]

    # Apply w2c transformation: xyz_cam = w2c @ xyz_world
    # w2c_ is already transposed, so we need to transpose it back

    # Transform to camera coordinates using the w2c (view) matrix, NOT the projection matrix
    xyz_cam_homo = (w2c @ xyz_world_homo.T).T  # [N, 4] # gaussians_in_cam_frame
    xyz_cam = xyz_cam_homo[:, :3]  # [N, 3] - 3D GS in camera coordinates (Z is depth) 
    
    print("\n=== Coordinate Transformation Verification ===")
    print("w2c matrix:")
    print(w2c)
    print(f"World coordinates shape: {xyz_world.shape}")
    print(f"Camera coordinates shape: {xyz_cam.shape}")
    # print(f"\nFirst 5 Gaussians:")
    print(f"Gaussians in World coords:\n{xyz_world}")

    print(f"Gaussians in camera coords:\n{xyz_cam}")
    print(f"Depth (Z) range: {xyz_cam[:, 2].min():.3f} to {xyz_cam[:, 2].max():.3f}")

    # Visualize Gaussians In Camera Frame
    # VisualizeGaussiansInCameraFrame(xyz_cam)




    # Get Gaussian covariance and opcaity.
    gaussian_3D_covs = gaussian_model.get_covariance().detach().cpu().numpy()
    opacity = gaussian_model.get_opacity.detach().cpu().numpy()

    # get and print scales:
    scales = gaussian_model.get_scaling.detach().cpu().numpy()
    # print(f"Gaussians in World scales ids: 29181 :\n{scales[29181]}")

    rotation = gaussian_model.get_rotation.detach().cpu().numpy()
    # print(f"Gaussians in World rotation ids: 29181 :\n{rotation[29181]}")

    print(f"Gaussians in World gaussian_3D_covs:\n{gaussian_3D_covs}")
    print(f"Gaussians in World opacity:\n{opacity}")
    print(" Projection Matrix __main__")
    print(viewpoint.projection_matrix.detach().cpu().numpy())

    print(" View Matrix")
    print(render_setting['viewmatrix'])


    # # Here xyz_cam is mu_c = w2c * mu_w i.e. Gaussians in camera coordinates
    gaussians_sorted_by_depth = OrderGaussiansByDepth(xyz_cam)
    # print(f"\n=== Gaussians Sorted by Depth (Camera Z) ===")
    # for elem in gaussians_sorted_by_depth:
    #     print(elem)
    # sys.exit(0)
   
    # print(f"\n=== Camera and Viewing Direction Info ===")
    # print(f"Camera center (world): {camera_center_world}")
    # print("viewpoint.camera_center: ",viewpoint.camera_center.cpu().numpy())
    
    # # Pass both camera coordinates (for projection) and world coordinates (for SH)
    xyz_world_np = xyz_world[:, :3].cpu().detach().numpy()  # Remove homogeneous coord, convert to numpy
    image_projected_gaussians_sorted_by_depth = GetImagePlaneMeanAndCovs(
       gaussian_model, gaussians_sorted_by_depth, gaussian_3D_covs, xyz_cam, opacity, xyz_world_np, viewpoint.camera_center.cpu().numpy(), fx, fy, cx, cy, viewpoint, w, h, render_setting)

    # sys.exit(0)

    # print(f"\n=== Projected 2D Gaussians Info ===")
    # print(f"\n=== 2D Means: ")
    for i, g in enumerate(image_projected_gaussians_sorted_by_depth[0:5]):
        print(f"Gaussian {g['idx']}: mu_I = {g['mean_2D']}, depth = {g['depth']:.3f}, alpha = {g['alpha']:.4f}")
    


    # sys.exit(0)
    # print(image_projected_gaussians_sorted_by_depth[0])
    # This will give ∂L/∂μ_I and ∂L/∂Σ_I
    gt_color = viewpoint.original_image.cpu().detach().numpy().transpose(1,2,0)  # (H, W, 3)
    gt_depth = viewpoint.depth.cpu().detach().numpy()  # (H, W)
    rendered_color = render_image.cpu().detach().numpy().transpose(1,2,0)  # (H, W, 3)
    rendered_depth = render_depth.cpu().detach().numpy()  # (H, W)

    # grad_mu_I, grad_Sigma_I =compute_gradients_2D(image_projected_gaussians_sorted_by_depth, rendered_color, rendered_depth, gt_color, gt_depth, image_size)

    print("Call compute_gradients_2D_vectorized_chunked")
    # Convert mask to numpy for gradient computation
    mask_np = mask_tensor.cpu().numpy()
    grad_mu_I_pixel, grad_Sigma_I_pixel, grad_depth_per_gaussian, grad_color_per_gaussian = compute_gradients_2D_vectorized_chunked(image_projected_gaussians_sorted_by_depth, rendered_color, rendered_depth, gt_color, gt_depth, mask_np, image_size, chunk_size=1000)


    grad_Sigma_I_pixel_flat = grad_Sigma_I_pixel.reshape(-1, 4)

    dL_dtau_depth_total = np.zeros((6,))


    print(f"\n=== Normalized Gradients (for pose optimization) ===")


    # Save both pixel-space and normalized gradients
    np.save('./Jacob_test_result/grad_mu_I_pixel.npy', grad_mu_I_pixel)
    np.save('./Jacob_test_result/grad_Sigma_I_pixel.npy', grad_Sigma_I_pixel)
    np.save('./Jacob_test_result/grad_depth_per_gaussian.npy', grad_depth_per_gaussian)

    print(f"\nGradients saved to ./Jacob_test_result/")
    print(f"  - grad_mu_I_pixel.npy (pixel space)")
    print(f"  - grad_Sigma_I_pixel.npy (pixel space)")
    print(f"  - grad_depth_per_gaussian.npy (pixel space)")
    print(f"  - grad_mu_I_normalized.npy (normalized space, ready for pose Jacobians)")
    print(f"  - grad_Sigma_I_normalized.npy (normalized space, ready for pose Jacobians)")

    
    print(f"\nFlattened covariance gradient shape: {grad_Sigma_I_pixel_flat.shape}")
    
    dmu_I_dT_all, dcov_I_dT_all = compute_analytical_jacobians_all_gaussians(xyz_world_homo, gaussian_3D_covs, w2c, fx, fy, w, h) 
    indices = np.array([g['idx'] for g in image_projected_gaussians_sorted_by_depth]) # get the Gaussian depth sorted indice

    dL_dtau = np.zeros((6,))  # 6 DOF for pose parameters
    dL_dtau_mu_total = np.zeros((6,))  # Track mean contribution
    dL_dtau_cov_total = np.zeros((6,))  # Track covariance contribution
    dL_dtau_sh_total = np.zeros((6,))  # Track SH view-direction contribution

    # Precompute quantities needed for SH backward path
    camera_center_world = viewpoint.camera_center.cpu().numpy()  # (3,) camera center in world coordinates
    sh_coeffs_all = gaussian_model.get_features.cpu().detach().numpy()  # (N, 16, 3)
    # Precompute clamped mask: color channels that were clamped to 0 get zero gradient
    # colors_before_clamp = eval_sh(deg, sh_coeffs, viewdirs) + 0.5
    # clamped[ch] = True if colors_before_clamp[ch] < 0
    xyz_world_np_for_sh = xyz_world[:, :3].cpu().detach().numpy() if hasattr(xyz_world, 'cpu') else xyz_world[:, :3]
    viewdirs_all = compute_viewing_directions(xyz_world_np_for_sh, camera_center_world)
    colors_before_clamp = eval_sh(3, sh_coeffs_all, viewdirs_all) + 0.5  # (N, 3)
    clamped_mask = (colors_before_clamp < 0.0)  # (N, 3) - True where clamped

    print("\n=== Computing dL/dtau via Chain Rule ===")
    print(f"Number of Gaussians to process: {len(indices)}")
    print(f"Clamped color channels: {clamped_mask.sum()} / {clamped_mask.size}")
    
    # Debug first few Gaussians
    debug_count = min(15, len(indices))
    
    for i in range(len(indices)):
        idx = indices[i]
        
        dL_dmuI = grad_mu_I_pixel[i] # dL_dmuI (shape: 2,)
        dL_dCovI = grad_Sigma_I_pixel_flat[i] # dL_dCovI (shape: 4,) [cov_00, cov_01, cov_10, cov_11]

        dmuI_dtau = dmu_I_dT_all[idx] # dmuI_dtau (shape: 2, 6)
        dcovI_dtau = dcov_I_dT_all[idx] # dCovI_dtau (shape: 4, 6)

        #dL_dtau (mu_I part): (2,) @ (2, 6) -> (6,)
        dL_dtau_muI = dL_dmuI @ dmuI_dtau
        #dL_dtau (cov_I part): (4,) @ (4, 6) -> (6,)
        dL_dtau_covI = dL_dCovI @ dcovI_dtau

        mu_w_hom = xyz_world_homo[idx]  # (4,)
        p_C = w2c @ mu_w_hom  # (4,), camera-space position

        dL_ddepth_i = grad_depth_per_gaussian[i]  # scalar
        dd_dtau = np.array([0, 0, 1, p_C[1], -p_C[0], 0], dtype=np.float64)  # (6,)
        dL_dtau_depth_i = dL_ddepth_i * dd_dtau  # (6,)
        
        dL_dtau_depth_total += dL_dtau_depth_i

        # === SH view-direction gradient path ===
        # Chain: τ(ρ) → c_W → dir_orig → dir_normalized → SH color → L
        # Only affects translational part (ρ, indices 0-2)
        dL_dRGB_i = grad_color_per_gaussian[i].copy()  # (3,)
        # Apply clamping mask: zero out gradient for channels that were clamped
        dL_dRGB_i[clamped_mask[idx]] = 0.0
        
        mu_w_3d = xyz_world_np_for_sh[idx]  # (3,) world position
        dir_orig = mu_w_3d - camera_center_world  # (3,) unnormalized view direction (pos - campos)
        dir_normalized = dir_orig / (np.linalg.norm(dir_orig) + 1e-8)
        
        # Compute dL/d(dir_normalized) through SH polynomial backward
        dL_ddir = compute_sh_backward_single(dir_normalized, sh_coeffs_all[idx], dL_dRGB_i, deg=3)
        
        # Backprop through normalization: dL/d(dir_orig)
        dL_dmean_sh = dnormvdv(dir_orig, dL_ddir)  # (3,)
        
        # d(dir_orig)/d(ρ) = d(pos - campos)/d(ρ)
        # Under left perturbation, d(campos)/d(ρ) ≈ I (for small perturbation)
        # So d(dir_orig)/d(ρ) = -d(campos)/d(ρ) = -I
        # dL/d(ρ) = dL/d(dir_orig) @ d(dir_orig)/d(ρ) = dL_dmean_sh @ (-I) = -dL_dmean_sh
        dL_dtau_sh_i = np.zeros(6, dtype=np.float64)
        dL_dtau_sh_i[0:3] = -dL_dmean_sh
        
        dL_dtau_sh_total += dL_dtau_sh_i

        if i < debug_count:
            print(f"\nGaussian {i} (idx={idx}):")
            print(f"  dL_dmuI: {dL_dmuI}")
            print(f"  dmuI_dtau:\n{dmuI_dtau}")
            print(f"  dL_dtau_muI: {dL_dtau_muI}")
            print(f"  dL_dCovI: {dL_dCovI}")
            print(f" dcovI_dtau:\n{dcovI_dtau}")
            print(f"  dL_dtau_covI: {dL_dtau_covI}")
            print(f"  dL_dtau_sh_i: {dL_dtau_sh_i}")
            print(f"  dL_dRGB_i: {dL_dRGB_i}")

        # Total accumulation
        dL_dtau_mu_total += dL_dtau_muI
        dL_dtau_cov_total += dL_dtau_covI
        dL_dtau = dL_dtau + dL_dtau_muI + dL_dtau_covI + dL_dtau_depth_i + dL_dtau_sh_i
    
    print("\n=== Final dL/dtau Breakdown ===")
    print(f"Mean contribution:       {dL_dtau_mu_total}")
    print(f"Covariance contribution: {dL_dtau_cov_total}")
    print(f"Depth contribution:      {dL_dtau_depth_total}")
    print(f"SH view-dir contribution:{dL_dtau_sh_total}")
    print(f"Total dL/dtau:           {dL_dtau}")
    

    '''
    First Run gave dL_dtau as: 

    array([-0.50577958, -1.29176557, -0.13440269,  1.46791875, -0.65785952, 0.36385095])



    '''
    print(" dL_dtau from Analytical formula: ", dL_dtau)
    np.save('./Jacob_test_result/dL_dtau.npy', dL_dtau)


    # Jacobian from the MonoGS Rasterizer
    '''
    grad_tau (dL_dtau) :   tensor([ 0.1546,  0.3316, -0.9618, -0.1568, -0.2514, -0.0860], device='cuda:0')

    dL_dtau for small 15 gaussians: tensor([ 5.8907e-03,  5.5374e-03,  1.8062e-02, -8.7316e-03,  1.1688e-02, 4.2078e-05], device='cuda:0')
    '''
    print("\nJacobian from the MonoGS Rasterizer: ")
    loss = compute_loss(gaussian_model, render_image, render_depth, viewpoint.original_image, viewpoint.depth, mask_tensor, compute_depth_loss=True)
    loss.backward()







