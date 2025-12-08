import numpy as np

def compute_gradients_2D(gaussians, rendered_color, rendered_depth, 
                         gt_color, gt_depth, image_size):
    """
    Compute ∂L/∂μ_I and ∂L/∂Σ_I for all Gaussians.
    
    Args:
        gaussians: List of projected 2D Gaussians (with mu_I, Sigma_I, etc.)
        rendered_color: (H, W, 3) rendered color image
        rendered_depth: (H, W) rendered depth image
        gt_color: (H, W, 3) ground truth color
        gt_depth: (H, W) ground truth depth
        image_size: (H, W)
    
    Returns:
        grad_mu_I: List of gradients ∂L/∂μ_I for each Gaussian (each is 2D vector)
        grad_Sigma_I: List of gradients ∂L/∂Σ_I for each Gaussian (each is 2×2 matrix)
    """
    H, W = image_size
    num_gaussians = len(gaussians)
    
    # Initialize gradients
    grad_mu_I = [np.zeros(2) for _ in range(num_gaussians)]
    grad_Sigma_I = [np.zeros((2, 2)) for _ in range(num_gaussians)]
    
    # Step 1: Compute pixel-level loss gradients
    # Using L1 loss here
    grad_color = np.sign(rendered_color - gt_color)  # (H, W, 3)
    grad_depth = np.sign(rendered_depth - gt_depth)  # (H, W)
    
    # Step 2: For each pixel, backpropagate through rendering
    for v in range(H):
        for u in range(W):
            pixel_pos = np.array([u, v])
            
            # Get loss gradient at this pixel
            dL_dC = grad_color[v, u]  # (3,)
            dL_dD = grad_depth[v, u]  # scalar
            
            # Sort Gaussians by depth for this pixel
            gaussians_with_idx = [(i, g) for i, g in enumerate(gaussians)]
            gaussians_sorted = sorted(gaussians_with_idx, 
                                     key=lambda x: x[1]['depth'])
            
            # Forward pass: compute alphas and transmittances
            alphas = []
            transmittances = []
            T = 1.0
            
            for idx, (gauss_idx, g) in enumerate(gaussians_sorted):
                alpha = compute_alpha_at_pixel(g, pixel_pos)
                alphas.append(alpha)
                transmittances.append(T)
                T *= (1 - alpha)
            
            # Backward pass: compute gradient w.r.t. alphas
            grad_alphas = [0.0] * len(gaussians_sorted)
            
            for idx, (gauss_idx, g) in enumerate(gaussians_sorted):
                alpha_i = alphas[idx]
                T_i = transmittances[idx]
                
                # Compute accumulated contribution after i
                color_after = np.zeros(3)
                depth_after = 0.0
                
                for j in range(idx + 1, len(gaussians_sorted)):
                    alpha_j = alphas[j]
                    T_j = transmittances[j]
                    g_j = gaussians_sorted[j][1]
                    
                    color_after += g_j['color'] * alpha_j * T_j
                    depth_after += g_j['depth'] * alpha_j * T_j
                
                # Gradient of rendered color w.r.t. alpha_i
                dC_dalpha_i = g['color'] * T_i
                if alpha_i < 0.999:  # Avoid division by zero
                    dC_dalpha_i -= color_after / (1 - alpha_i)
                
                # Gradient of rendered depth w.r.t. alpha_i
                dD_dalpha_i = g['depth'] * T_i
                if alpha_i < 0.999:
                    dD_dalpha_i -= depth_after / (1 - alpha_i)
                
                # Chain rule: ∂L/∂α_i
                dL_dalpha_i = np.dot(dL_dC, dC_dalpha_i) + dL_dD * dD_dalpha_i
                grad_alphas[idx] = dL_dalpha_i
            
            # Step 3: Backprop from alphas to μ_I and Σ_I
            for idx, (gauss_idx, g) in enumerate(gaussians_sorted):
                dL_dalpha = grad_alphas[idx]
                alpha = alphas[idx]
                
                if abs(alpha) < 1e-8:  # Skip if alpha is too small
                    continue
                
                # Compute Δ = p - μ_I
                Delta = pixel_pos - g['mu_I']  # (2,)
                
                # Compute Σ_I^{-1}
                Sigma_I_inv = np.linalg.inv(g['Sigma_I'])
                
                # Gradient w.r.t. μ_I: ∂α/∂μ_I = α * Σ^{-1} * Δ
                dalpha_dmu = alpha * (Sigma_I_inv @ Delta)  # (2,)
                
                # Chain rule: ∂L/∂μ_I
                grad_mu_I[gauss_idx] += dL_dalpha * dalpha_dmu
                
                # Gradient w.r.t. Σ_I: ∂α/∂Σ_I = 0.5 * α * Σ^{-1} * ΔΔ^T * Σ^{-1}
                dalpha_dSigma = 0.5 * alpha * (
                    Sigma_I_inv @ np.outer(Delta, Delta) @ Sigma_I_inv
                )  # (2, 2)
                
                # Chain rule: ∂L/∂Σ_I
                grad_Sigma_I[gauss_idx] += dL_dalpha * dalpha_dSigma
    
    return grad_mu_I, grad_Sigma_I


def compute_alpha_at_pixel(gaussian, pixel_pos):
    """
    Compute alpha value for a Gaussian at a pixel.
    
    Args:
        gaussian: Dict with 'mu_I', 'Sigma_I', 'opacity'
        pixel_pos: (u, v) pixel coordinates
    
    Returns:
        alpha: Float
    """
    Delta = pixel_pos - gaussian['mu_I']
    Sigma_inv = np.linalg.inv(gaussian['Sigma_I'])
    
    # Mahalanobis distance
    power = -0.5 * Delta @ Sigma_inv @ Delta
    
    # Gaussian value
    gaussian_val = np.exp(power)
    
    # Alpha
    alpha = gaussian['opacity'] * gaussian_val
    
    return np.clip(alpha, 0.0, 1.0)
    
