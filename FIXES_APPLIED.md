# Critical Fixes Applied to Analytical Jacobian Verification

## Summary of Issues Found

### 1. **CRITICAL: Sign Error in Covariance Gradient** ✅ FIXED
**Location**: Line ~822 in `compute_gradients_2D_vectorized_chunked`

**Problem**: You had a negative sign in the covariance gradient computation:
```python
dalpha_dSigma = -0.5 * alphas_all[:, :, None, None] * ...
```

**Why this is wrong**: The derivative of the Gaussian function w.r.t. the covariance matrix Σ is:

$$\frac{\partial}{\partial \Sigma} \exp\left(-\frac{1}{2} \Delta^T \Sigma^{-1} \Delta\right) = \frac{1}{2} \exp(...) \cdot \Sigma^{-1} \Delta \Delta^T \Sigma^{-1}$$

The **positive 1/2** comes from the chain rule when differentiating w.r.t. Σ (not Σ^{-1}).

**Fix Applied**:
```python
dalpha_dSigma = 0.5 * alphas_all[:, :, None, None] * np.einsum('cnij,njk->cnik', temp1, covs_inv)
```

---

### 2. **CRITICAL: Coordinate Space Inconsistency** ✅ FIXED
**Location**: `GetImagePlaneMeanAndCovs` function and gradient computation

**Problem**: Your analytical Jacobian `dmuI_dTcw` and `dcovI_dTcw` work in **normalized image coordinates** (before applying camera intrinsics K), but your loss gradients were computed in **pixel space** (after applying K).

This created a mismatch:
- Analytical: ∂(normalized)/∂T
- Numerical: ∂L/∂(pixel)

When you multiply them together with chain rule, you get wrong units!

**Fix Applied**:
1. Modified `GetImagePlaneMeanAndCovs` to store **normalized** coordinates and covariances:
   ```python
   # OLD (pixel space):
   u = fx * (x/z) + cx
   v = fy * (y/z) + cy
   cov_pixel = K @ cov_normalized @ K^T
   
   # NEW (normalized space):
   x_n = x/z
   y_n = y/z
   cov_2D_normalized = JW @ cov_3D @ (JW)^T  # No K applied
   ```

2. Modified gradient computation to work in **normalized pixel coordinates**:
   ```python
   # Convert pixel grid to normalized coordinates
   x_n = (u_coords - cx) / fx
   y_n = (v_coords - cy) / fy
   ```

3. Removed the incorrect K^{-1} transformation at the end since everything is now consistent in normalized space.

**Why this matters**: The analytical Jacobian from MonoGS paper derives ∂μ_I/∂T where μ_I is in normalized coords. Your loss gradients must be in the same space!

---

### 3. **Remaining Considerations**

#### Color Computation Issue (Not Fixed - Needs Verification)
Your colors are computed once for all Gaussians using viewing directions from camera origin:
```python
viewdirs = compute_viewing_directions(xyz_cam, camera_position=[0,0,0])
colors = compute_colors_from_sh(gaussian_model.get_features, viewdirs, deg=3)
```

**Potential Issue**: In true alpha-blending rendering, each pixel sees Gaussians from slightly different viewing angles. The current implementation uses a single color per Gaussian computed from the camera center, which is an approximation.

**Impact**: This might cause small errors in gradient computation, especially for Gaussians with high-frequency SH components. However, for degree-3 SH, the error should be minimal for objects not too close to the camera.

**Recommendation**: If results are still not matching, consider computing viewing directions per-pixel in the chunked computation loop.

---

## Testing Procedure

After these fixes, run your script and check:

1. **Sign consistency**: dL/dtau should have physically reasonable signs
2. **Magnitude check**: Compare with numerical gradients from autodiff
3. **Convergence test**: Use the analytical gradients in a gradient descent step - does the loss decrease?

### Expected Behavior

With these fixes, your analytical Jacobian chain:
```
dL/dτ = dL/dμ_I × dμ_I/dτ + dL/dΣ_I × dΣ_I/dτ
```

Now operates entirely in **normalized image space**, which is consistent with the MonoGS paper derivation.

---

## Key Formulas (for reference)

### Projection Jacobian (Normalized coords)
$$J = \frac{\partial \pi}{\partial \mu_c} = \frac{1}{z} \begin{bmatrix} 1 & 0 & -x/z \\ 0 & 1 & -y/z \\ 0 & 0 & 0 \end{bmatrix}$$

### Mean Jacobian
$$\frac{\partial \mu_I}{\partial T_{CW}} = J \frac{\partial \mu_c}{\partial T_{CW}}$$

where $\mu_I$ is in normalized coordinates (x/z, y/z).

### Covariance Jacobian (MonoGS Eq. 4)
$$\frac{\partial \Sigma_I}{\partial T_{CW}} = \frac{\partial \Sigma_I}{\partial J}\frac{\partial J}{\partial \mu_c}\frac{\partial \mu_c}{\partial T_{CW}} + \frac{\partial \Sigma_I}{\partial W}\frac{\partial W}{\partial T_{CW}}$$

where $\Sigma_I$ is also in normalized space.

---

## Files Modified

1. `Loss_Derivative_script.py`:
   - Line ~822: Fixed sign in `dalpha_dSigma` 
   - Lines ~640-697: Changed `GetImagePlaneMeanAndCovs` to use normalized coordinates
   - Lines ~800-810: Changed pixel grid to normalized coordinates
   - Lines ~1170-1190: Removed incorrect K^{-1} transformation

---

## Next Steps

1. Run the fixed script
2. Compare `dL_dtau` with numerical gradient (finite differences)
3. If still mismatched, check:
   - Gaussian sorting order (depth-sorted vs original indices)
   - Color computation (view-dependent effects)
   - Alpha clipping behavior
4. Verify with simple test case (e.g., 1 Gaussian, known pose perturbation)

---

## Questions to Consider

1. **Are your ground truth poses (w2c_gt) and noisy poses (T_noise) correct?**
   - Check the transformation order: Is it w2c @ T_noise or T_noise @ w2c?
   
2. **Is the mask being used consistently?**
   - Your loss computes on masked regions but gradients compute on all pixels.
   
3. **Is the loss function exactly matching?**
   - You compute L1 loss but gradients use sign(). Is this intentional?

---

Good luck with the verification! The main issues were the sign error and coordinate space inconsistency. These are now fixed.
