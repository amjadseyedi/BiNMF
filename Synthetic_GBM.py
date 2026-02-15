"""
GBM hyperspectral cube generator with spatially-coherent (segment-like) abundances
+ pseudo-RGB plot + .mat saving

Key change vs. fully random pixels:
- Abundances are generated as spatial segments (Voronoi-like regions) where each region
  has a dominant endmember and some mixing.
- Optional spatial smoothing + re-normalization (sum-to-one) gives realistic boundaries.
- Optional mild endmember variability makes regions look less "flat"/synthetic.

Model (GBM):
x = M a + sum_{i<j} gamma_ij * a_i a_j * (m_i ⊙ m_j) + noise
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from scipy.ndimage import gaussian_filter

# If you have spectral installed, this will produce nicer RGB for hyperspectral cubes
try:
    from spectral.graphics.graphics import get_rgb_meta
    HAS_SPECTRAL = True
except Exception:
    HAS_SPECTRAL = False


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)


def synth_endmembers(L: int, p: int) -> np.ndarray:
    """
    Create synthetic, smooth, positive endmembers (L bands, p endmembers).
    If you have real endmembers, replace this with your own M.
    """
    wl = np.linspace(0, 1, L)
    M = np.zeros((L, p), dtype=float)

    centers = np.linspace(0.2, 0.8, p)
    widths = np.linspace(0.06, 0.14, p)

    for i in range(p):
        bump1 = np.exp(-0.5 * ((wl - centers[i]) / widths[i]) ** 2)
        bump2 = 0.6 * np.exp(-0.5 * ((wl - (1 - centers[i])) / (widths[i] * 1.3)) ** 2)
        slope = (i + 1) * 0.15 * wl
        base = 0.05 + 0.03 * i
        M[:, i] = base + 0.8 * bump1 + 0.5 * bump2 + slope

    # Normalize each endmember to [0, 1]
    M -= M.min(axis=0, keepdims=True)
    M /= (M.max(axis=0, keepdims=True) + 1e-12)
    return M


def segmented_abundances(H: int, W: int, p: int,
                         n_regions: int = 30,
                         dominant_strength: float = 80.0,
                         mix_strength: float = 1.5,
                         seed: int = 0) -> np.ndarray:
    """
    Make region-wise (segment-like) abundance maps.
    Fast Voronoi-like partition using random seeds; each region has a dominant endmember.

    Returns:
        A: (p, N) with N = H*W, sum-to-one per pixel.
    """
    rng = np.random.default_rng(seed)

    # --- Voronoi partition via nearest random seeds ---
    ys = rng.integers(0, H, size=n_regions)
    xs = rng.integers(0, W, size=n_regions)

    yy, xx = np.mgrid[0:H, 0:W]
    d2 = (yy[..., None] - ys[None, None, :]) ** 2 + (xx[..., None] - xs[None, None, :]) ** 2
    region_labels = np.argmin(d2, axis=-1)

    # Dominant endmember per region
    region_dom = rng.integers(0, p, size=n_regions)

    A_maps = np.zeros((p, H, W), dtype=np.float32)

    for r in range(n_regions):
        dom = int(region_dom[r])
        alpha = np.full(p, mix_strength, dtype=np.float32)
        alpha[dom] = dominant_strength

        mask = (region_labels == r)
        n_pix = int(mask.sum())
        if n_pix == 0:
            continue

        samples = rng.dirichlet(alpha, size=n_pix).astype(np.float32)  # (n_pix, p)
        A_maps[:, mask] = samples.T

    A = A_maps.reshape(p, H * W)
    return A


def smooth_and_project_simplex(A: np.ndarray, H: int, W: int, sigma: float = 1.0) -> np.ndarray:
    """
    Smooth each abundance map spatially, then enforce:
    - nonnegativity
    - sum-to-one per pixel
    """
    p, N = A.shape
    A_maps = A.reshape(p, H, W).copy()

    for i in range(p):
        A_maps[i] = gaussian_filter(A_maps[i], sigma=sigma)

    A_maps = np.clip(A_maps, 1e-8, None)
    S = A_maps.sum(axis=0, keepdims=True)
    A_maps /= S
    return A_maps.reshape(p, N)


def apply_endmember_variability(A: np.ndarray, variability: float = 0.01, seed: int = 0) -> np.ndarray:
    """
    Mild per-pixel variability (scales abundances a bit), then re-project to simplex.
    Makes regions less "flat".
    """
    rng = np.random.default_rng(seed)
    p, N = A.shape
    scales = rng.normal(1.0, variability, size=(p, N)).astype(np.float32)
    scales = np.clip(scales, 0.8, 1.2)

    A_var = A * scales
    A_var = np.clip(A_var, 1e-8, None)
    A_var /= A_var.sum(axis=0, keepdims=True)
    return A_var


def gbm_mix(M: np.ndarray, A: np.ndarray, gamma) -> np.ndarray:
    """
    Apply GBM mixing for all pixels.

    M: (L, p)
    A: (p, N)
    gamma: scalar or (p, p) array, uses gamma[i, j] for i<j

    Returns:
        X: (L, N) mixed spectra (noise-free)
    """
    L, p = M.shape
    _, N = A.shape

    X = M @ A  # linear part

    scalar_gamma = np.isscalar(gamma)
    if not scalar_gamma:
        gamma = np.asarray(gamma, dtype=float)
        if gamma.shape != (p, p):
            raise ValueError(f"gamma must be scalar or shape (p,p) = ({p},{p}).")

    for i in range(p):
        mi = M[:, i:i + 1]  # (L, 1)
        ai = A[i:i + 1, :]  # (1, N)
        for j in range(i + 1, p):
            mj = M[:, j:j + 1]
            aj = A[j:j + 1, :]
            gij = float(gamma) if scalar_gamma else float(gamma[i, j])
            X += gij * (mi * mj) @ (ai * aj)

    return X


def add_awgn_snr(X: np.ndarray, snr_db: float | None, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Add white Gaussian noise to reach desired SNR (dB) w.r.t. signal power.
    X: (L, N)
    """
    if snr_db is None:
        return X
    if rng is None:
        rng = np.random.default_rng()

    signal_power = np.mean(X ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = rng.normal(0.0, np.sqrt(noise_power), size=X.shape)
    return X + noise


def to_cube(X: np.ndarray, H: int, W: int) -> np.ndarray:
    """X: (L, N) -> cube: (H, W, L)"""
    L, N = X.shape
    if N != H * W:
        raise ValueError("X has wrong number of pixels for given H, W.")
    return X.T.reshape(H, W, L)


def normalize01(img: np.ndarray) -> np.ndarray:
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-12:
        return np.zeros_like(img)
    return (img - mn) / (mx - mn)


def pseudo_rgb_from_bands(cube: np.ndarray, r_band: int, g_band: int, b_band: int) -> np.ndarray:
    """Create pseudo-RGB by selecting 3 bands."""
    H, W, L = cube.shape
    for idx, name in [(r_band, "r_band"), (g_band, "g_band"), (b_band, "b_band")]:
        if not (0 <= idx < L):
            raise ValueError(f"{name}={idx} out of range [0, {L - 1}].")

    rgb = np.stack([cube[:, :, r_band], cube[:, :, g_band], cube[:, :, b_band]], axis=-1)
    for c in range(3):
        rgb[:, :, c] = normalize01(rgb[:, :, c])
    return rgb


def hsi2rgb_spectral(cube: np.ndarray, bands=(29, 19, 9)) -> np.ndarray:
    """
    If spectral is installed, use get_rgb_meta for a more "HSI-like" rendering.
    Otherwise, fall back to pseudo_rgb_from_bands-ish behavior.
    """
    if HAS_SPECTRAL:
        rgb, _meta = get_rgb_meta(cube, bands=bands)
        return rgb
    # fallback: clamp bands
    H, W, L = cube.shape
    b = [min(max(int(x), 0), L - 1) for x in bands]
    return pseudo_rgb_from_bands(cube, b[0], b[1], b[2])


# -----------------------------
# Main
# -----------------------------
def main():
    set_seed(0)

    # --- Parameters ---
    H, W = 300, 300      # image size
    L = 200              # number of spectral bands
    p = 3                # number of endmembers

    # Abundance segmentation controls
    n_regions = 40
    dominant_strength = 200.0   # higher -> purer segments
    mix_strength = 1.5         # higher -> more within-segment mixing
    smooth_sigma = 1.0         # 0 or None to disable smoothing
    variability = 1       # 0 to disable variability

    # GBM + noise
    gamma = 0.6         # lower -> less nonlinear distortion
    snr_db = 30.0              # higher -> cleaner (set None for noise-free)

    # --- Endmembers ---
    M = synth_endmembers(L, p)  # (L, p)

    # --- Abundances (segment-like) ---
    A = segmented_abundances(
        H, W, p,
        n_regions=n_regions,
        dominant_strength=dominant_strength,
        mix_strength=mix_strength,
        seed=1
    )

    if smooth_sigma is not None and smooth_sigma > 0:
        A = smooth_and_project_simplex(A, H, W, sigma=smooth_sigma)

    if variability is not None and variability > 0:
        A = apply_endmember_variability(A, variability=variability, seed=1)

    # --- Mix with GBM ---
    X_clean = gbm_mix(M, A, gamma=gamma)                # (L, N)
    X_noisy = add_awgn_snr(X_clean, snr_db=snr_db)      # (L, N)

    # --- Convert to cube ---
    cube = to_cube(X_noisy, H, W)                       # (H, W, L)

    # --- Display RGB ---
    rgb1 = hsi2rgb_spectral(cube, bands=(29, 19, 9))
    plt.figure(figsize=(6, 6))
    plt.title("HSI cube rendered to RGB (spectral if available)")
    plt.imshow(rgb1)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # --- Pseudo-RGB plot ---
    r_band, g_band, b_band = int(0.75 * (L - 1)), int(0.50 * (L - 1)), int(0.25 * (L - 1))
    rgb2 = pseudo_rgb_from_bands(cube, r_band, g_band, b_band)

    plt.figure(figsize=(6, 6))
    plt.title(f"Pseudo-RGB (bands {r_band}, {g_band}, {b_band})")
    plt.imshow(rgb2)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # --- Show endmembers ---
    plt.figure(figsize=(7, 4))
    for i in range(p):
        plt.plot(M[:, i], label=f"Endmember {i + 1}")
    plt.title("Endmember spectra (synthetic)")
    plt.xlabel("Band index")
    plt.ylabel("Reflectance (normalized)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Save to MAT ---
    mdic = {"X": X_noisy, "M": M, "A": A, "w": W, "h": H}
    out_name = f"GBM_segmented_regions_{n_regions}_alpha_dom_{dominant_strength}_mix_{mix_strength}_gamma_{gamma}_snr_{snr_db}.mat"
    savemat(out_name, mdic)
    print(f"Saved: {out_name}")


if __name__ == "__main__":
    main()
