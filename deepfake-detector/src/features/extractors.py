
import numpy as np
import cv2
import pywt
from scipy.stats import kurtosis
try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    from skimage.feature import greycomatrix as graycomatrix, greycoprops as graycoprops
from skimage.feature import local_binary_pattern

def extract_fft_features(gray_img):
    f = np.fft.fft2(gray_img)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    phase = np.angle(fshift)
    mag_log = np.log1p(magnitude)
    h, w = gray_img.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_r = radius.max()
    low_mask = radius <= max_r * 0.15
    mid_mask = (radius > max_r * 0.15) & (radius <= max_r * 0.5)
    high_mask = radius > max_r * 0.5
    low_e = mag_log[low_mask].sum()
    mid_e = mag_log[mid_mask].sum()
    high_e = mag_log[high_mask].sum()
    r_int = radius.astype(int)
    radial_sum = np.bincount(r_int.ravel(), mag_log.ravel())
    radial_cnt = np.bincount(r_int.ravel())
    radial_profile = radial_sum / (radial_cnt + 1e-8)
    radial_profile = radial_profile[: int(max_r)]
    bins = np.array_split(radial_profile, 8) if len(radial_profile) >= 8 else [radial_profile]
    radial_bin_means = [b.mean() if len(b) else 0 for b in bins]
    while len(radial_bin_means) < 8:
        radial_bin_means.append(0)
    hist, _ = np.histogram(mag_log, bins=256, density=True)
    hist = hist + 1e-12
    spectral_entropy = -np.sum(hist * np.log2(hist))
    feats = {
        "fft_low_energy": low_e, "fft_mid_energy": mid_e, "fft_high_energy": high_e,
        "fft_high_low_ratio": high_e / (low_e + 1e-8),
        "fft_spectral_entropy": spectral_entropy,
        "fft_spectral_variance": float(mag_log.var()),
        "fft_spectral_kurtosis": float(kurtosis(mag_log.ravel())),
        "fft_phase_mean": float(phase.mean()), "fft_phase_std": float(phase.std()),
    }
    for i, v in enumerate(radial_bin_means[:8]):
        feats[f"fft_radial_bin_{i}"] = float(v)
    return feats

def extract_dct_features(gray_img):
    gray_f = np.float32(gray_img) / 255.0
    dct = cv2.dct(gray_f)
    h, w = dct.shape
    low = dct[: h // 4, : w // 4]
    high = dct[h // 2 :, w // 2 :]
    low_e = float(np.sum(low ** 2))
    high_e = float(np.sum(high ** 2))
    return {
        "dct_coeff_mean": float(dct.mean()), "dct_coeff_var": float(dct.var()),
        "dct_low_energy": low_e, "dct_high_energy": high_e,
        "dct_band_ratio": high_e / (low_e + 1e-8),
    }

def extract_dwt_features(gray_img):
    LL, (LH, HL, HH) = pywt.dwt2(gray_img, "haar")
    feats = {}
    for name, band in [("LL", LL), ("LH", LH), ("HL", HL), ("HH", HH)]:
        feats[f"dwt_{name}_energy"] = float(np.sum(band ** 2))
        feats[f"dwt_{name}_mean"] = float(band.mean())
        feats[f"dwt_{name}_var"] = float(band.var())
        feats[f"dwt_{name}_kurtosis"] = float(kurtosis(band.ravel()))
        hist, _ = np.histogram(band, bins=64, density=True)
        hist = hist + 1e-12
        feats[f"dwt_{name}_entropy"] = float(-np.sum(hist * np.log2(hist)))
    return feats

def extract_noise_features(gray_img):
    gray_f = gray_img.astype(np.float32)
    blur = cv2.GaussianBlur(gray_f, (3, 3), 0)
    residual = gray_f - blur
    laplacian = cv2.Laplacian(gray_f, cv2.CV_32F)
    hist, _ = np.histogram(residual, bins=64, density=True)
    hist = hist + 1e-12
    residual_entropy = -np.sum(hist * np.log2(hist))
    return {
        "noise_residual_mean": float(residual.mean()), "noise_residual_var": float(residual.var()),
        "noise_residual_entropy": float(residual_entropy),
        "noise_residual_kurtosis": float(kurtosis(residual.ravel())),
        "noise_laplacian_var": float(laplacian.var()),
    }

def extract_texture_features(gray_img):
    gray_u8 = gray_img.astype(np.uint8)
    lbp = local_binary_pattern(gray_u8, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    gray_64 = (gray_u8 // 4).astype(np.uint8)
    glcm = graycomatrix(gray_64, distances=[1], angles=[0], levels=64, symmetric=True, normed=True)
    contrast = graycoprops(glcm, "contrast")[0, 0]
    correlation = graycoprops(glcm, "correlation")[0, 0]
    energy = graycoprops(glcm, "energy")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
    edges = cv2.Canny(gray_u8, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size
    feats = {
        "glcm_contrast": float(contrast), "glcm_correlation": float(correlation),
        "glcm_energy": float(energy), "glcm_homogeneity": float(homogeneity),
        "edge_density": float(edge_density),
    }
    for i, v in enumerate(lbp_hist):
        feats[f"lbp_bin_{i}"] = float(v)
    return feats

def extract_all_features(gray_img):
    feats = {}
    feats.update(extract_fft_features(gray_img))
    feats.update(extract_dct_features(gray_img))
    feats.update(extract_dwt_features(gray_img))
    feats.update(extract_noise_features(gray_img))
    feats.update(extract_texture_features(gray_img))
    return feats
