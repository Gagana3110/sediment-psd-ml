"""Summary statistics of a particle size distribution (PSD)."""
import numpy as np


def bin_edges(centers):
    """Geometric bin edges for log-spaced LISST size-class centres."""
    logc = np.log(centers)
    mid = (logc[:-1] + logc[1:]) / 2
    lo = logc[0] - (mid[0] - logc[0])
    hi = logc[-1] + (logc[-1] - mid[-1])
    return np.exp(np.concatenate([[lo], mid, [hi]]))


def psd_stats(vconc, centers):
    """d50 and spread of volume-concentration spectra.

    vconc   : (n_samples, n_bins) volume concentration per size class
    centers : (n_bins,) size-class centre diameters in µm
    Returns d50 (µm), sigma (µm) - the volume-weighted standard deviation of
    diameter, i.e. the square root of the variance used in the reference paper.
    """
    vconc = np.atleast_2d(np.asarray(vconc, dtype=float))
    total = vconc.sum(axis=1, keepdims=True)
    p = np.divide(vconc, total, out=np.zeros_like(vconc), where=total > 0)

    log_upper = np.log(bin_edges(centers)[1:])
    cdf = np.cumsum(p, axis=1)
    d50 = np.array([np.exp(np.interp(0.5, c, log_upper)) for c in cdf])

    mean = (p * centers).sum(axis=1)
    sigma = np.sqrt((p * (centers - mean[:, None]) ** 2).sum(axis=1))

    bad = total[:, 0] <= 0
    d50[bad] = np.nan
    sigma[bad] = np.nan
    return d50, sigma
