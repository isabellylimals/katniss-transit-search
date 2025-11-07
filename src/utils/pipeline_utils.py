import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

def safe_to_numpy(x):
    if hasattr(x, "filled"):
        x = x.filled(np.nan)
    return np.asarray(x, dtype=float)

def detrend_local(time, flux, window_days=0.5, polydeg=1):
    mask = np.isfinite(flux)
    if mask.sum() < 5:
        return flux - np.nanmean(flux)
    try:
        base = savgol_filter(flux[mask], 101, 2)
        f_interp = interp1d(time[mask], base, kind="linear", fill_value="extrapolate")
        return flux - f_interp(time)
    except Exception:
        return flux - np.nanmedian(flux)

def extract_and_stack_transits(lc, period, t0, duration_days, half_window_factor=2.0, n_samples=1001):
    time = safe_to_numpy(lc.time.value)
    flux = safe_to_numpy(lc.flux.value)

    tmin, tmax = np.nanmin(time), np.nanmax(time)
    epochs = np.arange(np.floor((tmin - t0) / period) - 1, np.ceil((tmax - t0) / period) + 1) * period + t0
    epochs = epochs[(epochs > tmin) & (epochs < tmax)]

    half_window = half_window_factor * duration_days
    grid = np.linspace(-half_window, +half_window, n_samples)
    stack = []

    for e in epochs:
        sel = (time >= e - half_window) & (time <= e + half_window)
        if sel.sum() < 6:
            continue
        t_local = (time[sel] - e)
        f_local = flux[sel]
        f_local = detrend_local(t_local, f_local, window_days=half_window)

        oot_mask = (np.abs(t_local) > duration_days)
        if oot_mask.sum() >= 3:
            base = np.nanmedian(f_local[oot_mask])
            f_local = (f_local - base)
        else:
            f_local = f_local - np.nanmedian(f_local)

        f_local = np.clip(f_local, -5, 5)  # Protege contra spikes extremos

        try:
            f_i = interp1d(t_local, f_local, bounds_error=False, fill_value=np.nan)(grid)
            stack.append(f_i)
        except Exception:
            continue

    if len(stack) == 0:
        return grid, np.full_like(grid, np.nan)

    stack = np.vstack(stack)
    stack = np.where(np.isnan(stack), np.nanmedian(stack, axis=0), stack)  # Corrige NaNs nas bordas

    stacked = np.nanmedian(stack, axis=0)

    if np.nanstd(stacked) == 0 or not np.isfinite(np.nanstd(stacked)):
        stacked = stacked - np.nanmedian(stacked)
    else:
        stacked = (stacked - np.nanmedian(stacked)) / np.nanstd(stacked)

    return grid, stacked