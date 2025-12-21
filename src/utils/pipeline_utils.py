import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

def safe_to_numpy(x):
    if hasattr(x, "filled"):
        x = x.filled(np.nan)
    return np.asarray(x, dtype=float)

def detrend_local(time, flux, window_days=0.5, polydeg=2):
    mask = np.isfinite(flux)
    if mask.sum() < 5:
        return flux - np.nanmean(flux)
    
    try:
        n_points = len(flux[mask])
        
        win = min(n_points, 51)
        if win % 2 == 0:
            win -= 1
            
        win = max(win, 5)
        
        if win <= polydeg:
            return flux - np.nanmedian(flux)

        base = savgol_filter(flux[mask], win, polydeg)
        f_interp = interp1d(time[mask], base, kind="linear", fill_value="extrapolate")
        trend = f_interp(time)
        
        return flux - trend
        
    except Exception:
        return flux - np.nanmedian(flux)

def extract_and_stack_transits(lc, period, t0, duration_days, half_window_factor=2.0, n_samples=1001):
    time = safe_to_numpy(lc.time.value)
    flux = safe_to_numpy(lc.flux.value)

    tmin, tmax = np.nanmin(time), np.nanmax(time)
    
    k_min = np.floor((tmin - t0) / period)
    k_max = np.ceil((tmax - t0) / period)
    epochs = t0 + np.arange(k_min, k_max + 1) * period
    
    epochs = epochs[(epochs > tmin) & (epochs < tmax)]

    half_window = half_window_factor * duration_days
    grid = np.linspace(-half_window, +half_window, n_samples)
    stack = []

    for e in epochs:
        sel = (time >= e - half_window) & (time <= e + half_window)
        if sel.sum() < 4:
            continue
            
        t_local = time[sel] - e
        f_local = flux[sel]
        
        f_local = detrend_local(t_local, f_local, window_days=half_window)

        oot_mask = (np.abs(t_local) > duration_days / 2.0)
        if oot_mask.sum() >= 3:
            base = np.nanmedian(f_local[oot_mask])
            f_local = f_local - base
        else:
            f_local = f_local - np.nanmedian(f_local)

        try:
            interp_func = interp1d(t_local, f_local, bounds_error=False, fill_value=np.nan)
            f_i = interp_func(grid)
            stack.append(f_i)
        except Exception:
            continue

    if len(stack) < 2:
        return grid, np.full_like(grid, np.nan)

    stack = np.vstack(stack)
    
    valid_frac = np.mean(~np.isnan(stack), axis=1)
    stack = stack[valid_frac > 0.3]
    
    if len(stack) == 0:
        return grid, np.full_like(grid, np.nan)

    stacked = np.nanmedian(stack, axis=0)

    std_val = np.nanstd(stacked)
    
    if std_val == 0 or np.isnan(std_val) or std_val < 1e-6:
        return grid, np.full_like(grid, np.nan)

    stacked = (stacked - np.nanmedian(stacked)) / std_val
    stacked = np.clip(stacked, -5, 5)

    if np.all(np.isnan(stacked)):
        return grid, np.full_like(grid, np.nan)

    return grid, stacked
