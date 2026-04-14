import sys
import os
import warnings
import tempfile
import numpy as np
import shutil
import time
import traceback

warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
warnings.filterwarnings(
    "ignore",
    message=".*tpfmodel submodule is not available.*",
    category=UserWarning
)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache", "lightcurves")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
TEMPLATE_CSV = os.path.join(PROJECT_ROOT, "data", "raw", "koi_template.csv")
META_PATH = os.path.join(OUTPUT_DIR, "metadata.csv")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

warnings.filterwarnings("ignore", message="File may have been truncated", module="astropy")

BIN_TIME = 0.01
PERIOD_MIN_DEFAULT = 0.1
PERIOD_MAX_DEFAULT = 750.0
HINT_WINDOW_FACTOR = 0.20
OVERSAMPLING = 4
N_SAMPLES = 3000
MAX_KIC = 20000

def atomic_save_npy(path, arr):
    dirn = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix="tmp_npy_", dir=dirn)
    os.close(fd)
    with open(tmp_path, "wb") as f:
        np.save(f, arr)
        f.flush()
        os.fsync(f.fileno())
    time.sleep(0.05)
    try:
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass

def clear_lightkurve_cache_for_kic(kic_id):
    base = os.path.expanduser("~/.lightkurve/cache/mastDownload/Kepler/")
    if not os.path.exists(base):
        return
    for root, dirs, _ in os.walk(base):
        for d in dirs:
            if str(kic_id) in d:
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)

def download_lightcurve_cached_protected(kic_id):
    import lightkurve as lk
    npy_cache = os.path.join(CACHE_DIR, f"KIC_{kic_id}.npy")

    if os.path.exists(npy_cache):
        try:
            return np.load(npy_cache, allow_pickle=True).item()
        except:
            try:
                os.remove(npy_cache)
            except:
                pass

    for attempt in range(2):
        try:
            search = lk.search_lightcurve(f"KIC {kic_id}", author="Kepler")
            if len(search) == 0:
                return None

            lc = search.download_all().stitch().remove_nans().remove_outliers()
            if BIN_TIME is not None:
                lc = lc.bin(time_bin_size=BIN_TIME)

            payload = {"time": lc.time.value, "flux": lc.flux.value}
            atomic_save_npy(npy_cache, payload)
            return payload

        except Exception as e:
            print(f"[download] KIC {kic_id}: retry ({e})")
            clear_lightkurve_cache_for_kic(kic_id)
            time.sleep(0.2)

    print(f"[download] KIC {kic_id}: failed")
    return None

def save_partial_metadata(results):
    import pandas as pd
    if not results:
        return
    newdf = pd.DataFrame(results)
    if os.path.exists(META_PATH):
        old = pd.read_csv(META_PATH)
        newdf = pd.concat([old, newdf], ignore_index=True)
        newdf = newdf.drop_duplicates(subset=["koi_name"]) 
    newdf.to_csv(META_PATH, index=False)
    print("metadata update")

def process_single_kic(task_args):
    (idx, row, kic_column, oversampling, period_min, period_max) = task_args

    import numpy as np 
    import os
    import lightkurve as lk
    from transitleastsquares import transitleastsquares as tls
    from scipy import stats 
    from src.utils.pipeline_utils import extract_and_stack_transits

    kic_id = int(row[kic_column])
    
    koi_name = row.get("kepoi_name", f"KIC{kic_id}_P{idx}")

    gpath = os.path.join(OUTPUT_DIR, f"global_view_{koi_name}.npy")
    lpath = os.path.join(OUTPUT_DIR, f"local_view_{koi_name}.npy")

    if os.path.exists(gpath) and os.path.exists(lpath):
        return None

    hint = np.nan
    for col in ["period_days", "koi_period", "period", "orbital_period"]:
        if col in row and np.isfinite(row[col]):
            hint = float(row[col])
            break
    if np.isfinite(hint) and hint > PERIOD_MAX_DEFAULT:
        print(f"{koi_name}: hint muito alto ({hint}), ignorando hint")
        hint = np.nan

    print(
        f"[PID {os.getpid()}] {koi_name} (KIC {kic_id}): processing... "
        f"(Hint Period: {hint if np.isfinite(hint) else 'BLIND SEARCH'})"
    )

    cached = download_lightcurve_cached_protected(kic_id)
    if cached is None:
        return None

    time_arr = np.asarray(cached["time"], float)
    flux_arr = np.asarray(cached["flux"], float)

    nan_count = np.isnan(flux_arr).sum()
    valid_count = len(flux_arr) - nan_count
    if valid_count < 1500:
        print(f"[PID {os.getpid()}] {koi_name}: SKIP (points={valid_count})")
        return None

    if not np.all(np.diff(time_arr) > 0):
        idx_sort = np.argsort(time_arr)
        time_arr, flux_arr = time_arr[idx_sort], flux_arr[idx_sort]

    baseline_days = time_arr[-1] - time_arr[0]
    if baseline_days <= 0:
        return None

    if BIN_TIME is not None:
        theoretical_points = baseline_days / BIN_TIME
    else:
        theoretical_points = len(time_arr)
    duty_cycle = valid_count / theoretical_points
    if duty_cycle < 0.05:
        return None

    if np.isfinite(hint) and hint > 0:
        expected_transits = baseline_days / hint
        if expected_transits < 2:
            return None

    p_min = period_min
    p_max = min(period_max, PERIOD_MAX_DEFAULT)

    if np.isfinite(hint) and hint > 0:
        window = max(0.3 * hint, 1.0) 
        p_min = max(0.1, hint - window)
        p_max = min(hint + window, PERIOD_MAX_DEFAULT)


    else:
        p_min = 0.5
        p_max = min(PERIOD_MAX_DEFAULT, baseline_days / 2)
    

    

    if p_min >= p_max:
        return None
    
    flux_median = np.nanmedian(flux_arr)
    if flux_median <= 0 or np.isnan(flux_median):
        return None

    flux_norm = flux_arr / flux_median

   
    mask_finite = np.isfinite(time_arr) & np.isfinite(flux_norm)
    time_clean = time_arr[mask_finite]
    flux_clean = flux_norm[mask_finite]
    
    if len(flux_clean) < 1000:
        return None

    try:
        lc_temp = lk.LightCurve(time=time_clean, flux=flux_clean)
        lc_flat = lc_temp.flatten(window_length=501).flux.value
        flux_clean = lc_flat
    except Exception as e:
        print(f"[PID {os.getpid()}] {koi_name}: Detrending failed: {e}")

    mask_final = np.isfinite(flux_clean)
    time_clean = time_clean[mask_final]
    flux_clean = flux_clean[mask_final]

    if np.nanstd(flux_clean) == 0:
        print(f"[PID {os.getpid()}] {koi_name}: SKIP (Flux variation is zero)")
        return None
    
    print(f"[DEBUG] {koi_name} → p_min={p_min}, p_max={p_max}, baseline={baseline_days}, BIN_TIME={BIN_TIME}")
    ####avaliar melhor depois
    try:
        model = tls(time_clean, flux_clean)
        
        if BIN_TIME is not None:
            p_min_effective = max(p_min, 2 * BIN_TIME)
        else:
            p_min_effective = p_min
        p_max_effective = min(p_max, baseline_days / 2)
        
        if p_min_effective >= p_max_effective:
            return None
        
        res = model.power(
            period_min=p_min_effective,
            period_max=p_max_effective,
            oversampling_factor=3,
            use_threads=11,
            transit_depth_min=1e-5,  
            duration_grid_step=1.1,
            n_transits_min=2, 
            transit_template='default'
        )
        
        if not hasattr(res, 'SDE'):
            return None
            
        sde = float(res.SDE)
        min_sde = 3.0 if (np.isfinite(hint) and hint > 0) else 5.0
        
    
        if np.isnan(sde) or sde < min_sde:
            if np.isfinite(hint) and hint > 0:
                print(f"[PID {os.getpid()}] {koi_name}: SDE baixo ({sde:.2f}), mas é CONFIRMED. Forçando extração pelo Hint!")
                
                res.period = hint
                
                if hasattr(res, 'periods') and hasattr(res, 'power') and len(res.periods) > 0:
                    idx_hint = np.argmin(np.abs(res.periods - hint))
                    sde_real = float(res.power[idx_hint])
                    
                    res.SDE = sde_real 
                    sde = sde_real
                    print(f"    -> SDE real do Hint ({hint:.4f}d) resgatado da memória: {sde_real:.2f}")
                else:
                    res.SDE = 0.0
                    sde = 0.0

                t0_keys = ['koi_time0bk', 'time0bk', 'epoch', 'koi_time0']
                found_t0 = False
                for k in t0_keys:
                    if k in row and np.isfinite(row[k]):
                        res.T0 = float(row[k])
                        found_t0 = True
                        break
                if not found_t0:
                    print(f"T0 não encontrado no CSV, usando o chute do TLS: {res.T0}")

                dur_keys = ['koi_duration', 'duration']
                for k in dur_keys:
                    if k in row and np.isfinite(row[k]):
                        res.duration = float(row[k]) / 24.0
                        break
                if not hasattr(res, 'duration') or np.isnan(res.duration) or res.duration <= 0:
                    res.duration = 0.1 

                depth_keys = ['koi_depth', 'depth']
                for k in depth_keys:
                    if k in row and np.isfinite(row[k]):
                        res.depth = 1.0 - (float(row[k]) / 1e6)
                        break
                if not hasattr(res, 'depth') or np.isnan(res.depth):
                    res.depth = 0.99

            else:
                print(f"[PID {os.getpid()}] {koi_name}: SDE baixo ({sde:.2f}), descartando.")
                return None

    except Exception as e:
        print(f"[PID {os.getpid()}] {koi_name}: TLS error: {e}")
        return None
    try:
        lc = lk.LightCurve(time=time_arr, flux=flux_arr)
        folded = lc.fold(period=res.period, epoch_time=res.T0)

        gflux = folded.flux.value
        gflux = (gflux - np.nanmean(gflux)) / np.nanstd(gflux)

        dur = float(res.duration) if hasattr(res, "duration") else res.period / 20

        _, local = extract_and_stack_transits(
            lc, res.period, res.T0, dur,
            half_window_factor=4.0,
            n_samples=N_SAMPLES
        )

        if local is None or np.isnan(local).mean() > 0.70:
            return None

        atomic_save_npy(gpath, gflux)
        atomic_save_npy(lpath, local)

        return {
            "koi_name": koi_name,
            "kepler_name": row.get("kepler_name", ""),
            "kic_id": kic_id,
            "period_days": res.period,
            "SDE": res.SDE,
            "odd_even_mismatch": getattr(res, "odd_even_mismatch", np.nan),
            "duration_days": dur,
            "global_npy": gpath,
            "local_npy": lpath,
        }

    except Exception as e:
        print(f"[PID {os.getpid()}] {koi_name}: Error creating views: {e}")
        return None


if __name__ == "__main__":
    try:
        import pandas as pd
        from src.utils.data_io import load_csv
    except:
        print("Import error:")
        traceback.print_exc()
        raise

    try:
        print("Starting tls_dataset.py")

        df = load_csv(TEMPLATE_CSV)
        df = df[df["koi_disposition"].isin(["CONFIRMED"])]
        df = df.reset_index(drop=True)
        print(":", len(df))

        kic_column = next((c for c in df.columns if "kepid" in c.lower() or "kic" in c.lower()), None)
        if not kic_column:
            kic_column = 'kepid' if 'kepid' in df.columns else df.columns[0]

        processed_kois = set()
        processed_old_kics = set()
        if os.path.exists(META_PATH):
            try:
                meta = pd.read_csv(META_PATH)
                if "koi_name" in meta.columns:
                    processed_kois = set(meta["koi_name"].dropna().values)
                if "kic_id" in meta.columns:
                    processed_old_kics = set(meta["kic_id"].dropna().values)
                print(f"Memória carregada: {len(processed_kois)} KOIs novos, {len(processed_old_kics)} KICs antigos.")
            except:
                pass

        valid_count = len(processed_kois) if processed_kois else len(processed_old_kics)
        results = []
        i = 0
        seen_kics_in_this_run = set()

        while i < len(df) and i < MAX_KIC:
            row = df.iloc[i]
            kic_id = int(row[kic_column])
            koi_name = row.get("kepoi_name", f"KIC{kic_id}_P{i}")

     
            is_first_planet = kic_id not in seen_kics_in_this_run
            seen_kics_in_this_run.add(kic_id)


            if koi_name in processed_kois:
                i += 1
                continue

            if is_first_planet and kic_id in processed_old_kics:
                i += 1
                continue       
            
           
            out = process_single_kic(
                (i, row, kic_column,
                 OVERSAMPLING, PERIOD_MIN_DEFAULT, PERIOD_MAX_DEFAULT)
            )

            if out is not None:
                results.append(out)
                valid_count += 1
                print(f" → Valid KOI Saved: {koi_name} (Total acumulado: {valid_count})")

                if len(results) >= 1:
                    save_partial_metadata(results)
                    results.clear()

            i += 1

        if results:
            save_partial_metadata(results)
            results.clear()

        print("Nenhum planeta repetido.")

    except:
        print("Fatal error in main:")
        traceback.print_exc()
        raise