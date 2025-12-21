# import sys, os, multiprocessing, warnings
# sys.path.append(os.path.abspath("."))

# if __name__ == "__main__":
#     import numpy as np
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     import lightkurve as lk
#     from transitleastsquares import transitleastsquares as tls
#     from src.utils.data_io import load_csv
#     from src.utils.pipeline_utils import extract_and_stack_transits

#     warnings.filterwarnings("ignore", category=UserWarning)
#     warnings.filterwarnings("ignore", category=RuntimeWarning)

#     TEMPLATE_CSV    = "./data/raw/koi_template.csv"
#     OUTPUT_DIR      = "./data/processed/"
#     META_PATH       = os.path.join(OUTPUT_DIR, "metadata.csv")

#     period_min      = 0.5
#     period_max      = 20.0
#     oversampling    = 2

#     env_threads = os.getenv("TLS_THREADS")
#     threads = int(env_threads) if env_threads and env_threads.isdigit() else min(multiprocessing.cpu_count(), 8)

#     BATCH_SIZE      = 10
#     MAX_KOIS_CAP    = 400
#     CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint.txt")

#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     print("Reading catalog file (KOI/NASA)...")
#     df = load_csv(TEMPLATE_CSV)

#     kic_column = next((c for c in df.columns if "kep" in c.lower()), None)
#     if kic_column is None:
#         raise Exception("ERROR: Column with Kepler ID not found in CSV (expected 'kepid').")


#     original_count = len(df)
#     df.drop_duplicates(subset=[kic_column], keep="first", inplace=True)
#     df.reset_index(drop=True, inplace=True) 
#     new_count = len(df)
#     print(f"Loaded {original_count} rows, found {new_count} unique KICs.")
    

#     TOTAL_STARS = min(MAX_KOIS_CAP, new_count)

#     if os.path.exists(CHECKPOINT_FILE):
#         with open(CHECKPOINT_FILE, "r") as f:
#             start_index = int((f.read() or "0").strip())
#     else:
#         start_index = 0

#     end_index = min(start_index + BATCH_SIZE, TOTAL_STARS)
#     print(f"\nStarting dataset generation...\n")
#     print(f"Threads: {threads} | Oversampling: {oversampling}")

#     print(f"Processing KICs {start_index+1} to {end_index} of {TOTAL_STARS}")

#     def normalize_array(x: np.ndarray) -> np.ndarray:
#         if hasattr(x, "filled"):
#             x = x.filled(np.nan)
#         x = np.asarray(x, dtype=float)
#         m = np.nanmean(x)
#         s = np.nanstd(x)
#         return (x - m) / s if np.isfinite(s) and s > 0 else (x - m)

#     for idx in range(start_index, end_index):
#         row = df.iloc[idx]
#         kic_id = int(row[kic_column])

#         koi_name = str(row["kepoi_name"]) if "kepoi_name" in row else str(kic_id)

#         print("\n" + "="*60)
        
#         print(f"Processing KIC {kic_id} ({idx+1}/{TOTAL_STARS})")
#         print("="*60)

#         try:
#             search = lk.search_lightcurve(f"KIC {kic_id}", author="Kepler")
#             if len(search) == 0:
#                 print("WARNING: No light curves found. Skipping.")
#                 continue

#             lc = search.download_all().stitch().remove_nans().remove_outliers()
#             lc = lc.bin(time_bin_size=0.02)

    
#             period_hint = row.get("period_days", np.nan)
#             if not np.isfinite(period_hint):
#                 period_hint = None

#             model = tls(lc.time.value, lc.flux.value)

#             if period_hint:
#                 print(f"→ Using KOI period hint: {period_hint:.4f} days")
#                 period_window = 0.25 * period_hint  
#                 results = model.power(
#                     period_min=max(0.5, period_hint - period_window),
#                     period_max=period_hint + period_window,
#                     oversampling_factor=oversampling,
#                     n_threads=threads,
#                 )
#             else:
#                 print("→ No period hint found, using default TLS search range")
#                 results = model.power(
#                     period_min=period_min,
#                     period_max=period_max,
#                     oversampling_factor=oversampling,
#                     n_threads=threads,
#                 )


#             print(f"Period: {results.period:.5f} days | SDE: {results.SDE:.2f}")
#             if np.isnan(results.period):
#                 print(f"WARNING: Invalid period (nan) found for KIC {kic_id}. Skipping.")
#                 continue

#             lc_folded = lc.fold(period=results.period, epoch_time=results.T0)
#             phase_global = np.asarray(lc_folded.time.value, dtype=float)
#             flux_global  = normalize_array(lc_folded.flux.value)

#             dur = float(results.duration) if hasattr(results, "duration") else results.period / 20.0
#             grid, stacked_local = extract_and_stack_transits(
#                 lc, results.period, results.T0, dur, half_window_factor=2.0, n_samples=2001
#             )
            
#             if (
#                 stacked_local is None or
#                 np.all(np.isnan(stacked_local)) or
#                 stacked_local.size == 0 or
#                 np.nanstd(stacked_local) < 0.01
#             ):
#                 print(f"WARNING: Invalid or very flat local curve for KIC {kic_id}. Skipping.")
#                 continue


#             if stacked_local is None or np.all(np.isnan(stacked_local)) or stacked_local.size == 0:
#                 print(f"WARNING: Empty or invalid stacked transit for KIC {kic_id}. Skipping.")
#                 continue

#             npy_g = os.path.join(OUTPUT_DIR, f"global_view_{koi_name}.npy")
#             np.save(npy_g, flux_global)

#             npy_l = os.path.join(OUTPUT_DIR, f"local_view_{koi_name}.npy")
#             np.save(npy_l, stacked_local)

#             pd.DataFrame([{
#                 "koi_name": koi_name,
#                 "kepler_name": row.get("kepler_name", ""),
#                 "kic_id": kic_id,
#                 "period_days": results.period,
#                 "SDE": results.SDE,
#                 "odd_even_mismatch": getattr(results, "odd_even_mismatch", np.nan),
#                 "duration_days": dur,
#                 "global_npy": npy_g,
#                 "local_npy": npy_l,
#             }]).to_csv(META_PATH, mode="a", header=not os.path.exists(META_PATH), index=False)

#             print(f"Saved GLOBAL + LOCAL dataset for KIC {kic_id}")

#         except Exception as e:
#             print(f"ERROR with KIC {kic_id}: {e}")
#             continue

#     with open(CHECKPOINT_FILE, "w") as f:
#         f.write(str(end_index))

#     print(f"\nBlock finished. {end_index}/{TOTAL_STARS} processed.")
#     if end_index >= TOTAL_STARS:
#         print("FULL DATASET GENERATED!")
#         if os.path.exists(CHECKPOINT_FILE):
#             os.remove(CHECKPOINT_FILE) 
#     else:
#         print("Run the script again for next block.")
import sys
import os
import multiprocessing
import warnings
import tempfile
import numpy as np
import shutil
import time
import traceback
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.append(os.path.abspath("."))
warnings.filterwarnings(
    "ignore",
    message=".*tpfmodel submodule is not available.*",
    category=UserWarning
)
CACHE_DIR = "./cache/lightcurves/"
OUTPUT_DIR = "./data/processed/"
TEMPLATE_CSV = "./data/raw/koi_template.csv"
META_PATH = os.path.join(OUTPUT_DIR, "metadata.csv")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

warnings.filterwarnings("ignore", message="File may have been truncated", module="astropy")

BIN_TIME = 0.04
PERIOD_MIN_DEFAULT = 0.1
PERIOD_MAX_DEFAULT = 20.0
HINT_WINDOW_FACTOR = 0.10
OVERSAMPLING = 1
N_SAMPLES = 1001
NUM_WORKERS = 1
TARGET_VALID = 1000
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


def already_processed(kic_id):
    # g = os.path.join(OUTPUT_DIR, f"global_view_KIC{kic_id}.npy")
    # l = os.path.join(OUTPUT_DIR, f"local_view_KIC{kic_id}.npy")

    # if os.path.exists(g) and os.path.exists(l):
    #     return True

    if os.path.exists(META_PATH):
        try:
            import pandas as pd
            meta = pd.read_csv(META_PATH)
            if "kic_id" in meta.columns and kic_id in meta["kic_id"].values:
                return True
        except:
            pass

    return False



def save_partial_metadata(results):
    import pandas as pd

    if not results:
        return

    newdf = pd.DataFrame(results)

    if os.path.exists(META_PATH):
        old = pd.read_csv(META_PATH)
        newdf = pd.concat([old, newdf], ignore_index=True)
        newdf = newdf.drop_duplicates(subset=["kic_id"])

    newdf.to_csv(META_PATH, index=False)
    print("✓ metadata update")


def process_single_kic(task_args):
    (idx, row, kic_column, oversampling, period_min, period_max) = task_args

    import numpy as _np
    import multiprocessing
    import lightkurve as lk
    from transitleastsquares import transitleastsquares as tls
    from src.utils.pipeline_utils import extract_and_stack_transits

    kic_id = int(row[kic_column])
    koi_name = f"KIC{kic_id}"

    gpath = os.path.join(OUTPUT_DIR, f"global_view_{koi_name}.npy")
    lpath = os.path.join(OUTPUT_DIR, f"local_view_{koi_name}.npy")

    if os.path.exists(gpath) and os.path.exists(lpath):
        # print(f"[PID {os.getpid()}] KIC {kic_id}: skip (exists)")
        return None


    hint = _np.nan
    possible_cols = ["period_days", "koi_period", "period", "orbital_period"]
    for col in possible_cols:
        if col in row and _np.isfinite(float(row[col])):
            hint = float(row[col])
            break

    print(f"[PID {os.getpid()}] KIC {kic_id}: processing... (Hint Period: {hint if _np.isfinite(hint) else 'BLIND SEARCH'})")

    cached = download_lightcurve_cached_protected(kic_id)
    if cached is None:
        print(f"[PID {os.getpid()}] KIC {kic_id}: no data")
        return None

    time_arr = _np.asarray(cached["time"], float)
    flux_arr = _np.asarray(cached["flux"], float)

 
    nan_count = _np.count_nonzero(_np.isnan(flux_arr))
    if nan_count == len(flux_arr):
        return None
        
    valid_count = len(flux_arr) - nan_count
    if valid_count < 1500:
        print(f"[PID {os.getpid()}] KIC {kic_id}: SKIP (points={valid_count})")
        return None

    
    if not _np.all(_np.diff(time_arr) > 0):
        idx_sort = _np.argsort(time_arr)
        time_arr = time_arr[idx_sort]
        flux_arr = flux_arr[idx_sort]


    baseline_days = time_arr[-1] - time_arr[0]
    if baseline_days <= 0:
        return None
    
  
    gaps = _np.diff(time_arr)
    max_gap = _np.nanmax(gaps)
    if max_gap > (baseline_days * 0.4):
        print(f"[PID {os.getpid()}] KIC {kic_id}: SKIP (Massive Gap > 40% of baseline: {max_gap:.1f}d)")
        return None


    theoretical_points = baseline_days / BIN_TIME
    duty_cycle = valid_count / theoretical_points
    if duty_cycle < 0.15:
        print(f"[PID {os.getpid()}] KIC {kic_id}: SKIP (Duty Cycle={duty_cycle:.2f})")
        return None

    
    if _np.isfinite(hint) and hint > 0:
        expected_transits = baseline_days / hint
        if expected_transits < 3:
            print(f"[PID {os.getpid()}] KIC {kic_id}: SKIP (expected transits={expected_transits:.1f})")
            return None
            
    
        phases = (time_arr % hint) / hint
        hist, _ = _np.histogram(phases, bins=10)
        empty_bins = _np.count_nonzero(hist == 0)
        if empty_bins >= 4:
            print(f"[PID {os.getpid()}] KIC {kic_id}: SKIP (Bad Phase Coverage for P={hint:.2f}d)")
            return None


    try:
       
        if _np.isfinite(hint) and hint > 0:
  
            if baseline_days / hint < 3:
                print(f"[PID {os.getpid()}] KIC {kic_id}: SKIP")
                return None
     
        if _np.isfinite(hint) and baseline_days / hint > 5000:
            print(f"[PID {os.getpid()}] KIC {kic_id}: SKIP")
            return None


        model = tls(time_arr, flux_arr)
    except Exception as e:
        print(f"[PID {os.getpid()}] TLS init error: {e}")
        return None

    try:
     
        if _np.isfinite(hint) and hint > 0:
            window = max(0.1, 0.5 * hint)
            p_min = max(0.1, hint - window) 
            p_max = hint + window          
        else:
            p_min = period_min
            p_max = min(period_max, PERIOD_MAX_DEFAULT)

        
        if p_min >= p_max:
             print(f"[PID {os.getpid()}] KIC {kic_id}: SKIP (Range Error p_min={p_min:.2f} > p_max={p_max:.2f})")
             return None

        n_cores = 8

   
        res = model.power(
            period_min=p_min,
            period_max=p_max,
            oversampling_factor=1,
            use_threads=n_cores,
            transit_depth_min=10e-6  
        )
    
        if hasattr(res, "transit_count") and hasattr(res, "transit_count_with_data"):
            if res.transit_count > 0:
                frac = res.transit_count_with_data / res.transit_count
    
                if frac < 0.2: 
                    print(f"[PID {os.getpid()}] KIC {kic_id}: SKIP (Bad Coverage: {frac:.2%})")
                    return None
            else:
                 return None
        if not hasattr(res, "SDE") or np.isnan(res.SDE) or res.SDE < 6.0:
            print(
                f"[PID {os.getpid()}] KIC {kic_id}: "
                f"SKIP (SDE baixo = {getattr(res, 'SDE', float('nan')):.2f})"
               )
            return None
    except MemoryError:
        print(f"[PID {os.getpid()}] KIC {kic_id}: SKIP (MemoryError - RAM insuficiente)")
        import gc
        gc.collect() 
        return None
    except Exception as e:
        print(f"[PID {os.getpid()}] TLS Power Error: {e}")
        return None


    if _np.isnan(res.period):
        return None

   
    try:
        lc = lk.LightCurve(time=time_arr, flux=flux_arr)
        folded = lc.fold(period=res.period, epoch_time=res.T0)
        gflux = _np.asarray(folded.flux.value, float)
        m, s = _np.nanmean(gflux), _np.nanstd(gflux)
        gflux = (gflux - m) / s if s > 0 else gflux - m

        dur = float(res.duration) if hasattr(res, "duration") else res.period / 20
        grid, local = extract_and_stack_transits(
            lc, res.period, res.T0, dur,
            half_window_factor=2.0, n_samples=N_SAMPLES
        )

        if local is None:
            return None

        if _np.isnan(local).mean() > 0.70:
            print(f"[PID {os.getpid()}] KIC {kic_id}: SKIP (Local view too many NaNs)")
            return None

        atomic_save_npy(gpath, gflux)
        atomic_save_npy(lpath, local)

        return {
            "koi_name": koi_name,
            "kepler_name": row.get("kepler_name", ""),
            "kic_id": kic_id,
            "period_days": res.period,
            "SDE": getattr(res, "SDE", _np.nan),
            "odd_even_mismatch": getattr(res, "odd_even_mismatch", _np.nan),
            "duration_days": dur,
            "global_npy": gpath,
            "local_npy": lpath,
            }

    except:
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
        print("CSV loaded, rows =", len(df))

        kic_column = next((c for c in df.columns if "kep" in c.lower()), None)
        if not kic_column:
            raise Exception("Missing KEPID column")

        df = df.drop_duplicates(subset=[kic_column]).reset_index(drop=True)

        CHECK = os.path.join(OUTPUT_DIR, "checkpoint.txt")
        start = int(open(CHECK).read().strip() or 0) if os.path.exists(CHECK) else 0

        print(f"Starting at row {start}")

        valid_count = 0
        results = []
        i = start

        # while valid_count < TARGET_VALID and i < len(df) and i < MAX_KIC:
        while i < len(df) and i < MAX_KIC:
            row = df.iloc[i]
            kic_id = int(row[kic_column])

            if already_processed(kic_id):
                # print(f"KIC {kic_id}: skip")
                i += 1
                continue

            out = process_single_kic(
                (i, row, kic_column,
                 OVERSAMPLING, PERIOD_MIN_DEFAULT, PERIOD_MAX_DEFAULT)
            )

            if out is not None:
                results.append(out)
                valid_count += 1
                print(f" → Valid KOI #{valid_count}/{TARGET_VALID}")

               
                if valid_count % 2 == 0:
                    save_partial_metadata(results)
                    results.clear()
                with open(CHECK, "w") as f:
                    f.write(str(i))

                print(f"Processed {valid_count} valid KOIs. Checkpoint moved to {i}.")

            i += 1

       
        if results:
            save_partial_metadata(results)
            results.clear()

        # with open(CHECK, "w") as f:
        #     f.write(str(i))

        # print(f"Processed {valid_count} valid KOIs. Checkpoint moved to {i}.")

    except:
        print("Fatal error in main:")
        traceback.print_exc()
        raise 