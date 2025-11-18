# src/pipeline/tls_dataset.py
import sys
import os
import multiprocessing
import warnings
import tempfile
import numpy as np
import shutil
import time
import traceback

sys.path.append(os.path.abspath("."))

CACHE_DIR = "./cache/lightcurves/"
OUTPUT_DIR = "./data/processed/"
TEMPLATE_CSV = "./data/raw/koi_template.csv"
META_PATH = os.path.join(OUTPUT_DIR, "metadata.csv")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

warnings.filterwarnings("ignore", message="File may have been truncated", module="astropy")

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
        for _ in range(5):
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                    break
                except PermissionError:
                    time.sleep(0.1)
                except Exception:
                    break

def clear_lightkurve_cache_for_kic(kic_id):
    base = os.path.expanduser("~/.lightkurve/cache/mastDownload/Kepler/")
    if not os.path.exists(base):
        return
    for root, dirs, _ in os.walk(base):
        for d in dirs:
            if str(kic_id) in d:
                try:
                    shutil.rmtree(os.path.join(root, d), ignore_errors=True)
                except Exception:
                    pass

def download_lightcurve_cached_protected(kic_id):
    import lightkurve as lk
    npy_cache = os.path.join(CACHE_DIR, f"KIC_{kic_id}.npy")

    if os.path.exists(npy_cache):
        try:
            return np.load(npy_cache, allow_pickle=True).item()
        except Exception:
            try:
                os.remove(npy_cache)
            except Exception:
                pass

    for attempt in range(2):
        try:
            search = lk.search_lightcurve(f"KIC {kic_id}", author="Kepler")
            if len(search) == 0:
                return None

            lc = search.download_all().stitch().remove_nans().remove_outliers()
            lc = lc.bin(time_bin_size=0.02)

            payload = {"time": lc.time.value, "flux": lc.flux.value}
            atomic_save_npy(npy_cache, payload)
            return payload

        except Exception as e:
            print(f"[download] KIC {kic_id}: corrupted FITS -> retry")
            clear_lightkurve_cache_for_kic(kic_id)
            time.sleep(0.2)

    print(f"[download] KIC {kic_id}: failed")
    return None

def already_processed(kic_id):
    g = os.path.join(OUTPUT_DIR, f"global_view_KIC{kic_id}.npy")
    l = os.path.join(OUTPUT_DIR, f"local_view_KIC{kic_id}.npy")
    if os.path.exists(g) and os.path.exists(l):
        return True

    if os.path.exists(META_PATH):
        try:
            import pandas as pd
            meta = pd.read_csv(META_PATH)
            if "kic_id" in meta.columns and kic_id in meta["kic_id"].values:
                return True
        except Exception:
            pass
    return False

def process_single_kic(task_args):
    (idx, row, kic_column, oversampling, period_min, period_max) = task_args
    import numpy as _np
    import lightkurve as lk
    from transitleastsquares import transitleastsquares as tls
    from src.utils.pipeline_utils import extract_and_stack_transits

    kic_id = int(row[kic_column])
    koi_name = f"KIC{kic_id}"

    gpath = os.path.join(OUTPUT_DIR, f"global_view_{koi_name}.npy")
    lpath = os.path.join(OUTPUT_DIR, f"local_view_{koi_name}.npy")
    if os.path.exists(gpath) and os.path.exists(lpath):
        print(f"[PID {os.getpid()}] KIC {kic_id}: skip (exists)")
        return None

    print(f"[PID {os.getpid()}] KIC {kic_id}: processing...")

    cached = download_lightcurve_cached_protected(kic_id)
    if cached is None:
        print(f"[PID {os.getpid()}] KIC {kic_id}: no data")
        return None

    time_arr = _np.asarray(cached["time"], float)
    flux_arr = _np.asarray(cached["flux"], float)

    try:
        model = tls(time_arr, flux_arr)
    except Exception as e:
        print(f"[PID {os.getpid()}] KIC {kic_id}: TLS error")
        return None

    hint = row.get("period_days", _np.nan)
    try:
        if _np.isfinite(hint):
            window = 0.25 * hint
            res = model.power(
                period_min=max(0.5, hint - window),
                period_max=hint + window,
                oversampling_factor=oversampling,
                n_threads=1,
            )
        else:
            res = model.power(
                period_min=period_min,
                period_max=period_max,
                oversampling_factor=oversampling,
                n_threads=1,
            )
    except Exception:
        print(f"[PID {os.getpid()}] KIC {kic_id}: TLS power error")
        return None

    if _np.isnan(res.period):
        print(f"[PID {os.getpid()}] KIC {kic_id}: invalid period")
        return None

    try:
        lc = lk.LightCurve(time=time_arr, flux=flux_arr)
        folded = lc.fold(period=res.period, epoch_time=res.T0)
        gflux = _np.asarray(folded.flux.value, float)
        m, s = _np.nanmean(gflux), _np.nanstd(gflux)
        gflux = (gflux - m) / s if s > 0 else gflux - m
    except Exception:
        print(f"[PID {os.getpid()}] KIC {kic_id}: fold error")
        return None

    dur = float(res.duration) if hasattr(res, "duration") else res.period / 20.0
    grid, local = extract_and_stack_transits(lc, res.period, res.T0, dur, half_window_factor=2.0, n_samples=2001)

    if local is None or _np.all(_np.isnan(local)) or _np.nanstd(local) < 0.01:
        print(f"[PID {os.getpid()}] KIC {kic_id}: invalid transit")
        return None

    try:
        atomic_save_npy(gpath, gflux)
        atomic_save_npy(lpath, local)
    except Exception:
        print(f"[PID {os.getpid()}] KIC {kic_id}: save error")
        try:
            if os.path.exists(gpath) and os.path.getsize(gpath) == 0:
                os.remove(gpath)
        except Exception:
            pass
        try:
            if os.path.exists(lpath) and os.path.getsize(lpath) == 0:
                os.remove(lpath)
        except Exception:
            pass
        return None

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

if __name__ == "__main__":
    try:
        import pandas as pd
        from src.utils.data_io import load_csv
    except Exception as e:
        print("Import error")
        traceback.print_exc()
        raise

    try:
        print("Starting tls_dataset.py")
        print("Checking files...")

        if not os.path.exists(TEMPLATE_CSV):
            print("Catalog file missing")
            raise FileNotFoundError(TEMPLATE_CSV)

        df = load_csv(TEMPLATE_CSV)
        print("CSV loaded, rows =", len(df))

        kic_column = next((c for c in df.columns if "kep" in c.lower()), None)
        if kic_column is None:
            raise Exception("Missing KEPID column")

        df = df.drop_duplicates(subset=[kic_column]).reset_index(drop=True)

        BATCH_SIZE = 4
        MAX_KIC = 400

        CHECK = os.path.join(OUTPUT_DIR, "checkpoint.txt")
        if os.path.exists(CHECK):
            start = int(open(CHECK).read().strip() or 0)
        else:
            start = 0

        end = min(start + BATCH_SIZE, len(df), MAX_KIC)

        num_workers = 3

        print(f"Processing KICs {start+1} to {end}")
        print(f"Workers: {num_workers}")

        tasks = []
        for i in range(start, end):
            kic_id_check = int(df.iloc[i][kic_column])
            if already_processed(kic_id_check):
                print(f"KIC {kic_id_check}: skip")
                continue
            tasks.append((i, df.iloc[i], kic_column, 1, 0.5, 20.0))

        if not tasks:
            print("Nothing to process")
        else:
            results = []
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=num_workers) as ex:
                for out in ex.map(process_single_kic, tasks):
                    if out is not None:
                        results.append(out)

            if results:
                newdf = pd.DataFrame(results)
                if os.path.exists(META_PATH):
                    old = pd.read_csv(META_PATH)
                    newdf = pd.concat([old, newdf], ignore_index=True)
                    newdf = newdf.drop_duplicates(subset=["kic_id"])
                newdf.to_csv(META_PATH, index=False)
                print("Metadata updated")

            with open(CHECK, "w") as f:
                f.write(str(end))

            print(f"Block {end}/{min(MAX_KIC, len(df))} done")

    except Exception:
        print("Fatal error in main")
        traceback.print_exc()
        raise
