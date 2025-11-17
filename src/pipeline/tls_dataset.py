import sys, os, multiprocessing, warnings
sys.path.append(os.path.abspath("."))

CACHE_DIR = "./cache/lightcurves/"
os.makedirs(CACHE_DIR, exist_ok=True)

import numba
import numpy as np

@numba.njit(fastmath=True)
def normalize_array_numba(x):
    m = np.nanmean(x)
    s = np.nanstd(x)
    if s > 0:
        return (x - m) / s
    return x - m

def download_lightcurve_cached(kic_id):
    import lightkurve as lk
    import numpy as np

    fname = os.path.join(CACHE_DIR, f"KIC_{kic_id}.npy")

    if os.path.exists(fname):
        try:
            return np.load(fname, allow_pickle=True).item()
        except:
            pass  

    search = lk.search_lightcurve(f"KIC {kic_id}", author="Kepler")
    if len(search) == 0:
        return None

    lc = search.download_all().stitch().remove_nans().remove_outliers()
    lc = lc.bin(time_bin_size=0.02)

    np.save(fname, {"time": lc.time.value, "flux": lc.flux.value})
    return {"time": lc.time.value, "flux": lc.flux.value}

def process_single_kic(args):
    (
        idx,
        row,
        kic_column,
        oversampling,
        OUTPUT_DIR,
        META_PATH,
        period_min,
        period_max
    ) = args

    import pandas as pd
    from transitleastsquares import transitleastsquares as tls
    from src.utils.pipeline_utils import extract_and_stack_transits

    kic_id = int(row[kic_column])
    koi_name = str(row.get("kepoi_name", kic_id))

    print(f"[PID {os.getpid()}] Processing KIC {kic_id}...")

    cached = download_lightcurve_cached(kic_id)
    if cached is None:
        print(f"KIC {kic_id}: no data.")
        return None

    time = np.asarray(cached["time"], dtype=float)
    flux = np.asarray(cached["flux"], dtype=float)
    model = tls(time, flux)

    period_hint = row.get("period_days", np.nan)
    if np.isfinite(period_hint):
        period_window = 0.25 * period_hint
        results = model.power(
            period_min=max(0.5, period_hint - period_window),
            period_max=period_hint + period_window,
            oversampling_factor=oversampling,
            n_threads=2
        )
    else:
        results = model.power(
            period_min=period_min,
            period_max=period_max,
            oversampling_factor=oversampling,
            n_threads=2,
        )

    if np.isnan(results.period):
        print(f"KIC {kic_id}: invalid period.")
        return None

    import lightkurve as lk

    lc = lk.LightCurve(time=time, flux=flux)
    lc_fold = lc.fold(period=results.period, epoch_time=results.T0)

    phase_global = np.asarray(lc_fold.time.value, dtype=float)
    flux_global = normalize_array_numba(np.asarray(lc_fold.flux.value, dtype=float))
    dur = float(results.duration) if hasattr(results, "duration") else results.period / 20
    grid, stacked_local = extract_and_stack_transits(
        lc, results.period, results.T0, dur,
        half_window_factor=2.0, n_samples=2001
    )

    if (
        stacked_local is None
        or np.all(np.isnan(stacked_local))
        or stacked_local.size == 0
        or np.nanstd(stacked_local) < 0.01
    ):
        print(f"KIC {kic_id}: invalid stacked transit.")
        return None

    npy_g = os.path.join(OUTPUT_DIR, f"global_view_{koi_name}.npy")
    np.save(npy_g, flux_global)

    npy_l = os.path.join(OUTPUT_DIR, f"local_view_{koi_name}.npy")
    np.save(npy_l, stacked_local)

    df_row = {
        "koi_name": koi_name,
        "kepler_name": row.get("kepler_name", ""),
        "kic_id": kic_id,
        "period_days": results.period,
        "SDE": results.SDE,
        "odd_even_mismatch": getattr(results, "odd_even_mismatch", np.nan),
        "duration_days": dur,
        "global_npy": npy_g,
        "local_npy": npy_l,
    }

    return df_row

if __name__ == "__main__":
    import pandas as pd
    from src.utils.data_io import load_csv

    warnings.filterwarnings("ignore")

    TEMPLATE_CSV = "./data/raw/koi_template.csv"
    OUTPUT_DIR = "./data/processed/"
    META_PATH = os.path.join(OUTPUT_DIR, "metadata.csv")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_csv(TEMPLATE_CSV)

    kic_column = next((c for c in df.columns if "kep" in c.lower()), None)
    if kic_column is None:
        raise Exception("ERROR: Column with Kepler ID not found")

    df.drop_duplicates(subset=[kic_column], inplace=True)
    df.reset_index(drop=True, inplace=True)

    TOTAL_STARS = min(400, len(df))
    BATCH_SIZE = 10

    NUM_PROCESSES = max(1, multiprocessing.cpu_count() // 2)
    CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint.txt")
    if os.path.exists(CHECKPOINT_FILE):
        start_index = int(open(CHECKPOINT_FILE).read().strip() or 0)
    else:
        start_index = 0

    end_index = min(start_index + BATCH_SIZE, TOTAL_STARS)

    print(f"\nProcessing KICs {start_index+1} to {end_index} of {TOTAL_STARS}")
    print(f"Parallel processes: {NUM_PROCESSES}")

    tasks = []
    for idx in range(start_index, end_index):
        tasks.append((
            idx,
            df.iloc[idx],
            kic_column,
            2,             
            OUTPUT_DIR,
            META_PATH,
            0.5,            
            20.0            
        ))

    from concurrent.futures import ProcessPoolExecutor

    results = []
    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as ex:
        for out in ex.map(process_single_kic, tasks):
            if out is not None:
                results.append(out)

    if results:
        pd.DataFrame(results).to_csv(
            META_PATH, mode="a",
            header=not os.path.exists(META_PATH),
            index=False
        )

    with open(CHECKPOINT_FILE, "w") as f:
        f.write(str(end_index))

    print(f"\nCompleted block {end_index}/{TOTAL_STARS} processed.")
