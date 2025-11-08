import sys, os, multiprocessing, warnings
sys.path.append(os.path.abspath("."))

if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import lightkurve as lk
    from transitleastsquares import transitleastsquares as tls
    from src.utils.data_io import load_csv
    from src.utils.pipeline_utils import extract_and_stack_transits

    warnings.filterwarnings("ignore", category=UserWarning)
    # MUDANÇA 1: Adicionado para suprimir os avisos de tempo de execução (RuntimeWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    TEMPLATE_CSV    = "./data/raw/koi_template.csv"
    OUTPUT_DIR      = "./data/processed/"
    META_PATH       = os.path.join(OUTPUT_DIR, "metadata.csv")

    period_min      = 0.5
    period_max      = 20.0
    oversampling    = 2

    env_threads = os.getenv("TLS_THREADS")
    threads = int(env_threads) if env_threads and env_threads.isdigit() else min(multiprocessing.cpu_count(), 8)

    BATCH_SIZE      = 2
    MAX_KOIS_CAP    = 400 # Renomeado de MAX_KOIS para clareza
    CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint.txt")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Reading catalog file (KOI/NASA)...")
    df = load_csv(TEMPLATE_CSV)

    kic_column = next((c for c in df.columns if "kep" in c.lower()), None)
    if kic_column is None:
        raise Exception("ERROR: Column with Kepler ID not found in CSV (expected 'kepid').")

    # MUDANÇA 2: Remover KICs duplicados para processar apenas estrelas únicas
    original_count = len(df)
    df.drop_duplicates(subset=[kic_column], keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True) # Redefinir o índice é crucial para .iloc[idx] funcionar
    new_count = len(df)
    print(f"Loaded {original_count} rows, found {new_count} unique KICs.")
    
    # Agora, a lógica de processamento usará o total de KICs únicos
    TOTAL_STARS = min(MAX_KOIS_CAP, new_count)

    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            start_index = int((f.read() or "0").strip())
    else:
        start_index = 0

    end_index = min(start_index + BATCH_SIZE, TOTAL_STARS)
    print(f"\nStarting dataset generation...\n")
    print(f"Threads: {threads} | Oversampling: {oversampling}")
    # A mensagem de log agora reflete o processamento de KICs únicos
    print(f"Processing KICs {start_index+1} to {end_index} of {TOTAL_STARS}")

    def normalize_array(x: np.ndarray) -> np.ndarray:
        if hasattr(x, "filled"):
            x = x.filled(np.nan)
        x = np.asarray(x, dtype=float)
        m = np.nanmean(x)
        s = np.nanstd(x)
        return (x - m) / s if np.isfinite(s) and s > 0 else (x - m)

    for idx in range(start_index, end_index):
        row = df.iloc[idx]
        kic_id = int(row[kic_column])
        # O koi_name agora será o primeiro KOI encontrado para aquele KIC
        koi_name = str(row["kepoi_name"]) if "kepoi_name" in row else str(kic_id)

        print("\n" + "="*60)
        # A mensagem de log agora usa o total de KICs únicos
        print(f"Processing KIC {kic_id} ({idx+1}/{TOTAL_STARS})")
        print("="*60)

        try:
            search = lk.search_lightcurve(f"KIC {kic_id}", author="Kepler")
            if len(search) == 0:
                print("WARNING: No light curves found. Skipping.")
                continue

            lc = search.download_all().stitch().remove_nans().remove_outliers()
            lc = lc.bin(time_bin_size=0.02)

    
            period_hint = row.get("period_days", np.nan)
            if not np.isfinite(period_hint):
                period_hint = None

            model = tls(lc.time.value, lc.flux.value)

            if period_hint:
                print(f"→ Using KOI period hint: {period_hint:.4f} days")
                period_window = 0.25 * period_hint   # ajustável
                results = model.power(
                    period_min=max(0.5, period_hint - period_window),
                    period_max=period_hint + period_window,
                    oversampling_factor=oversampling,
                    n_threads=threads,
                )
            else:
                print("→ No period hint found, using default TLS search range")
                results = model.power(
                    period_min=period_min,
                    period_max=period_max,
                    oversampling_factor=oversampling,
                    n_threads=threads,
                )


            print(f"Period: {results.period:.5f} days | SDE: {results.SDE:.2f}")

            lc_folded = lc.fold(period=results.period, epoch_time=results.T0)
            phase_global = np.asarray(lc_folded.time.value, dtype=float)
            flux_global  = normalize_array(lc_folded.flux.value)

            dur = float(results.duration) if hasattr(results, "duration") else results.period / 20.0
            grid, stacked_local = extract_and_stack_transits(
                lc, results.period, results.T0, dur, half_window_factor=2.0, n_samples=2001
            )
            
            if (
                stacked_local is None or
                np.all(np.isnan(stacked_local)) or
                stacked_local.size == 0 or
                np.nanstd(stacked_local) < 0.01
            ):
                print(f"WARNING: Curva local inválida ou muito plana para KIC {kic_id}. Skipping.")
                continue


            if stacked_local is None or np.all(np.isnan(stacked_local)) or stacked_local.size == 0:
                print(f"WARNING: Empty or invalid stacked transit for KIC {kic_id}. Skipping.")
                continue

            npy_g = os.path.join(OUTPUT_DIR, f"global_view_{koi_name}.npy")
            np.save(npy_g, flux_global)

            npy_l = os.path.join(OUTPUT_DIR, f"local_view_{koi_name}.npy")
            np.save(npy_l, stacked_local)

            pd.DataFrame([{
                "koi_name": koi_name,
                "kepler_name": row.get("kepler_name", ""),
                "kic_id": kic_id,
                "period_days": results.period,
                "SDE": results.SDE,
                "odd_even_mismatch": getattr(results, "odd_even_mismatch", np.nan),
                "duration_days": dur,
                "global_npy": npy_g,
                "local_npy": npy_l,
            }]).to_csv(META_PATH, mode="a", header=not os.path.exists(META_PATH), index=False)

            print(f"Saved GLOBAL + LOCAL dataset for KIC {kic_id}")

        except Exception as e:
            print(f"ERROR with KIC {kic_id}: {e}")
            continue

    with open(CHECKPOINT_FILE, "w") as f:
        f.write(str(end_index))

    print(f"\nBlock finished. {end_index}/{TOTAL_STARS} processed.")
    if end_index >= TOTAL_STARS:
        print("FULL DATASET GENERATED!")
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE) # Limpa o checkpoint ao concluir
    else:
        print("Run the script again for next block.")