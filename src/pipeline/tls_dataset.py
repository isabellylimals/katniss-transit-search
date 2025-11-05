"""
COMPLETE PIPELINE TO GENERATE PLANETARY TRANSIT DATASET
 - Lightkurve (download + cleaning)
 - TransitLeastSquares (detection)
 - GLOBAL / LOCAL VIEW
 - Normalization
 - Save as .npy (for neural network)
"""

import multiprocessing
import sys, os
sys.path.append(os.path.abspath("."))


if __name__ == "__main__":
    import os
    import numpy as np
    import lightkurve as lk
    import pandas as pd
    from transitleastsquares import transitleastsquares as tls
    from src.utils.data_io import load_csv

    import matplotlib.pyplot as plt

    TEMPLATE_CSV = "./data/raw/koi_template.csv"
    OUTPUT_DIR = "./data/processed/"
    period_min = 0.5     # mínimo de período testado pelo TLS (em dias)
    period_max = 20.0  
    import multiprocessing
    threads = 8   # força 8 threads mesmo que o Colab diga menos
  # usa TODOS os núcleos do Colab
    
    


    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Reading catalog file (KOI/NASA)...")
    df = load_csv(TEMPLATE_CSV)

    kic_column = next((c for c in df.columns if "kep" in c.lower()), None)
    if kic_column is None:
        raise Exception("ERROR: Column with Kepler ID (kepid) not found in CSV.")

    print("\nStarting dataset generation for AI...\n")


    BATCH_SIZE = 1     # processar 10 KOIs por execução
    MAX_KOIS = 2   # processar apenas 60 KOIs no total
    CHECKPOINT_FILE = "./data/processed/checkpoint.txt"

    # Ler quantos já foram processados
   # máximo (reduz tempo de execução)

    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            start_index = int(f.read().strip())
    else:
        start_index = 0

    end_index = min(start_index + BATCH_SIZE, MAX_KOIS)

    print(f"Processing KOIs {start_index+1} to {end_index} of {MAX_KOIS}")

    # LOOP DOS KOIs (com checkpoint e blocos)
    for index in range(start_index, end_index):
        row = df.iloc[index]
        kic_id = int(row[kic_column])

        try:
            search = lk.search_lightcurve(f"KIC {kic_id}", author="Kepler")
            lc = search.download_all().stitch().remove_nans().remove_outliers()
            lc = lc.bin(time_bin_size=0.02)

            print("Data loaded and cleaned!")

            model = tls(lc.time.value, lc.flux.value)
            results = model.power(period_min=period_min,
                                  period_max=period_max,
                                  oversampling_factor=2,
                                  n_threads=threads)

            print(f"Period: {results.period:.5f} days   |   SDE: {results.SDE:.2f}")

            lc_folded = lc.fold(period=results.period, epoch_time=results.T0)

            # NORMALIZAÇÃO (GLOBAL VIEW)
            flux_global = (lc_folded.flux.value - np.mean(lc_folded.flux.value)) / np.std(lc_folded.flux.value)
            np.save(f"{OUTPUT_DIR}/global_view_{kic_id}.npy", flux_global)

            # GLOBAL VIEW PNG
            plt.figure(figsize=(10, 4))
            plt.scatter(lc_folded.time.value, flux_global, s=3, color="black")
            plt.xlabel("Phase (days)")
            plt.ylabel("Flux (normalized)")
            plt.title(f"Global View - KIC {kic_id}")
            plt.savefig(f"{OUTPUT_DIR}/global_view_{kic_id}.png", dpi=300)
            plt.close()

            # LOCAL VIEW NORMALIZACAO (zoom no trânsito)
            mask = (lc_folded.time.value > -results.duration * 2) & (lc_folded.time.value < results.duration * 2)
            flux_local = (lc_folded.flux.value[mask] - np.mean(lc_folded.flux.value[mask])) / np.std(lc_folded.flux.value[mask])
            np.save(f"{OUTPUT_DIR}/local_view_{kic_id}.npy", flux_local)

            plt.figure(figsize=(10, 4))
            plt.scatter(lc_folded.time.value[mask], flux_local, s=5, color="red")
            plt.xlabel("Phase (days)")
            plt.ylabel("Flux (normalized)")
            plt.title(f"Local View - KIC {kic_id}")
            plt.savefig(f"{OUTPUT_DIR}/local_view_{kic_id}.png", dpi=300)
            plt.close()

            print(f"Saved: GLOBAL + LOCAL VIEW for {kic_id}")

        except Exception as e:
            print(f"ERROR with KIC {kic_id}: {e}")
            continue

 
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(str(end_index))

    print(f"Block finished. {end_index}/{MAX_KOIS} processed.")

    if end_index >= MAX_KOIS:
        print("\nFULL DATASET SUCCESSFULLY GENERATED!")
    else:
        print("Run the script again for the next block.")
