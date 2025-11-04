
if __name__ == "__main__":
    
    from data_io import load_csv
    path = "data/raw/koi_template.csv" 

    df_teste = load_csv(path)
    
    if df_teste is not None:
        print("\nall right")

        print(df_teste.head())
    else:
        print("\nerror")