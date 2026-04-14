# import os
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split

# print("pré-processamento")


# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

# meta_path = os.path.join(PROJECT_ROOT, "data", "processed", "metadata.csv")
# template_path = os.path.join(PROJECT_ROOT, "data", "raw", "koi_template.csv")

# meta_path = os.path.normpath(meta_path)
# template_path = os.path.normpath(template_path)

# print(f"Buscando metadados em: {meta_path}")
# print(f"Buscando template em: {template_path}")

# if not os.path.exists(meta_path) or not os.path.exists(template_path):
#     print(" ERRO: Ficheiros não encontrados!")
#     exit()


# meta_df = pd.read_csv(meta_path)
# print(f"\n1. Metadados carregados: {len(meta_df)} extrações no total.")

# raw_df = pd.read_csv(template_path, comment='#', low_memory=False)


# kic_column = next((c for c in raw_df.columns if "kepid" in c.lower() or "kic" in c.lower()), raw_df.columns[0])
# raw_df = raw_df.rename(columns={kic_column: 'kic_id'})

# if 'kepoi_name' in raw_df.columns:
#     raw_df = raw_df.rename(columns={'kepoi_name': 'koi_name_real'})

# dict_koi = dict(zip(raw_df['koi_name_real'], raw_df['koi_disposition']))

# raw_first = raw_df.drop_duplicates(subset=['kic_id'])
# dict_kic = dict(zip(raw_first['kic_id'], raw_first['koi_disposition']))

# def descobrir_label(row):
#     nome = str(row.get('koi_name', ''))
#     if nome.startswith('K0'):
#         return dict_koi.get(nome, np.nan)
#     else:
#         return dict_kic.get(row['kic_id'], np.nan)

# meta_df['koi_disposition'] = meta_df.apply(descobrir_label, axis=1)

# df = meta_df.dropna(subset=['koi_disposition']).copy()

# candidatos_df = df[df['koi_disposition'] == 'CANDIDATE'].copy()

# df = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()

# df['label'] = df['koi_disposition'].apply(lambda x: 1 if x == 'CONFIRMED' else 0)


# df = df.rename(columns={
#     'global_npy': 'global_path',
#     'local_npy': 'local_path',
#     'SDE': 'sde',
#     'period_days': 'period'
# })

# candidatos_df = candidatos_df.rename(columns={
#     'global_npy': 'global_path',
#     'local_npy': 'local_path',
#     'SDE': 'sde',
#     'period_days': 'period'
# })


# print(f"Distribuição:\n{df['label'].value_counts()}")


# df_proc = df.copy() 

# print("\nDividindo em treino/teste")

# train_df, test_df = train_test_split(
#     df_proc,
#     test_size=0.2,
#     random_state=42,
#     stratify=df_proc['label']
# )

# for col in ['sde', 'period', 'odd_even_mismatch']:
#     if col in train_df.columns:
#         mediana = train_df[col].median()
#         train_df.loc[:, col] = train_df[col].fillna(mediana)
#         test_df.loc[:, col] = test_df[col].fillna(mediana)

# print("\nDistribuição treino:")
# print(train_df['label'].value_counts(normalize=True))

# print("\nDistribuição teste:")
# print(test_df['label'].value_counts(normalize=True))

# print("\nGuardando ficheiros...")

# train_dir = os.path.join(PROJECT_ROOT, "data", "processed", "train")
# test_dir = os.path.join(PROJECT_ROOT, "data", "processed", "test")
# disc_dir = os.path.join(PROJECT_ROOT, "data", "processed", "discovery")

# os.makedirs(train_dir, exist_ok=True)
# os.makedirs(test_dir, exist_ok=True)
# os.makedirs(disc_dir, exist_ok=True)

# train_df.to_csv(os.path.join(train_dir, "train_split.csv"), index=False)
# test_df.to_csv(os.path.join(test_dir, "test_split.csv"), index=False)
# candidatos_df.to_csv(os.path.join(disc_dir, "candidatos_para_prever.csv"), index=False)

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  

print("pré-processamento")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

meta_path = os.path.join(PROJECT_ROOT, "data", "processed", "metadata.csv")
template_path = os.path.join(PROJECT_ROOT, "data", "raw", "koi_template.csv")

meta_path = os.path.normpath(meta_path)
template_path = os.path.normpath(template_path)

print(f"Buscando metadados em: {meta_path}")
print(f"Buscando template em: {template_path}")

if not os.path.exists(meta_path) or not os.path.exists(template_path):
    print(" ERRO: Ficheiros não encontrados!")
    exit()


meta_df = pd.read_csv(meta_path)
print(f"\n1. Metadados carregados: {len(meta_df)} extrações no total.")

raw_df = pd.read_csv(template_path, comment='#', low_memory=False)


kic_column = next((c for c in raw_df.columns if "kepid" in c.lower() or "kic" in c.lower()), raw_df.columns[0])
raw_df = raw_df.rename(columns={kic_column: 'kic_id'})

if 'kepoi_name' in raw_df.columns:
    raw_df = raw_df.rename(columns={'kepoi_name': 'koi_name_real'})


raw_first = raw_df.drop_duplicates(subset=['kic_id'])


dict_koi_label = dict(zip(raw_df['koi_name_real'], raw_df['koi_disposition']))
dict_kic_label = dict(zip(raw_first['kic_id'], raw_first['koi_disposition']))


dict_koi_snr = dict(zip(raw_df['koi_name_real'], raw_df['koi_model_snr']))
dict_kic_snr = dict(zip(raw_first['kic_id'], raw_first['koi_model_snr']))


dict_koi_impact = dict(zip(raw_df['koi_name_real'], raw_df['koi_impact']))
dict_kic_impact = dict(zip(raw_first['kic_id'], raw_first['koi_impact']))


dict_koi_depth = dict(zip(raw_df['koi_name_real'], raw_df['koi_depth']))
dict_kic_depth = dict(zip(raw_first['kic_id'], raw_first['koi_depth']))


dict_koi_prad = dict(zip(raw_df['koi_name_real'], raw_df['koi_prad']))
dict_kic_prad = dict(zip(raw_first['kic_id'], raw_first['koi_prad']))

def descobrir_valores(row, dict_koi, dict_kic):
    nome = str(row.get('koi_name', ''))
    if nome.startswith('K0'):
        return dict_koi.get(nome, np.nan)
    else:
        return dict_kic.get(row['kic_id'], np.nan)


meta_df['koi_disposition'] = meta_df.apply(lambda r: descobrir_valores(r, dict_koi_label, dict_kic_label), axis=1)
meta_df['koi_model_snr'] = meta_df.apply(lambda r: descobrir_valores(r, dict_koi_snr, dict_kic_snr), axis=1)
meta_df['koi_impact'] = meta_df.apply(lambda r: descobrir_valores(r, dict_koi_impact, dict_kic_impact), axis=1)


meta_df['koi_depth'] = meta_df.apply(lambda r: descobrir_valores(r, dict_koi_depth, dict_kic_depth), axis=1)
meta_df['koi_prad'] = meta_df.apply(lambda r: descobrir_valores(r, dict_koi_prad, dict_kic_prad), axis=1)

df = meta_df.dropna(subset=['koi_disposition']).copy()


candidatos_df = df[df['koi_disposition'] == 'CANDIDATE'].copy()

df = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()

df['label'] = df['koi_disposition'].apply(lambda x: 1 if x == 'CONFIRMED' else 0)


df = df.rename(columns={
    'global_npy': 'global_path',
    'local_npy': 'local_path',
    'SDE': 'sde',
    'period_days': 'period'
})

candidatos_df = candidatos_df.rename(columns={
    'global_npy': 'global_path',
    'local_npy': 'local_path',
    'SDE': 'sde',
    'period_days': 'period'
})

print(f"\nGolden Dataset criado! Total: {len(df)}")
print(f"Distribuição:\n{df['label'].value_counts()}")


df_proc = df.copy()

print("\nDividindo em treino/teste...")

train_df, test_df = train_test_split(
    df_proc,
    test_size=0.2,
    random_state=42,
    stratify=df_proc['label']
)

colunas_para_preencher = ['sde', 'period', 'odd_even_mismatch', 'koi_model_snr', 'koi_impact', 'koi_depth', 'koi_prad']


for col in colunas_para_preencher:
    if col in train_df.columns:
        mediana = train_df[col].median()
        train_df.loc[:, col] = train_df[col].fillna(mediana)
        test_df.loc[:, col] = test_df[col].fillna(mediana)
        candidatos_df.loc[:, col] = candidatos_df[col].fillna(mediana)

print("\nNormalizando Features da NASA...")
scaler = StandardScaler()


cols_to_scale = ['koi_model_snr', 'koi_impact', 'koi_depth', 'koi_prad']


train_df.loc[:, cols_to_scale] = scaler.fit_transform(train_df[cols_to_scale])
test_df.loc[:, cols_to_scale] = scaler.transform(test_df[cols_to_scale])
candidatos_df.loc[:, cols_to_scale] = scaler.transform(candidatos_df[cols_to_scale])

print("\nDistribuição treino:")
print(train_df['label'].value_counts(normalize=True))

print("\nDistribuição teste:")
print(test_df['label'].value_counts(normalize=True))

print("\nGuardando ficheiros...")

train_dir = os.path.join(PROJECT_ROOT, "data", "processed", "train")
test_dir = os.path.join(PROJECT_ROOT, "data", "processed", "test")
disc_dir = os.path.join(PROJECT_ROOT, "data", "processed", "discovery")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(disc_dir, exist_ok=True)

train_df.to_csv(os.path.join(train_dir, "train_split.csv"), index=False)
test_df.to_csv(os.path.join(test_dir, "test_split.csv"), index=False)
candidatos_df.to_csv(os.path.join(disc_dir, "candidatos_para_prever.csv"), index=False)

print("\nDataset final gerado com sucesso!")