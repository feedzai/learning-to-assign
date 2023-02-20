import pandas as pd
import yaml

cfg_path = './cfg.yaml'
with open(cfg_path, 'r') as infile:
    cfg = yaml.safe_load(infile)

# DATA LOADING
data = pd.read_parquet(cfg['input_data_path'])

LABEL_COL = cfg['data_cols']['label']
TIMESTAMP_COL = cfg['data_cols']['timestamp']
PROTECTED_COL = cfg['data_cols']['protected']
CATEGORICAL_COLS = cfg['data_cols']['categorical']
data[CATEGORICAL_COLS] = data[CATEGORICAL_COLS].astype('category')  # LightGBM now processes these as categorical

data = data.sample(frac=1, random_state=42)  # currently is not shuffled
data = data.sort_values(by=TIMESTAMP_COL)
data = data.reset_index(drop=True)

data.to_parquet(cfg['data_path'])

