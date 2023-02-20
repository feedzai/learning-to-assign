import numpy as np
import pandas as pd


def set_display_configs(width=450, max_columns=25):
    pd.set_option('display.width', width)
    np.set_printoptions(linewidth=width)
    pd.set_option('display.max_columns', max_columns)
