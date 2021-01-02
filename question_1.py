import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer


def create_data_frame():
    #  Create database using sci kit learn cancer database
    cancer = load_breast_cancer()
    df_data = np.column_stack((cancer.data, cancer.target))
    df_columns = np.append(cancer.feature_names, 'target')
    df_index = pd.RangeIndex(start=0, stop=569, step=1)
    df = pd.DataFrame(data=df_data, index=df_index, columns=df_columns)
    return df

