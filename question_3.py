import pandas as pd

import question_1


def split_to_xy():
    df = question_1.create_data_frame()
    X = df.drop('target', axis=1)
    y = df.get('target')
    return X, y
