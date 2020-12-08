import pandas as pd
import question_1


def get_target_count():
    df = question_1.create_data_frame()
    #  benign = 1, malignant = 0
    se_index = ['benign', 'malignant']
    target_count = pd.Series(df['target'].value_counts().values, index=se_index)
    return target_count
