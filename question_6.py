import question_1
import question_5


def predict_test_data():
    df = question_1.create_data_frame()
    means = df.mean()[:-1].values.reshape(1, -1)
    knn = question_5.create_classifier()
    return knn.predict(means)