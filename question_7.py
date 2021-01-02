import question_5
import question_4


def predict_class_labels():
    #  Predict new data using classifier
    X_train, X_test, y_train, y_test = question_4.create_train_test_split()
    knn = question_5.create_classifier()
    return knn.predict(X_test)
