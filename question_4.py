import question_3
from sklearn.model_selection import train_test_split


def create_train_test_split():
    X, y = question_3.split_to_xy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return X_train, X_test, y_train, y_test
