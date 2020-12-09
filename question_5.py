from sklearn.neighbors import KNeighborsClassifier
import question_4


def create_classifier():
    X_train, X_test, y_train, y_test = question_4.create_train_test_split()
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    knn.score(X_test, y_test)
    return knn
