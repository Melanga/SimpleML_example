import question_4
import question_5
import question_6


def find_score():
    #  find the score of the classifier using training data sets
    X_train, X_test, y_train, y_test = question_4.create_train_test_split()
    knn = question_5.create_classifier()
    score = knn.score(X_test, y_test)
    return score
