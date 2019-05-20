import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, classification_report

if __name__ == "__main__":

    data = pd.read_csv('./heart.csv')

    # Dữ liệu được trình bày như cc
    data = shuffle(data, random_state=0)

    train_set = data.iloc[:280]
    X_train, Y_train = np.split(train_set, [13], axis=1)

    test_set = data.iloc[280:]
    X_test, Y_real = np.split(test_set, [13], axis=1)
    
    clf = GaussianNB()
    clf.fit(X_train, np.ravel(Y_train))

    Y_predict = clf.predict(X_test)
    print(Y_predict)
    print(np.ravel(Y_real))
    print("accuracy: ", accuracy_score(np.ravel(Y_real), Y_predict))
    print(classification_report(np.ravel(Y_real), Y_predict))
