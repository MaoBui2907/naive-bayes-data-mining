import numpy as np
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

if __name__ == "__main__":

    # ! Chuẩn bị dữ liệu
    # * Đọc dữ liệu
    with open('vectorlist.bin', 'rb') as fb:
        word_vector = pickle.load(fb)

    # * chia dữ liệu 20000 dòng để train
    train_set = word_vector[:200]
    train_data = np.array([i[:-1] for i in train_set])
    train_label = np.array([i[-1] for i in train_set])

    # * lấy 6000 dòng còn lại để test
    test_set = word_vector[200:250]
    test_data = np.array([i[:-1] for i in test_set])
    test_label = np.array([i[-1] for i in test_set])

    # ! Học với Naive Bayes
    clf = MultinomialNB()
    clf.fit(train_data, train_label)

    test_predict = clf.predict(test_data)
    print(test_predict)
    print(test_label)
    print("accuracy: ", accuracy_score(test_label, test_predict))
    print(classification_report(test_label, test_predict))
