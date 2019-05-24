import pandas as pd
from sklearn.utils import shuffle
import numpy as np

if __name__ == "__main__":
    data = pd.read_csv('./heart.csv')

    # Dữ liệu đã được sắp xếp theo kiểu khó chịu nên cần shuffle
    data = shuffle(data, random_state=0)

    # * chia dữ liệu
    train_set = data.iloc[:280]
    X_train, Y_train = np.split(train_set, [13], axis=1)

    test_set = data.iloc[280:]
    X_test, Y_real = np.split(test_set, [13], axis=1)

    print(data.head())
