import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle

if __name__ == "__main__":
    data = pd.read_csv('./heart.csv')

    # Dữ liệu đã được sắp xếp theo kiểu khó chịu nên cần shuffle
    data = shuffle(data, random_state=0)
    print(data.head())
    