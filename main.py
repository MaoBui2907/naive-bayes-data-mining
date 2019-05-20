import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    data = pd.read_csv('./heart.csv')
    print(data.head(20))
    