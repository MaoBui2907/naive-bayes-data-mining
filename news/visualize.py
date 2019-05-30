
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

if __name__ == "__main__":
    # ! Đọc dữ liệu
    # * Đọc dữ liệu
    with open('vectorlist.bin', 'rb') as fb:
        word_vector = pickle.load(fb)

    # * chia dữ liệu 20000 dòng để train
    train_set = word_vector[:2000]
    train_data = np.array([i[:-1] for i in train_set])
    train_label = np.array([i[-1] for i in train_set])

    # * lấy 6000 dòng còn lại để test
    test_set = word_vector[200:250]
    test_data = np.array([i[:-1] for i in test_set])
    test_label = np.array([i[-1] for i in test_set])

    # ! Giảm chiều
    tsne_model = TSNE(n_components=2, perplexity=40, learning_rate=500, random_state=42)
    data_tsne = tsne_model.fit_transform(train_set)

    # ! Trực quan hóa
    # * trực quan hóa bằng đồ thị các điểm sau khi đã giảm chiều
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=train_label)
    plt.show()