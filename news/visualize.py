
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import time

if __name__ == "__main__":
    # ? Bắt đầu tính giờ
    start = time.time()
    # ! Đọc dữ liệu
    # * Đọc dữ liệu
    with open('vectorlist.bin', 'rb') as fb:
        word_vector = pickle.load(fb)

    # * Giá trị số thuộc tính sẽ sử dụng (số chiều training) max = -1
    feature_num = 2000

    # * chia dữ liệu 20000 dòng để train
    train_set = word_vector[:2000]
    train_data = np.array([i[:feature_num] for i in train_set])
    train_label = np.array([i[-1] for i in train_set])

    # * lấy 6000 dòng còn lại để test
    test_set = word_vector[20000:]
    test_data = np.array([i[:feature_num] for i in test_set])
    test_label = np.array([i[-1] for i in test_set])

    # ? Xong bước chuẩn bị dữ liệu
    ready_check = time.time()
    # ! Giảm chiều
    tsne_model = TSNE(n_components=2, perplexity=10,
                      learning_rate=1000, random_state=42)
    data_tsne = tsne_model.fit_transform(train_set)

    # ? Xong bước giảm chiều dữ liệu
    reduction_done = time.time()
    # ! Trực quan hóa
    # * trực quan hóa bằng đồ thị các điểm sau khi đã giảm chiều
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=train_label)
    # ? Kết thúc
    end = time.time()

    # ! in thời gian
    print("load data time: ", ready_check - start)
    print("reduction time: ", reduction_done - ready_check)
    print("visualize time: ", end - reduction_done)
    print("total time: ", end - start)
    plt.show()
