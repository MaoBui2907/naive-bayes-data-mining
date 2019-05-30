import pickle
import numpy as np
from collections import defaultdict

def split_with_label(dataset):
    """Trả về dữ liệu được chia thành các nhóm có nhãn mỗi nhóm là giống nhau"""
    output_data = defaultdict(list)
    for ind, val in enumerate(dataset):
        output_data[val[-1]].append(ind)
    return output_data

def calculate_probability(label_index):
    """Trả về xác suất của mỗi nhóm trong dữ liệu với label"""
    probability = {}
    for label in label_index:
        probability[label] = len(label_index[label])
    sample_number = sum(probability.values())
    for label in probability:
        probability[label] /= sample_number
    return probability

def calculate_conditional_probability(dataset, label_index):
    """Trả về một dict có key là các nhãn, value là một vector có số chiều bằng số thuộc tính, 
    lưu xác suất có điều kiện với MLE của mỗi thuộc tính"""
    conditional_prob = {}
    for label in label_index:
        index = label_index[label]
        data = dataset[index, : -1].sum(axis=0)
        print (data)



if __name__ == "__main__":

    # ! Chuẩn bị dữ liệu
    # * Đọc dữ liệu
    with open('vectorlist.bin', 'rb') as fb:
        word_vector = pickle.load(fb)

    # * chia dữ liệu 20000 dòng để train
    train_set = word_vector[:10]
    # print(train_set)

    # * lấy 6000 dòng còn lại để test
    test_set = word_vector[200:250]
    # print(test_set)

    # * tính xác suất của nhãn
    label_index = split_with_label(train_set)
    # print(label_index)
    label_proba = calculate_probability(label_index)
    # print(label_proba)

    # * tính xác suất có điều kiện của mỗi thuộc tính với điều kiện là nhãn Maximum Likelihood Estimation
    calculate_conditional_probability(train_set, label_index)


