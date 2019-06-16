import pickle
import numpy as np
from collections import defaultdict
from scipy import sparse
import time

def validation(tp, tn, fp, fn):
    '''
    Trả về (accuracy, precision, recall, F1)
    @param tp true positive
    @param tn true negative
    @param fp false positive
    @param fn false negative
    '''
    # độ chính xác = số lần đoán đúng / số lần đoán
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = -1
    recall = -1
    F1 = -1
    try:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        F1 = 2 * (precision * recall) / (precision + recall)
    except:
        pass
    return (accuracy, precision, recall, F1)


def split_with_label(dataset):
    """
    Trả về index dòng dữ liệu trong các nhóm có nhãn giống nhau
    @param dataset là mảng dữ liệu training
    """
    # tao dictionary key, value
    output_data = defaultdict(list)
    for ind, val in enumerate(dataset):
        output_data[val[-1]].append(ind)
    return output_data
    # kieu dwx lieu sẽ có
    # {
    #     1 : [0,2,3,4,5,6],
    #     0 : [1,7, 8, 9]
    # }

def calculate_probability(label_index):
    """
    Trả về xác suất của mỗi nhóm trong dữ liệu với label
    @param label_index 1 dict chứa vị trí data có nhãn
    """
    probability = {}
    for label in label_index:
        probability[label] = len(label_index[label])
    sample_number = sum(probability.values())
    for label in probability:
        probability[label] /= sample_number
    return probability
    # {
    #     1: 6/10,
    #     0: 4/10
    # }


def calculate_conditional_probability(dataset, features_num, label_index, laplace_smooth=0):
    """
    Trả về một dict có key là các nhãn, value là một vector có số chiều bằng số thuộc tính, 
    lưu xác suất có điều kiện với MLE của mỗi thuộc tính. theo mô hình multinomial naive bayes
    @param dataset 1 mảng các sparse vector training
    @param features_num số thuộc tính sẽ sử dụng
    @param label_index 1 dict chứa vị trí data có nhãn
    @param laplace_smooth số laplace mặc định = 0
    """
    conditional_prob = {}
    for label in label_index:
        index = label_index[label]
        data_list = np.array([i[:features_num] for i in np.asarray(dataset)[index]])
        data_list = data_list.sum(axis=0) + laplace_smooth
        total_number = data_list.sum(axis=0)
        conditional_prob[label] = data_list / total_number
    return conditional_prob
    # {
    #     1: [0.4, 0.5, 0.6, 0.7],
    #     0: [0.6, 0.5, 0.4, 0.3]
    # }


def predict(test_vector, label_prob, feature_prob):
    """
    Trả về nhãn dự đoán với vector dữ liệu đầu vào
    @param test_vector: 1 sparse vector các thuộc tính đầu vào
    @param label_prop: 1 dict chứa xác suất của mỗi label
    @param feature_prob: 1 dict chứa 1 list các xác suất với mỗi thuộc tính
    """
    # * Ở đây sẽ dùng logarit tự nhiên với tích p(X|c) * p(c) trở thành ln(p(X|c)) + ln(p(c))
    predict = {}
    # * Tính ra tổng log với mỗi label
    for label in label_prob:
        # * Tính log của xác suất mỗi label
        p_label = np.log(label_prob[label])
        # * Lấy data khác 0 của một sparse vector
        data = test_vector.data
        # * Lấy index những vị trí khác 0 của một sparse vector
        index = test_vector.indices

        # * Tính ln(p(X|c)) của test vector
        p_features_log = sum(np.log(feature_prob[label][index]) * data)

        predict[label] = p_label + p_features_log

    # * Lấy giá trị thấp nhất
    min_log_label = min(predict.values())
    # * Đến đây đã được một dict chứa giá trị tỉ lệ thuật với xác suất cần tìm
    for label in label_prob:
        try:
            # * giảm bớt việc tính toán bằng cách trừ đi giá trị min
            predict[label] = np.exp(predict[label] - min_log_label)
        except:
            # * Nếu có lỗi, hoặc giá trị tìm ra quá lớn
            predict[label] = float('inf')
    
    sum_posterior = sum(predict.values())
    for label in predict:
        # * nếu giá trị quá lớn thì gán giá trị bằng 1
        if predict[label] == float('inf'):
            predict[label] = 1
        # * Tính xác suất bằng a/( a + b)
        else:
            predict[label] = predict[label] / sum_posterior
    return predict

if __name__ == "__main__":

    # ! Chuẩn bị dữ liệu
    # * Đọc dữ liệu
    with open('vectorlist.bin', 'rb') as fb:
        word_vector = pickle.load(fb)

    # * Giá trị số thuộc tính sẽ sử dụng (số chiều training) max = -1
    feature_num = -1

    # * chia dữ liệu 20000 dòng để train
    train_set = word_vector[:20]
    # print(train_set)

    # * lấy 6000 dòng còn lại để test
    test_set = word_vector[20:25]
    test_data = np.array([i[:feature_num] for i in test_set])
    test_label = np.array([i[-1] for i in test_set])
    # print(test_set)

    # ? Bắt đầu tính giờ chạy
    start = time.time()
    # ! training
    # * tính xác suất của nhãn
    label_index = split_with_label(train_set)
    # print(label_index)
    label_proba = calculate_probability(label_index)
    # print(label_proba)

    # * tính xác suất có điều kiện của mỗi thuộc tính theo mô hình multinomial naive bayes
    conditional_prob = calculate_conditional_probability(
        train_set, feature_num, label_index, laplace_smooth=1)

    # ! test
    predict_list = []
    for samp in test_data:
        predict_prob = predict(sparse.csr_matrix(samp), label_proba, conditional_prob)
        predict_label = list(predict_prob.keys())[list(predict_prob.values()).index(max(predict_prob.values()))]
        predict_list.append(predict_label)        
    # ? Kết thúc giờ chạy
    end = time.time()

    # ! Thống kê chỉ số
    result = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    for index in range(len(predict_list)):
        if predict_list[index] == 1 and test_label[index] == 1:
            result['tp'] += 1
        elif predict_list[index] == 0 and test_label[index] == 0:
            result['tn'] += 1
        elif predict_list[index] == 1 and test_label[index] == 0:
            result['fp'] += 1
        elif predict_list[index] == 0 and test_label[index] == 1:
            result['fn'] += 1

    # ! Hiển thị kết quả ra màn hình
    print("pred: ", np.asarray(predict_list))
    print("real: ", test_label)
    print("execute time: ", end -start)
    print('tp, tn, fp, fn: ' + str(result))
    print('accuracy, precision, recall, F1: ' +
          str(validation(result['tp'], result['tn'], result['fp'], result['fn'])))
