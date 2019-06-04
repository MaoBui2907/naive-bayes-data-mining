import pickle
import numpy as np
from collections import defaultdict


def validation(tp, tn, fp, fn):
    '''
    Trả về (accuracy, precision, recall, F1)
    @param tp true positive
    @param tn true negative
    @param fp false positive
    @param fn false negative
    '''
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
    output_data = defaultdict(list)
    for ind, val in enumerate(dataset):
        output_data[val[-1]].append(ind)
    return output_data


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


def calculate_conditional_probability(dataset, label_index, laplace_smooth=0):
    """
    Trả về một dict có key là các nhãn, value là một vector có số chiều bằng số thuộc tính, 
    lưu xác suất có điều kiện với MLE của mỗi thuộc tính. theo mô hình multinomial naive bayes
    @param dataset 1 mảng các sparse vector training
    @param label_index 1 dict chứa vị trí data có nhãn
    @param laplace_smooth số laplace mặc định = 0
    """
    conditional_prob = {}
    for label in label_index:
        index = label_index[label]
        data_list = np.array([i[:-1] for i in np.asarray(dataset)[index]])
        data_list = data_list.sum(axis=0) + laplace_smooth
        total_number = data_list.sum(axis=0)
        conditional_prob[label] = data_list / total_number
    return conditional_prob


def predict(test_vector, label_prob, feature_prob):
    """
    Trả về nhãn dự đoán với vector dữ liệu đầu vào
    @param test_vector: 1 sparse vector các thuộc tính đầu vào
    @param label_prop: 1 dict chứa xác suất của mỗi label
    @param feature_prob: 1 dict chứa 1 list các xác suất với mỗi thuộc tính
    """
    # * Ở đây sẽ dùng logarit tự nhiên với tích p(X|c) * p(c) trở thành ln(p(X|c)) + ln(p(c))

    predict = {}
    for label in label_prob:
        # * Tính log của xác suất mỗi label
        p_label = np.log(label_prob[label])
        # * Lấy data khác 0 của một sparse vector
        data = test_vector.data
        # * Lấy index những vị trí khác 0 của một sparse vector
        index = test_vector.indices

        # * Tính ln(p(X|c)) của test vector
        p_features_log = np.log(feature_prob[index]) * data
        
        p_predict_label = np.exp(p_features_log + p_label)
        predict[label] = p_predict_label
    return predict


if __name__ == "__main__":

    # ! Chuẩn bị dữ liệu
    # * Đọc dữ liệu
    with open('vectorlist.bin', 'rb') as fb:
        word_vector = pickle.load(fb)

    # * chia dữ liệu 20000 dòng để train
    train_set = word_vector[:100]
    # print(train_set)

    # * lấy 6000 dòng còn lại để test
    test_set = word_vector[100:110]
    # print(test_set)

    # ! training
    # * tính xác suất của nhãn
    label_index = split_with_label(train_set)
    # print(label_index)
    label_proba = calculate_probability(label_index)
    # print(label_proba)

    # * tính xác suất có điều kiện của mỗi thuộc tính theo mô hình multinomial naive bayes
    conditional_prob = calculate_conditional_probability(
        train_set, label_index, laplace_smooth=1)

    print(label_proba)
    print(conditional_prob)

    # ! test
    result = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    for samp in test_set:
        samp_features = samp[:-1]
        print(samp_features)
        predict = predict(samp_features, label_proba, conditional_prob)
        real = samp[-1]
        print("preidct", predict)
        continue
        if predict == 1 and real == 1:
            result['tp'] += 1
        elif predict == 0 and real == 0:
            result['tn'] += 1
        elif predict == 1 and real == 0:
            result['fp'] += 1
        elif predict == 0 and real == 1:
            result['fn'] += 1

    print('tp, tn, fp, fn: ' + str(result))
    print('accuracy, precision, recall, F1: ' +
          str(validation(result['tp'], result['tn'], result['fp'], result['fn'])))
