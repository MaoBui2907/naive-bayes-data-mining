
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pickle


def is_letter_only(word):
    """Trả về true nếu là 1 từ chỉ toàn ký tự chữ cái, không số, không dấu,..."""
    for i in word:
        if not i.isalpha():
            return False
    return True


def make_vector(word_list, sentence_word_list, label):
    """Return vector with dim = len(word_list)"""
    temp = [sentence_word_list.count(i) for i in word_list]
    temp.append(label)
    return temp


if __name__ == "__main__":

    # ! Tiền xử lý
    # * đọc dữ liệu
    data = []
    with open('./Sarcasm_Headlines_Dataset.json', 'r') as f:
        data = [eval(i) for i in f]

    # * tạo danh sách từ, tách từ ở các headline và chuẩn hóa
    word_list = set()
    headline_word_list = []
    lem = WordNetLemmatizer()
    word_vector = []

    # * đọc danh sách stopwords nltk
    stop_words = stopwords.words('english')
    for i in range(len(data)):
        headline_word_list.append([])
        # * chuyển tiêu đề thành 1 list các từ được chuẩn hóa và lược bỏ
        for word in data[i]['headline'].split(' '):
            if is_letter_only(word) and lem.lemmatize(word) not in stop_words:
                temp = lem.lemmatize(word)
                word_list.add(temp)
                headline_word_list[i].append(temp)

    # print(word_list)
    # * tạo vector với BoW:
    for i in range(len(data)):
        word_vector.append(make_vector(
            word_list, headline_word_list[i], data[i]['is_sarcastic']))

    # * Ghi dữ liệu ra file data để chạy thuật toán
    # with open('vectorlist.bin', 'wb') as fp:
    #     pickle.dump(word_vector, fp)
