
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pickle
import time
from collections import defaultdict

# h1 = [a,b,c, a]   0
# h2 = [b,c,d]      1
# h3 = [a,c,d]      ?
# h = [a,b,c,d]
# h1 = [2,1,1,0]    0
# h2 = [0,1,1,1]    1
# h3 = [1,0,1,1] * L=0 ?
# h3 = [1,0,1,1] * L=1 ?

def is_letter_only(word):
    """Trả về true nếu là 1 từ chỉ toàn ký tự chữ cái, không số, không dấu,..."""
    if word is "":
        return False
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
    
    # ? Bắt đầu tính giờ
    start = time.time()
    
    # ! Tiền xử lý
    # * đọc dữ liệu
    data = []
    with open('./data.json', 'r') as f:
        data = [eval(i) for i in f]

    # * tạo danh sách từ, tách từ ở các headline và chuẩn hóa
    count_word_list = defaultdict(lambda: 0)
    headline_word_list = []
    lem = WordNetLemmatizer()
    word_vector = []

    # * đọc danh sách stopwords nltk
    stop_words = stopwords.words('english')
    for i in range(len(data)):
        headline_word_list.append([])
        # * chuyển tiêu đề thành 1 list các từ được chuẩn hóa và lược bỏ
        for word in data[i]['headline'].split(' '):
            # * Kiểm tra nếu từ đó không thuộc stopwords và chỉ gồm các ký tự
            if is_letter_only(word) and lem.lemmatize(word) not in stop_words:
                temp = lem.lemmatize(word)
                count_word_list[temp] += 1
                headline_word_list[i].append(temp)

    # * Sắp xếp dictionary được tạo ra với giá trị từ xuất hiện nhiều ở phía trước
    sorted_word_list = [(k, count_word_list[k]) for k in sorted(count_word_list, key=count_word_list.get, reverse=True)]

    # * Lấy danh sách các từ từ bộ dữ liệu đã được tạo với 
    word_list = [key for key, val in sorted_word_list]

    # print(word_list)
    # * tạo vector với BoW:
    for i in range(len(data)):
        word_vector.append(make_vector(
            word_list, headline_word_list[i], data[i]['is_sarcastic']))
    
    # ? Thời gian tiền xử lý
    end_pre = time.time()

    # * Ghi dữ liệu ra file data để chạy thuật toán
    with open('vectorlist.bin', 'wb') as fp:
        pickle.dump(word_vector, fp)
    
    # ? Dừng đồng hồ
    end = time.time()

    # ! Hiện thời gian chạy
    print("preprocess time: ", end_pre - start)
    print("write vector data: ", end - end_pre)
    print("total time: ", end - start)
