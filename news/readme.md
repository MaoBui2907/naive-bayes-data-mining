# Khai thác dữ liệu tiêu đề tin tức với naive bayes

## Nguồn dữ liệu:

https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection

## Bài toán:
- Phát hiện tin bài nhảm qua các tiêu đề bài viết.

## Dữ liệu:
- Cấu trúc:
    + Gồm 26709 dòng dữ liệu gồm headline, links và nhãn là tin bài mỉa mai hay không (0 hoặc 1).
    + Tiêu đề tin là một dòng văn bản bằng tiếng Anh.

- Phương pháp khai thác:
    + Chỉ dùng 10000 dòng dữ liệu đầu tiên để phục vụ việc huấn luyện
    + Dùng khoảng 6000 dòng cho quá trình test
    + Dùng kỹ thuật Bag of Words để chuyển dữ liệu dạng text sang các vector số.
    + Nếu số lượng thuộc tính quá lớn, chỉ chọn ra khoảng 1000 thuộc tính xuất hiện nhiều nhất.
    + Dùng thuật toán naive bayes để xác định nhãn cho dữ liệu.
    + Tính toán và thống kê các chỉ số.

- Tiền xử lý:
    + Xử lý dữ liệu text:
        - Tách từ
        - Loại bỏ stop words
        - lemmatizing
        - Chỉ chọn 5000 từ xuất hiện nhiều nhất làm thuộc tính

    + Xử lý dữ liệu dạng số
        - Giảm chiều dữ liệu
        - Trực quan hóa dữ liệu bằng đồ thị

- Học với thuật toán naive bayes
    + thử nghiệm với code tự viết
    + Chạy lại với code của thư viện sklearn

- Rút ra kết luận, đánh giá
## Cài đặt môi trường
- Yêu cầu python 3 và pip
- Chạy pip install -r requirements.txt để cài các thư viện cần thiết
- Tải bộ corpora stop_words và wordnet của nltk

## Chạy code:
- Chạy __python make_vector.py__ để tạo file dữ liệu vector
- __python sklearn_nb.py__ để học với hàm dựng sẵn ở sklearn
- __python naive_bayes.py__ để học với hàm tự dựng