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