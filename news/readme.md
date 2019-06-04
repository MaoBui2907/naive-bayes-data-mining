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
    + Dùng 20000 dòng dữ liệu đầu tiên để phục vụ việc huấn luyện
    + Dùng khoảng 6000 dòng còn lại cho quá trình test
    + Dùng kỹ thuật Bag of Words để chuyển dữ liệu dạng text sang các vector số.
    + Giảm chiều dữ liệu để trực quan hóa
    + Dùng thuật toán naive bayes để xác định nhãn cho dữ liệu.
    + Tính toán và thống kê các chỉ số.

- Tiền xử lý:
    + Xử lý dữ liệu text:
        - Tách từ
        - Loại bỏ stop words
        - lemmatizing
    
    + Xử lý dữ liệu dạng số
        - Giảm chiều dữ liệu
        - Trực quan hóa dữ liệu bằng đồ thị

- Học với thuật toán naive bayes
    + thử nghiệm với code tự viết
    + Chạy lại với code của thư viện sklearn

- Rút ra kết luận, đánh giá

## Chạy code:
- __python make_vector.py__ để tạo file dữ liệu vector (Yêu cầu đầu tiên)
- __python visualize.py__ để vẽ biểu đồ dữ liệu trên không gian 2 chiều
- __python sklearn_nb.py__ để train và test với hàm dựng sẵn ở sklearn
- __python naive_bayes.py__ để train và test với hàm tự dựng