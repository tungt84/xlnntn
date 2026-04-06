Yêu cầu: 

I. Huấn luyện mô hình. Mô hình sẽ được huấn luyện ở phía người sử dụng với:
 1) Số lượng tham số không quá 40 triệu.
 2) Sử dụng pytorch và Hugging Face transformers để phù hợp với chương trình chấm điểm trên CodaBench.
 3) Tương thích với các thư viện sau: Việc cài đặt mô hình có thể sử dụng thư viện Scikit-Learn (1.7.2), Scipy (1.15.3), PyTorch (2.5.1), Transformers (4.57.3).

II. Tokenizer và kiến trúc mô hình. Xây dựng một tokenizer để phân tách câu tiếng Anh thành các tokens phù hợp với mô hình và huấn luyện một mô hình NLI dựa vào các file hỗ trợ được đặt trong NLI-Starting-Kit.zip. Ý nghĩa của các file trong NLI-Starting-Kit.zip như sau:
 1) File train.py: xây dựng một tokenizer và lưu trong thư mục MODEL. Sau đó, sử dụng kiến trúc mạng NLI đã được định nghĩa để huấn luyện mô hình NLI trên ngữ liệu tiếng Anh. Mô hình tốt nhất được lưu trong thư mục MODEL (cùng thư mục với tokenizer). Ngữ liệu sử dụng trong file train.py là MNLI. Người tham gia có thể chọn và tổng hợp nhiều bộ ngữ liệu để huấn luyện. File train.py nhằm gợi ý cách xây dựng một tokenizer và huấn luyện một mô hình. Người tham gia không bắt buộc phải làm đúng như file train.py.
 2) File test.py: load mô hình đã huấn luyện để thử nghiệm. Người tham gia có thể dùng chương trình này để kiểm tra kết quả huấn luyện mô hình trước khi nộp lên CodaBench. Nếu có lỗi khi chạy file test.py thì chương trình chấm điểm cũng sẽ bị lỗi. Người tham gia có thể thay đổi ngữ liệu để test, nhưng không nên thay đổi những nội dung khác trong file test.py.
 3) File model.py: mô tả kiến trúc mạng của mô hình và cách xử lý dữ liệu đầu vào. File model.py có 2 lớp NLIConfig và NLI, đồng thời có hai hàm collate_fn và tokenizes.
  - Hàm collate_fn nhận dữ liệu đầu vào là một danh sách các lô chứa các mẫu dữ liệu ở dạng danh sách token ID và các nhãn tương ứng với từng mẫu dữ liệu. Hàm này cần biến đổi dữ liệu đầu vào sao cho phù hợp với các tham số của phương thức forward ở lớp NLI.
  - Hàm tokenizes sẽ sử dụng tokenizer để phân tách dữ liệu đầu vào thành chuỗi các token với chiều dài lớn nhất. Nếu mô hình có giới hạn về số lượng input tokens, cần chỉ ra trong hàm tokenizes này.
  - Lớp NLIConfig, cần phải kế thừa từ lớp PreTrainedConfig, để ghi nhận các tham số gồm vocab_size, hidden_size và nclass. Trong đó, vocab_size cho biết số token trong tokenizer đã xây dựng; hidden_size là số chiều của lớp word-embedding và nclass là số nhãn cần dự đoán. Lớp NLIConfig có thể tùy chỉnh.
  - Lớp NLI, phải kế thừa từ lớp PreTrainedModel, mô tả kiến trúc mạng và quá trình tính toán. Người tham gia cần override 2 phương thức quan trọng.
   - init() để mô tả các lớp mạng trong kiến trúc. Kiến trúc mạng trong file được cung cấp gồm 1 lớp embedding, 1 lớp lstm và 1 lớp linear. Người tham gia có thể thay đổi kiến trúc theo nhu cầu của mình nhưng lưu ý số lượng tham số không quá 40 triệu.
   - forward() để tính toán logits và loss từ dữ liệu đầu vào và kết quả mong đợi. Các tham số của phương thức forward được chọn tùy thuộc vào phương pháp tính toán của kiến trúc mạng do người tham gia xác định. Bộ tham số của phương thức forword cần xác định đầy đủ trong hàm collate_fn.

Gợi ý:
- Tokenizer đang dùng WordLevel quá lớn  chọn phương án khác 
- Huấn luyện mô hình t5 ?