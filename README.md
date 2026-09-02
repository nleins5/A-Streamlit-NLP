# Streamlit NLP Suite: Dịch Thuật & Sửa Lỗi Chính Tả

Ứng dụng web Xử lý Ngôn ngữ Tự nhiên (NLP) xây dựng bằng **Streamlit**, hỗ trợ 2 tác vụ chính:
1. **Dịch văn bản (Text Translation):** Tự động nhận diện ngôn ngữ nguồn và dịch sang ngôn ngữ đích mong muốn.
2. **Sửa lỗi chính tả (Spell Checking):** Tự động nhận diện ngôn ngữ, phân tách token bằng NLTK và kiểm tra/sửa lỗi chính tả từng token.

---

## Kiến Trúc Dự Án (`app.py`)

Dự án tuân thủ kiến trúc file duy nhất `app.py` nhưng được phân tách lớp rõ ràng:
- **Phần 1: NLP Helper Functions (Backend / Logic)**:
  - `validate_input_text()`: Kiểm tra ràng buộc độ dài văn bản tối thiểu (>= 3 ký tự).
  - `get_language_display_name()`: Ánh xạ mã ngôn ngữ sang tên tiếng Việt dễ đọc (`langcodes`).
  - `helper_detect_language()`: Nhận diện ngôn ngữ tự động (`langdetect`).
  - `helper_translate_text()`: Dịch văn bản qua `deep-translator` (`GoogleTranslator`) kèm xử lý kết nối Internet.
  - `helper_spell_check_text()`: Kiểm tra & sửa lỗi từng token qua `nltk` và `pyspellchecker` kèm xử lý ràng buộc ngôn ngữ không hỗ trợ (đặc biệt Tiếng Việt `vi`).
- **Phần 2: UI Components & Styling (Streamlit / Frontend)**:
  - Header banner, custom CSS styling (vibrant dark theme, badges, glassmorphism cards).
  - Sidebar chuyển đổi tác vụ, hiển thị ràng buộc & danh sách thư viện.
  - Giao diện tác vụ 1: Dịch thuật đa ngôn ngữ.
  - Giao diện tác vụ 2: Sửa lỗi chính tả (Highlight từ sai, bảng phân tích token chi tiết, thông báo ràng buộc).

---

## Các Thư Viện Sử Dụng

- `streamlit`: Khung ứng dụng Web giao diện người dùng.
- `langdetect`: Tự động nhận diện ngôn ngữ của văn bản đầu vào.
- `deep-translator`: Thư viện kết nối dịch thuật GoogleTranslator.
- `pyspellchecker`: Thư viện kiểm tra lỗi chính tả dựa trên khoảng cách Levenshtein & tần suất từ.
- `nltk`: Thư viện NLP dùng để tokenize văn bản thành các token từ.
- `langcodes` & `language_data`: Chuyển đổi mã ngôn ngữ (ISO 639-1) thành tên ngôn ngữ hiển thị.

---

## Hướng Dẫn Chạy Ứng Dụng

### 1. Cài đặt môi trường & dependencies
```bash
# Tạo môi trường ảo
python3 -m venv venv
source venv/bin/activate

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### 2. Chạy ứng dụng Streamlit
```bash
streamlit run app.py
```

Sau khi chạy lệnh, trình duyệt sẽ tự động mở trang web ứng dụng tại địa chỉ `http://localhost:8501`.

---

## Ràng Buộc Hệ Thống

1. **Độ dài tối thiểu:** Văn bản cần có độ dài từ **3 ký tự trở lên**. Nếu ngắn hơn, ứng dụng sẽ thông báo nhắc nhở mà không thực hiện xử lý NLP.
2. **Kết nối Internet:** Tác vụ dịch thuật sử dụng `GoogleTranslator` yêu cầu có kết nối mạng Internet. Ứng dụng đã bắt lỗi kết nối và hiển thị cảnh báo thân thiện nếu mất mạng.
3. **Giới hạn `pyspellchecker` với Tiếng Việt:** Thư viện `pyspellchecker` không hỗ trợ bộ từ điển Tiếng Việt (`vi`). Khi người dùng kiểm tra văn bản Tiếng Việt, ứng dụng sẽ nhận diện và thông báo rõ ràng ràng buộc này, đồng thời gợi ý các ngôn ngữ được hỗ trợ (`en`, `es`, `fr`, `de`, `pt`, `ru`, `ar`, `it`...).
