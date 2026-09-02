# CLAUDE.md - Bộ Nhớ & Quy Chuẩn Dự Án NLP Streamlit Suite

## 📌 Tổng Quan Dự Án
Ứng dụng NLP đơn tệp (`app.py`) giao diện 2 tab phục vụ tác vụ **Dịch Văn Bản** và **Sửa Lỗi Chính Tả**.

## 🛠️ Công Nghệ Sử Dụng
- **Python 3.10+**
- **Streamlit**: Framework UI web cho Data/NLP
- **langdetect**: Nhận diện ngôn ngữ gốc tự động
- **pyspellchecker**: Sửa lỗi chính tả theo từng token
- **nltk**: Tokenize & Detokenize với `TreebankWordDetokenizer`
- **deep-translator**: Dịch thuật đa ngôn ngữ qua `GoogleTranslator`
- **langcodes & language_data**: Chuẩn hóa tên ngôn ngữ tiếng Việt

---

## 📐 Quy Tắc Kiến Trúc & Thiết Kế (Mandatory Constraints)

1. **Cấu trúc Đơn Tệp (`app.py`)**:
   - Phần 1: Helper Functions (Logic NLP Backend)
   - Phần 2: UI Components (Streamlit Layout & Forms)

2. **5 Hàm Cốt Lõi**:
   - `detect_language(raw: str) -> str`: Trả về `''` nếu văn bản ít hơn 3 ký tự.
   - `get_spellchecker(code: str) -> Optional[SpellChecker]`: Được trang bị `@st.cache_resource` để chống re-instantiate từ điển.
   - `preserve_case(original: str, corrected: str) -> str`: Giữ nguyên in hoa chữ đầu, viết hoa toàn bộ hoặc chữ thường. Áp dụng cho cả từ sửa và danh sách từ gợi ý (`candidates`).
   - `fix_typos(text: str, code: str) -> str`: Tokenize bảo toàn dấu câu ➔ Sửa lỗi ➔ Detokenize bằng `TreebankWordDetokenizer`.
   - `run_translation(text: str, target: str) -> dict` & `run_spellcheck(text: str) -> dict`: Pipelines đóng gói kết quả và metadata.

3. **Chống Anti-Pattern Trong Streamlit**:
   - `@st.cache_resource` cho từ điển chính tả.
   - Bọc toàn bộ ô nhập liệu trong `st.form` để tránh ứng dụng tự động rerun liên tục khi người dùng đang nhập văn bản.

---

## 🚀 Lệnh Khởi Chạy & Kiểm Thử

```bash
# 1. Cài đặt phụ thuộc
pip install -r requirements.txt

# 2. Khởi chạy ứng dụng
streamlit run app.py

# 3. Kiểm thử cú pháp sạch
python3 -c "import app"

# 4. Kiểm thử hàm fix_typos inline
python3 -c "from app import fix_typos; print(fix_typos('Yesturday, I recieveed a mesage from my freind.', 'en'))"
```
