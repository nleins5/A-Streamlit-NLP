"""
===============================================================================
ỨNG DỤNG NLP BẰNG STREAMLIT: DỊCH THUẬT VÀ SỬA LỖI CHÍNH TẢ
===============================================================================
Kiến trúc: File duy nhất app.py với 2 phần tách biệt:
- Phần 1: Helper Functions (Logic NLP)
- Phần 2: UI (Streamlit Layout & Forms)

Pipeline: User input -> detect_language() -> [run_translation() | run_spellcheck()] -> Output

Hàm bắt buộc:
1. detect_language(raw) -> str (trả '' nếu < 3 ký tự)
2. get_spellchecker(code) (decorate @st.cache_resource)
3. fix_typos(text, code) -> str (sửa ở mức token, giữ hoa/thường, giữ dấu câu)
4. run_translation(text, target) -> dict
5. run_spellcheck(text) -> dict

Anti-pattern (đã xử lý):
- @st.cache_resource cho SpellChecker (không khởi tạo lại mỗi lần rerun)
- Không xóa dấu câu khi sửa chính tả, detokenize lại chuẩn NLTK Treebank
- Bọc input trong st.form (tránh rerun liên tục khi đang gõ)
===============================================================================
"""

import re
import sys
import time
from typing import Dict, List, Tuple, Any, Optional

import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator
from spellchecker import SpellChecker
import langcodes

# -----------------------------------------------------------------------------
# KHỞI TẠO VÀ CẤU HÌNH BAN ĐẦU
# -----------------------------------------------------------------------------

# Cấu hình seed cho langdetect để kết quả luôn nhất quán
DetectorFactory.seed = 0

# Tải dữ liệu tokenizer NLTK an toàn
@st.cache_resource(show_spinner=False)
def init_nltk_data():
    """Tải dữ liệu punkt cần thiết cho NLTK."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
        except Exception:
            pass

init_nltk_data()

# Danh sách ngôn ngữ đích phổ biến cho Dịch thuật (Tên hiển thị -> Mã ngôn ngữ)
TARGET_LANGUAGES = {
    "Tiếng Việt (vi)": "vi",
    "Tiếng Anh (en)": "en",
    "Tiếng Pháp (fr)": "fr",
    "Tiếng Đức (de)": "de",
    "Tiếng Tây Ban Nha (es)": "es",
    "Tiếng Nhật (ja)": "ja",
    "Tiếng Hàn (ko)": "ko",
    "Tiếng Trung - Giản thể (zh-CN)": "zh-CN",
    "Tiếng Trung - Phồn thể (zh-TW)": "zh-TW",
    "Tiếng Nga (ru)": "ru",
    "Tiếng Ý (it)": "it",
    "Tiếng Thái (th)": "th",
}

# Danh sách ngôn ngữ mà pyspellchecker hỗ trợ chính thức
SPELLCHECK_SUPPORTED_LANGS = {
    'en': 'Tiếng Anh (English)',
    'es': 'Tiếng Tây Ban Nha (Spanish)',
    'fr': 'Tiếng Pháp (French)',
    'de': 'Tiếng Đức (German)',
    'pt': 'Tiếng Bồ Đào Nha (Portuguese)',
    'ru': 'Tiếng Nga (Russian)',
    'ar': 'Tiếng Ả Rập (Arabic)',
    'it': 'Tiếng Ý (Italian)',
    'lv': 'Tiếng Latvia (Latvian)',
    'eu': 'Tiếng Basque (Basque)',
    'nl': 'Tiếng Hà Lan (Dutch)',
    'fa': 'Tiếng Ba Tư (Persian)'
}

# Ánh xạ mã ngôn ngữ sang tên Tiếng Việt chuẩn hóa cho các ngôn ngữ phổ biến
COMMON_LANG_NAMES = {
    'vi': 'Tiếng Việt',
    'en': 'Tiếng Anh',
    'fr': 'Tiếng Pháp',
    'de': 'Tiếng Đức',
    'es': 'Tiếng Tây Ban Nha',
    'ja': 'Tiếng Nhật',
    'ko': 'Tiếng Hàn',
    'zh-cn': 'Tiếng Trung (Giản thể)',
    'zh-tw': 'Tiếng Trung (Phồn thể)',
    'zh': 'Tiếng Trung',
    'ru': 'Tiếng Nga',
    'it': 'Tiếng Ý',
    'pt': 'Tiếng Bồ Đào Nha',
    'ar': 'Tiếng Ả Rập',
    'nl': 'Tiếng Hà Lan',
    'th': 'Tiếng Thái',
}


# ==============================================================================
# PHẦN 1: HELPER FUNCTIONS (LOGIC NLP - BACKEND)
# ==============================================================================

def detect_language(raw: str) -> str:
    """
    HÀM BẮT BUỘC 1: Nhận diện ngôn ngữ từ văn bản thô.
    RÀNG BUỘC: Trả về chuỗi rỗng '' nếu văn bản ít hơn 3 ký tự.
    
    Args:
        raw (str): Văn bản thô đầu vào
        
    Returns:
        str: Mã ngôn ngữ (VD: 'vi', 'en') hoặc '' nếu < 3 ký tự / phát hiện thất bại.
    """
    cleaned = raw.strip() if raw else ""
    if len(cleaned) < 3:
        return ""
    try:
        return detect(cleaned)
    except Exception:
        return ""


@st.cache_resource
def get_spellchecker(code: str) -> Optional[SpellChecker]:
    """
    HÀM BẮT BUỘC 2 & ANTI-PATTERN 1: Load và cache instance SpellChecker theo mã ngôn ngữ.
    Sử dụng @st.cache_resource để không tạo mới SpellChecker mỗi lần app rerun.
    
    Args:
        code (str): Mã ngôn ngữ (VD: 'en', 'es')
        
    Returns:
        SpellChecker | None: Instance của SpellChecker hoặc None nếu ngôn ngữ không hỗ trợ (như 'vi').
    """
    if not code or code not in SPELLCHECK_SUPPORTED_LANGS:
        return None
    try:
        return SpellChecker(language=code)
    except Exception:
        return None


def preserve_case(original: str, corrected: str) -> str:
    """
    Helper giữ nguyên định dạng in hoa / in thường của từ gốc khi thay thế từ mới.
    - Nếu từ gốc VIẾT HOA HẾT -> Từ mới cũng VIẾT HOA HẾT.
    - Nếu từ gốc Viết Hoa Chữ Cái Đầu -> Từ mới cũng Viết Hoa Chữ Cái Đầu.
    - Ngược lại -> Viết chữ thường.
    """
    if not corrected:
        return original
    if original.isupper():
        return corrected.upper()
    if original.istitle() or (len(original) > 0 and original[0].isupper()):
        return corrected.capitalize()
    return corrected.lower()


def fix_typos(text: str, code: str) -> str:
    """
    HÀM BẮT BUỘC 3 & ANTI-PATTERN 2: Sửa lỗi chính tả ở mức token.
    Giữ nguyên in hoa/in thường của token gốc, không xóa dấu câu, và detokenize lại đúng chuẩn.
    
    Args:
        text (str): Văn bản gốc
        code (str): Mã ngôn ngữ
        
    Returns:
        str: Văn bản đã được sửa lỗi chính tả và detokenize hoàn chỉnh.
    """
    spell = get_spellchecker(code)
    if not spell:
        return text
        
    # Phân tách token với NLTK (giữ lại dấu câu)
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = re.findall(r'\w+|\S', text)
        
    # Lọc từ để tìm lỗi chính tả
    word_tokens = [t for t in tokens if t.isalpha()]
    misspelled_set = spell.unknown(word_tokens)
    
    corrected_tokens = []
    for token in tokens:
        if token.isalpha() and (token in misspelled_set or token.lower() in misspelled_set):
            suggestion = spell.correction(token)
            if suggestion:
                corrected_word = preserve_case(token, suggestion)
            else:
                corrected_word = token
            corrected_tokens.append(corrected_word)
        else:
            # Giữ nguyên dấu câu, số, ký tự đặc biệt và từ đúng
            corrected_tokens.append(token)
            
    # Detokenize lại đúng chuẩn TreebankWordDetokenizer (không bị mất dấu câu)
    return TreebankWordDetokenizer().detokenize(corrected_tokens)


def get_language_display_name(lang_code: str, locale: str = 'vi') -> str:
    """Helper lấy tên ngôn ngữ hiển thị bằng tiếng Việt."""
    if not lang_code:
        return "Không xác định"
    code_lower = lang_code.lower()
    if code_lower in COMMON_LANG_NAMES:
        return COMMON_LANG_NAMES[code_lower]
    try:
        language_obj = langcodes.Language.get(lang_code)
        display_name = language_obj.display_name(locale)
        return " ".join([word.capitalize() for word in display_name.split()])
    except Exception:
        return lang_code.upper()


def run_translation(text: str, target: str) -> Dict[str, Any]:
    """
    HÀM BẮT BUỘC 4: Pipeline Dịch thuật văn bản.
    Nhận diện ngôn ngữ nguồn -> Dịch sang ngôn ngữ đích qua GoogleTranslator.
    
    Args:
        text (str): Văn bản đầu vào
        target (str): Mã ngôn ngữ đích
        
    Returns:
        Dict: Kết quả dịch thuật và metadata
    """
    start_time = time.time()
    
    # 1. Pipeline: detect_language(text)
    source_code = detect_language(text)
    
    result = {
        "success": False,
        "original_text": text,
        "translated_text": "",
        "source_code": source_code,
        "source_name": get_language_display_name(source_code),
        "target_code": target,
        "target_name": get_language_display_name(target),
        "execution_time": 0.0,
        "error_message": ""
    }
    
    # Kiểm tra ràng buộc độ dài (< 3 ký tự)
    if not source_code:
        result["error_message"] = "Văn bản quá ngắn! Vui lòng nhập tối thiểu **3 ký tự**."
        return result
        
    # 2. Thực hiện dịch thuật
    try:
        translator = GoogleTranslator(source='auto', target=target)
        translated = translator.translate(text)
        if translated:
            result["success"] = True
            result["translated_text"] = translated
        else:
            result["error_message"] = "Không nhận được phản hồi từ dịch vụ dịch thuật."
    except Exception as e:
        err_msg = str(e)
        if any(kw in err_msg.lower() for kw in ["http", "connection", "connect", "timeout", "network"]):
            result["error_message"] = (
                "⚠️ **Không thể kết nối đến máy chủ Google Translator!**\n\n"
                "Ràng buộc: GoogleTranslator yêu cầu kết nối Internet. Vui lòng kiểm tra lại đường truyền mạng."
            )
        else:
            result["error_message"] = f"Lỗi dịch thuật: {err_msg}"
            
    result["execution_time"] = round(time.time() - start_time, 3)
    return result


def run_spellcheck(text: str) -> Dict[str, Any]:
    """
    HÀM BẮT BUỘC 5: Pipeline Kiểm tra và Sửa lỗi chính tả.
    Nhận diện ngôn ngữ -> Kiểm tra get_spellchecker -> Gọi fix_typos -> Chi tiết từng token.
    
    Args:
        text (str): Văn bản đầu vào
        
    Returns:
        Dict: Kết quả kiểm tra chính tả chi tiết
    """
    start_time = time.time()
    
    # 1. Pipeline: detect_language(text)
    detected_code = detect_language(text)
    detected_name = get_language_display_name(detected_code)
    
    result = {
        "detected_code": detected_code,
        "detected_name": detected_name,
        "is_supported": False,
        "supported_langs": SPELLCHECK_SUPPORTED_LANGS,
        "tokens_detail": [],
        "corrected_text": "",
        "misspelled_count": 0,
        "total_tokens": 0,
        "execution_time": 0.0,
        "error_message": ""
    }
    
    # Kiểm tra ràng buộc độ dài (< 3 ký tự)
    if not detected_code:
        result["error_message"] = "Văn bản quá ngắn! Vui lòng nhập tối thiểu **3 ký tự**."
        return result
        
    # 2. Lấy SpellChecker từ cache qua get_spellchecker()
    spell = get_spellchecker(detected_code)
    
    if not spell:
        supported_str = ", ".join([f"**{name} ({c})**" for c, name in SPELLCHECK_SUPPORTED_LANGS.items()])
        result["error_message"] = (
            f"🚫 **Ràng buộc ngôn ngữ:** `pyspellchecker` **KHÔNG hỗ trợ sửa lỗi chính tả cho {detected_name} (`{detected_code}`)**.\n\n"
            f"📌 Các ngôn ngữ được hỗ trợ bao gồm: {supported_str}.\n\n"
            f"👉 *Gợi ý:* Hãy thử nghiệm với một câu tiếng Anh (VD: *\"This is a testt with lazzy spellings\"*)."
        )
        return result
        
    result["is_supported"] = True
    
    # 3. Thực hiện sửa lỗi bằng fix_typos()
    corrected_text = fix_typos(text, detected_code)
    result["corrected_text"] = corrected_text
    
    # 4. Phân tích token chi tiết cho giao diện UI
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = re.findall(r'\w+|\S', text)
        
    result["total_tokens"] = len(tokens)
    word_tokens = [t for t in tokens if t.isalpha()]
    misspelled_set = spell.unknown(word_tokens)
    result["misspelled_count"] = len(misspelled_set)
    
    tokens_detail = []
    for token in tokens:
        if token.isalpha():
            is_wrong = token in misspelled_set or token.lower() in misspelled_set
            if is_wrong:
                suggestion = spell.correction(token)
                corr = preserve_case(token, suggestion) if suggestion else token
                raw_candidates = list(spell.candidates(token) or [])
                candidates = [preserve_case(token, c) for c in raw_candidates if c.lower() != token.lower()][:5]
            else:
                corr = token
                candidates = []
                
            tokens_detail.append({
                "token": token,
                "corrected": corr,
                "is_misspelled": is_wrong,
                "candidates": candidates
            })
        else:
            tokens_detail.append({
                "token": token,
                "corrected": token,
                "is_misspelled": False,
                "candidates": []
            })
            
    result["tokens_detail"] = tokens_detail
    result["execution_time"] = round(time.time() - start_time, 3)
    return result


# ==============================================================================
# PHẦN 2: STREAMLIT UI COMPONENTS (GIAO DIỆN NGƯỜI DÙNG VỚI ST.FORM)
# ==============================================================================

def inject_custom_css():
    """Injects modern, premium CSS styles into Streamlit app."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    .hero-header {
        background: linear-gradient(135deg, #1E1E2E 0%, #2A2B42 50%, #3B3C5E 100%);
        color: #FFFFFF;
        padding: 2.2rem 2rem;
        border-radius: 1.5rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(90deg, #FFFFFF, #B4B6F5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        font-size: 1.02rem;
        color: #A6ACCD;
        margin-top: 0.4rem;
        margin-bottom: 0;
    }

    .lang-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(123, 97, 255, 0.15);
        color: #9D85FF;
        border: 1px solid rgba(123, 97, 255, 0.3);
        padding: 0.35rem 0.85rem;
        border-radius: 2rem;
        font-size: 0.88rem;
        font-weight: 600;
    }

    .metric-box {
        background: linear-gradient(135deg, rgba(30, 30, 46, 0.6), rgba(42, 43, 66, 0.6));
        border: 1px solid rgba(123, 97, 255, 0.2);
        border-radius: 1rem;
        padding: 1rem 1.25rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #7B61FF;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #8E92B2;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }

    .token-misspelled {
        background-color: rgba(230, 59, 46, 0.2);
        color: #FF6B6B;
        border-bottom: 2px dashed #E63B2E;
        padding: 0.1rem 0.3rem;
        border-radius: 0.25rem;
        font-weight: 600;
    }

    section[data-testid="stSidebar"] {
        background-color: #12131C;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    """Renders the top header banner."""
    st.markdown("""
    <div class="hero-header">
        <h1 class="hero-title">🔤 NLP Intelligence Suite</h1>
        <p class="hero-subtitle">Pipeline Xử Lý Ngôn Ngữ Tự Nhiên: Dịch Thuật Đa Ngôn Ngữ & Sửa Lỗi Chính Tả</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar() -> str:
    """Renders sidebar navigation, info, and system constraints."""
    st.sidebar.title("🛠️ Điều Hướng & Cấu Hình")
    
    task_option = st.sidebar.radio(
        "Chọn Tác Vụ NLP:",
        options=[
            "🌐 1. Dịch Văn Bản (Translation)",
            "✏️ 2. Sửa Lỗi Chính Tả (Spell Check)"
        ],
        index=0
    )
    
    st.sidebar.divider()
    
    st.sidebar.markdown("### 📚 Các Hàm Core NLP")
    st.sidebar.code("""
- detect_language(raw) -> str
- get_spellchecker(code) [@cache]
- fix_typos(text, code) -> str
- run_translation(text, target) -> dict
- run_spellcheck(text) -> dict
    """, language="python")
    
    st.sidebar.divider()
    
    st.sidebar.markdown("### ⚠️ Ràng Buộc & Thiết Kế")
    st.sidebar.info("""
    - **Tối thiểu:** 3 ký tự (ngắn hơn trả về '').
    - **Caching:** `@st.cache_resource` lưu từ điển `SpellChecker`.
    - **Dấu câu:** Giữ nguyên dấu câu khi sửa chính tả.
    - **st.form:** Bọc Input để tránh rerun liên tục khi nhập.
    """)
    
    st.sidebar.caption("Antigravity NLP Engine • Optimized Architecture")
    return task_option


def render_translation_ui():
    """Renders Task 1 UI using st.form to prevent continuous reruns."""
    st.markdown("## 🌐 Tác Vụ 1: Dịch Văn Bản (Text Translation)")
    st.caption("Pipeline: User input -> detect_language() -> run_translation() -> output")
    
    # Nút bấm nạp mẫu văn bản ngoài form
    c1, c2, _ = st.columns([1, 1, 2])
    with c1:
        if st.button("📝 Mẫu Tiếng Việt", key="btn_sample_tr_vi"):
            st.session_state["tr_text_input"] = "Xin chào! Ứng dụng này giúp bạn dịch thuật văn bản tự động và nhanh chóng."
    with c2:
        if st.button("📝 Mẫu Tiếng Anh", key="btn_sample_tr_en"):
            st.session_state["tr_text_input"] = "Artificial intelligence and natural language processing are transforming our world."

    # ANTI-PATTERN 3: Bọc input trong st.form để tránh rerun liên tục khi người dùng gõ phím
    with st.form(key="translation_form"):
        input_text = st.text_area(
            "Nhập văn bản cần dịch:",
            value=st.session_state.get("tr_text_input", ""),
            height=140,
            placeholder="Nhập nội dung văn bản (tối thiểu 3 ký tự)..."
        )
        
        col_lang, col_btn = st.columns([2, 1])
        with col_lang:
            target_lang_label = st.selectbox(
                "Chọn ngôn ngữ đích:",
                options=list(TARGET_LANGUAGES.keys()),
                index=0
            )
            target_code = TARGET_LANGUAGES[target_lang_label]
            
        with col_btn:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            submit_tr = st.form_submit_button("🚀 Tiến Hành Dịch", type="primary", use_container_width=True)
            
    # Chạy pipeline khi submit form
    if submit_tr:
        if not input_text or len(input_text.strip()) < 3:
            st.warning("Văn bản quá ngắn! Vui lòng nhập tối thiểu **3 ký tự**.")
            return
            
        with st.spinner("🔍 Đang chạy pipeline nhận diện & dịch thuật..."):
            res = run_translation(input_text, target_code)
            
        if not res["success"]:
            st.error(res["error_message"])
        else:
            st.success("✅ Dịch thuật hoàn tất!")
            
            st.markdown(f"""
            <div style="margin-bottom: 1rem;">
                <span class="lang-badge">
                    🔍 Ngôn ngữ nhận diện (`detect_language`): <b>{res['source_name']} ({res['source_code']})</b>
                </span>
                <span style="margin-left: 0.5rem; color: #8E92B2; font-size: 0.85rem;">
                    ⏱️ Thời gian xử lý: {res['execution_time']}s
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            out_col1, out_col2 = st.columns(2)
            with out_col1:
                st.markdown(f"### 📄 Văn bản gốc ({res['source_name']})")
                st.info(res["original_text"])
                
            with out_col2:
                st.markdown(f"### 🎯 Kết quả dịch ({res['target_name']})")
                st.code(res["translated_text"], language="text")


def render_spellcheck_ui():
    """Renders Task 2 UI using st.form to prevent continuous reruns."""
    st.markdown("## ✏️ Tác Vụ 2: Sửa Lỗi Chính Tả (Spell Checking)")
    st.caption("Pipeline: User input -> detect_language() -> get_spellchecker() -> fix_typos() -> output")
    
    c1, c2, _ = st.columns([1, 1, 2])
    with c1:
        if st.button("📝 Mẫu Tiếng Anh (Lỗi)", key="btn_sample_sc_en"):
            st.session_state["sc_text_input"] = "Heo World! THIS IS A LAZZY TESTT sentence with incorect spellings."
    with c2:
        if st.button("📝 Mẫu Tiếng Việt (Thử nghiệm)", key="btn_sample_sc_vi"):
            st.session_state["sc_text_input"] = "Xin chào đây là một câu tiếng Việt thử nghiệm."

    # ANTI-PATTERN 3: Bọc input trong st.form
    with st.form(key="spellcheck_form"):
        input_text = st.text_area(
            "Nhập văn bản cần kiểm tra chính tả:",
            value=st.session_state.get("sc_text_input", ""),
            height=140,
            placeholder="Nhập văn bản (Ví dụ tiếng Anh: 'Heo world this is a testt')..."
        )
        submit_sc = st.form_submit_button("🔍 Kiểm Tra & Sửa Lỗi Chính Tả", type="primary")

    # Chạy pipeline khi submit form
    if submit_sc:
        if not input_text or len(input_text.strip()) < 3:
            st.warning("Văn bản quá ngắn! Vui lòng nhập tối thiểu **3 ký tự**.")
            return
            
        with st.spinner("⏳ Đang chạy pipeline kiểm tra chính tả..."):
            res = run_spellcheck(input_text)
            
        if not res["is_supported"]:
            st.markdown(f"""
            <div style="background-color: rgba(230, 59, 46, 0.1); border-left: 4px solid #E63B2E; padding: 1rem 1.25rem; border-radius: 0.5rem; margin-top: 1rem;">
                <h4 style="color: #E63B2E; margin-top: 0;">⚠️ Thông Báo Ràng Buộc Ngôn Ngữ</h4>
                <p>Ngôn ngữ được nhận diện: <b>{res['detected_name']} ({res['detected_code']})</b></p>
                <p>{res['error_message']}</p>
            </div>
            """, unsafe_allow_html=True)
            return
            
        if res["error_message"]:
            st.error(res["error_message"])
            return
            
        st.success("✅ Kiểm tra & Sửa lỗi chính tả hoàn tất!")
        
        st.markdown(f"""
        <div style="margin-bottom: 1.5rem;">
            <span class="lang-badge">
                🔍 Ngôn ngữ nhận diện (`detect_language`): <b>{res['detected_name']} ({res['detected_code']})</b>
            </span>
            <span style="margin-left: 0.5rem; color: #8E92B2; font-size: 0.85rem;">
                ⏱️ Thời gian xử lý: {res['execution_time']}s
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{res['total_tokens']}</div>
                <div class="metric-label">Tổng số Token</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            color = "#E63B2E" if res['misspelled_count'] > 0 else "#2ECC71"
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value" style="color: {color};">{res['misspelled_count']}</div>
                <div class="metric-label">Từ viết sai chính tả</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            accuracy = round((1 - res['misspelled_count'] / max(res['total_tokens'], 1)) * 100, 1)
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value" style="color: #2ECC71;">{accuracy}%</div>
                <div class="metric-label">Độ chính xác từ vựng</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.divider()
        
        tab1, tab2, tab3 = st.tabs([
            "✨ Văn Bản Hoàn Chỉnh (fix_typos Output)",
            "🔍 Trực Quan Hóa Lỗi (Highlighted)",
            "📊 Chi Tiết Token & Gợi Ý Sửa"
        ])
        
        with tab1:
            st.markdown("### 📝 Kết quả từ `fix_typos()` (Đã giữ hoa/thường & dấu câu):")
            st.code(res["corrected_text"], language="text")
            
        with tab2:
            st.markdown("### 🎨 Trực quan hóa (Từ viết sai được highlight màu đỏ):")
            highlighted_html = ""
            for item in res["tokens_detail"]:
                if item["is_misspelled"]:
                    highlighted_html += f'<span class="token-misspelled" title="Sửa thành: {item["corrected"]}">{item["token"]}</span> '
                else:
                    highlighted_html += f'{item["token"]} '
                    
            st.markdown(f"""
            <div style="background-color: rgba(255, 255, 255, 0.05); padding: 1.25rem; border-radius: 0.75rem; border: 1px solid rgba(255,255,255,0.1); line-height: 1.8;">
                {highlighted_html}
            </div>
            """, unsafe_allow_html=True)
            st.caption("💡 Rê chuột vào từ màu đỏ để xem từ gợi ý sửa.")
            
        with tab3:
            st.markdown("### 📋 Phân tích từng Token:")
            table_data = []
            for idx, item in enumerate(res["tokens_detail"], 1):
                status_str = "❌ Lỗi chính tả" if item["is_misspelled"] else "✅ Đúng chính tả"
                candidates_str = ", ".join(item["candidates"]) if item["candidates"] else "None"
                table_data.append({
                    "STT": idx,
                    "Token Gốc": item["token"],
                    "Trạng Thái": status_str,
                    "Token Đã Sửa": item["corrected"],
                    "Gợi Ý Khác": candidates_str
                })
                
            st.dataframe(
                table_data,
                use_container_width=True,
                column_config={
                    "STT": st.column_config.NumberColumn("STT", width="small"),
                    "Token Gốc": st.column_config.TextColumn("Token Gốc"),
                    "Trạng Thái": st.column_config.TextColumn("Trạng Thái"),
                    "Token Đã Sửa": st.column_config.TextColumn("Token Đã Sửa"),
                    "Gợi Ý Khác": st.column_config.TextColumn("Gợi Ý Khác (Candidates)")
                }
            )


# ==============================================================================
# PHẦN 3: MAIN ENTRY POINT
# ==============================================================================

def main():
    st.set_page_config(
        page_title="NLP Suite - Dịch Thuật & Sửa Lỗi Chính Tả",
        page_icon="🔤",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    inject_custom_css()
    render_header()
    
    selected_task = render_sidebar()
    
    if "1. Dịch Văn Bản" in selected_task:
        render_translation_ui()
    elif "2. Sửa Lỗi Chính Tả" in selected_task:
        render_spellcheck_ui()


if __name__ == "__main__":
    main()
