import re
import statistics
from collections import Counter

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QBrush, QColor, QIcon, QKeySequence, QLinearGradient, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QTextEdit,
    QProgressBar, QFrame, QGraphicsDropShadowEffect, QScrollArea
)

try:
    from langdetect import detect, LangDetectException
except ImportError:
    detect = None
    LangDetectException = Exception

try:
    from razdel import sentenize
except ImportError:
    sentenize = None

try:
    from docx import Document
except ImportError:
    Document = None

def split_sentences(text: str):
    text = text.strip()
    if not text:
        return []
    if sentenize is not None:
        return [s.text.strip() for s in sentenize(text) if s.text.strip()]
    return [p.strip() for p in re.split(r'(?<=[.!?…])\s+', text) if p.strip()]


def split_paragraphs(text: str):
    return [p.strip() for p in re.split(r'\n\s*\n|\n', text) if p.strip()]


def tokenize(text: str):
    return re.findall(r"\b[\w’'`-]+\b", text.lower(), flags=re.UNICODE)


def detect_language(text: str):
    if detect is None:
        return "невідомо"
    try:
        lang = detect(text)
        return {"uk": "українська", "ru": "російська", "en": "англійська"}.get(lang, lang)
    except LangDetectException:
        return "невідомо"


def safe_mean(values):
    return sum(values) / len(values) if values else 0.0


def safe_std(values):
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def normalize(value, min_v, max_v):
    if max_v == min_v:
        return 0.0
    x = (value - min_v) / (max_v - min_v)
    return max(0.0, min(1.0, x))


def lexical_diversity(tokens):
    return len(set(tokens)) / len(tokens) if tokens else 0.0


def hapax_ratio(tokens):
    if not tokens:
        return 0.0
    freq = Counter(tokens)
    hapax = sum(1 for c in freq.values() if c == 1)
    return hapax / len(freq) if freq else 0.0


def repetition_score(tokens):
    if not tokens:
        return 0.0
    freq = Counter(tokens)
    repeated = sum(count for count in freq.values() if count > 1)
    return repeated / len(tokens)


def sentence_length_stats(sentences):
    lengths = [len(tokenize(s)) for s in sentences if tokenize(s)]
    if not lengths:
        return {"avg": 0.0, "std": 0.0}
    return {"avg": safe_mean(lengths), "std": safe_std(lengths)}


def paragraph_length_stats(paragraphs):
    lengths = [len(tokenize(p)) for p in paragraphs if tokenize(p)]
    if not lengths:
        return {"avg": 0.0, "std": 0.0}
    return {"avg": safe_mean(lengths), "std": safe_std(lengths)}


def paragraph_uniformity(text):
    paragraphs = split_paragraphs(text)
    if len(paragraphs) < 2:
        return 0.0
    lengths = [len(tokenize(p)) for p in paragraphs]
    avg = safe_mean(lengths)
    if avg == 0:
        return 0.0
    return max(0.0, 1.0 - (safe_std(lengths) / avg))


def common_ngram_repetition(tokens, n=3):
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    freq = Counter(ngrams)
    repeated = sum(count for count in freq.values() if count > 1)
    return repeated / len(ngrams)


def repeated_sentence_start_ratio(sentences, n=2):
    starts = []
    for s in sentences:
        t = tokenize(s)
        if len(t) >= n:
            starts.append(tuple(t[:n]))
    if not starts:
        return 0.0
    freq = Counter(starts)
    repeated = sum(c for c in freq.values() if c > 1)
    return repeated / len(starts)


def transition_smoothness(sentences):
    if len(sentences) < 2:
        return 0.0
    sims = []
    for i in range(len(sentences) - 1):
        a = set(tokenize(sentences[i]))
        b = set(tokenize(sentences[i + 1]))
        if not a or not b:
            continue
        union = len(a | b)
        sims.append(len(a & b) / union if union else 0.0)
    return safe_mean(sims)


def suspicious_phrase_score(text):
    patterns = [
        r"\bотже\b", r"\bтаким чином\b", r"\bслід зазначити\b", r"\bварто зазначити\b",
        r"\bнеобхідно зазначити\b", r"\bу висновку\b", r"\bможна зробити висновок\b",
        r"\bдане дослідження\b", r"\bактуальність теми\b",
        r"\bв заключение\b", r"\bтаким образом\b", r"\bследовательно\b",
        r"\bin conclusion\b", r"\bmoreover\b", r"\bfurthermore\b", r"\bthus\b",
    ]
    text_l = text.lower()
    hits = sum(len(re.findall(p, text_l)) for p in patterns)
    words = len(tokenize(text))
    return (hits / words * 100) if words else 0.0


def avg_word_length(tokens):
    return safe_mean([len(t) for t in tokens]) if tokens else 0.0


def sentence_complexity_ratio(sentences):
    if not sentences:
        return 0.0
    markers = [",", ";", ":", "—", "-", "що", "який", "яка", "коли", "оскільки", "which", "because", "while"]
    count = 0
    for s in sentences:
        sl = s.lower()
        if any(m in sl for m in markers):
            count += 1
    return count / len(sentences)


def function_word_profile(tokens):
    function_words = {
        "і", "й", "та", "але", "або", "що", "як", "щоб", "у", "в", "на", "до", "за",
        "не", "це", "для", "із", "з", "по", "при", "про",
        "and", "but", "or", "that", "as", "if", "for", "to", "in", "on", "at", "by", "with", "from", "of", "the"
    }
    freq = Counter(tokens)
    total = max(len(tokens), 1)
    return {w: freq[w] / total for w in sorted(function_words)}


def style_vector(text):
    sentences = split_sentences(text)
    paragraphs = split_paragraphs(text)
    tokens = tokenize(text)
    s = sentence_length_stats(sentences)
    p = paragraph_length_stats(paragraphs)
    v = {
        "avg_sentence_len": s["avg"],
        "sentence_len_std": s["std"],
        "avg_paragraph_len": p["avg"],
        "paragraph_len_std": p["std"],
        "lexical_diversity": lexical_diversity(tokens),
        "hapax_ratio": hapax_ratio(tokens),
        "repetition_score": repetition_score(tokens),
        "transition_smoothness": transition_smoothness(sentences),
        "avg_word_length": avg_word_length(tokens),
        "complex_sentence_ratio": sentence_complexity_ratio(sentences),
    }
    for k, val in function_word_profile(tokens).items():
        v[f"fw_{k}"] = val
    return v


def average_vectors(vectors):
    if not vectors:
        return {}
    keys = vectors[0].keys()
    return {k: safe_mean([v[k] for v in vectors]) for k in keys}


def vector_distance(v1, v2):
    keys = sorted(set(v1) & set(v2))
    if not keys:
        return None
    diffs = []
    for k in keys:
        a, b = v1[k], v2[k]
        denom = max(abs(a), abs(b), 0.05)
        diffs.append(abs(a - b) / denom)
    return safe_mean(diffs) * 100


def analyze_text(text: str, baseline_texts=None):
    baseline_texts = baseline_texts or []

    sentences = split_sentences(text)
    paragraphs = split_paragraphs(text)
    tokens = tokenize(text)

    s_stats = sentence_length_stats(sentences)
    p_stats = paragraph_length_stats(paragraphs)

    lex_div = lexical_diversity(tokens)
    hapax = hapax_ratio(tokens)
    rep = repetition_score(tokens)
    tri_rep = common_ngram_repetition(tokens, 3)
    smooth = transition_smoothness(sentences)
    para_uni = paragraph_uniformity(text)
    phrase = suspicious_phrase_score(text)
    repeated_starts = repeated_sentence_start_ratio(sentences, 2)
    avg_word = avg_word_length(tokens)
    complex_ratio = sentence_complexity_ratio(sentences)

    reasons = []
    warnings = []
    false_alarm_flags = []

    base_score = 0.0
    base_score += (1.0 - normalize(s_stats["std"], 1.5, 10.0)) * 10
    base_score += normalize(tri_rep, 0.01, 0.08) * 12
    base_score += normalize(repeated_starts, 0.04, 0.18) * 10
    base_score += normalize(phrase, 0.2, 1.8) * 8
    base_score += normalize(smooth, 0.10, 0.32) * 8
    base_score += (1.0 - normalize(lex_div, 0.24, 0.60)) * 8
    base_score += (1.0 - normalize(hapax, 0.28, 0.62)) * 7
    base_score += normalize(rep, 0.30, 0.68) * 6
    base_score += normalize(para_uni, 0.55, 0.90) * 5
    base_score += (1.0 - normalize(p_stats["std"], 8, 90)) * 4

    if len(tokens) < 150:
        warnings.append("Текст короткий, тому оцінка менш надійна.")
        base_score -= 6

    if tri_rep > 0.03:
        reasons.append("Підвищена повторюваність однакових мовних шаблонів.")
    if repeated_starts > 0.10:
        reasons.append("Багато речень починаються схожим чином.")
    if phrase > 0.5:
        reasons.append("Є помітна кількість шаблонних академічних фраз.")
    if smooth > 0.22:
        reasons.append("Переходи між реченнями занадто гладкі та одноманітні.")
    if lex_div < 0.33:
        reasons.append("Словникове різноманіття нижче очікуваного.")
    if hapax < 0.36:
        reasons.append("Мало унікальних слів відносно обсягу тексту.")

    if phrase > 0.4 and tri_rep < 0.02 and repeated_starts < 0.08:
        false_alarm_flags.append("Формальний академічний стиль може завищувати підозру.")
    if complex_ratio > 0.75 and s_stats["std"] > 4.0:
        false_alarm_flags.append("Складний людський стиль може помилково виглядати надто академічним.")
    if avg_word > 5.8 and lex_div >= 0.36:
        false_alarm_flags.append("Насичена лексика більше схожа на людський текст, ніж на шаблон.")

    style_mismatch_score = None
    if baseline_texts:
        target = style_vector(text)
        base_vecs = [style_vector(t) for t in baseline_texts if tokenize(t)]
        if base_vecs:
            dist = vector_distance(target, average_vectors(base_vecs))
            if dist is not None:
                style_mismatch_score = max(0.0, min(100.0, dist))
                style_mismatch_score = max(0.0, (style_mismatch_score - 12) * 1.1)
                if style_mismatch_score > 32:
                    reasons.append("Стиль помітно відрізняється від еталонних робіт студента.")

    if style_mismatch_score is None:
        final_score = min(base_score, 58.0)
    else:
        final_score = 0.62 * base_score + 0.38 * style_mismatch_score

    final_score -= min(len(false_alarm_flags) * 6, 14)
    final_score = max(0.0, min(100.0, final_score))

    if final_score < 25:
        label = "Низька підозра"
    elif final_score < 45:
        label = "Помірна підозра"
    elif final_score < 65:
        label = "Підвищена підозра"
    else:
        label = "Висока підозра"

    confidence = "низька" if len(tokens) < 150 else ("середня" if style_mismatch_score is None else "вища")

    return {
        "language": detect_language(text),
        "paragraphs": len(paragraphs),
        "sentences": len(sentences),
        "words": len(tokens),
        "lexical_diversity": round(lex_div, 4),
        "trigram_repetition": round(tri_rep, 4),
        "template_phrase_score": round(phrase, 4),
        "repeated_sentence_starts": round(repeated_starts, 4),
        "base_risk_score": round(base_score, 2),
        "style_mismatch_score": None if style_mismatch_score is None else round(style_mismatch_score, 2),
        "risk_score": round(final_score, 2),
        "label": label,
        "confidence": confidence,
        "reasons": reasons,
        "warnings": warnings,
        "false_alarm_flags": false_alarm_flags,
    }


def generate_report(r: dict):
    lines = [
        "ПЕРЕВІРКА ТЕКСТУ НА ОЗНАКИ ШІ — STUDIO UI",
        "=" * 72,
        f"Мова: {r['language']}",
        f"Абзаців: {r['paragraphs']}",
        f"Речень: {r['sentences']}",
        f"Слів: {r['words']}",
        f"Різноманітність словника: {r['lexical_diversity']}",
        f"Повтори триграм: {r['trigram_repetition']}",
        f"Шаблонні фрази: {r['template_phrase_score']}",
        f"Однакові початки речень: {r['repeated_sentence_starts']}",
        "-" * 72,
        f"Базова підозра: {r['base_risk_score']} / 100",
        f"Стилістична невідповідність: {'не виконувалась' if r['style_mismatch_score'] is None else str(r['style_mismatch_score']) + ' / 100'}",
        f"Підсумковий індекс: {r['risk_score']} / 100",
        f"Рівень: {r['label']}",
        f"Надійність оцінки: {r['confidence']}",
        "-" * 72,
        "Причини спрацювання:"
    ]
    if r["reasons"]:
        lines += [f"{i}. {x}" for i, x in enumerate(r["reasons"], 1)]
    else:
        lines.append("Явно виражених тригерів не виявлено.")

    if r["false_alarm_flags"]:
        lines.append("-" * 72)
        lines.append("Ознаки можливої хибної тривоги:")
        lines += [f"{i}. {x}" for i, x in enumerate(r["false_alarm_flags"], 1)]

    if r["warnings"]:
        lines.append("-" * 72)
        lines.append("Застереження:")
        lines += [f"{i}. {x}" for i, x in enumerate(r["warnings"], 1)]

    lines += [
        "-" * 72,
        "Це не доказ використання ШІ.",
        "Результат слід трактувати як підставу для ручної перевірки, а не як вирок."
    ]
    return "\n".join(lines)


def read_text_file(path):
    path = str(path)
    if path.lower().endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    if path.lower().endswith(".docx"):
        if Document is None:
            raise RuntimeError("Для .docx встановіть python-docx")
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    raise RuntimeError("Підтримуються лише .txt та .docx файли")

class GlassCard(QFrame):
    def __init__(self, object_name="glassCard", blur=False):
        super().__init__()
        self.setObjectName(object_name)
        effect = QGraphicsDropShadowEffect(self)
        effect.setBlurRadius(36 if blur else 28)
        effect.setOffset(0, 10 if blur else 8)
        effect.setColor(QColor(0, 0, 0, 110))
        self.setGraphicsEffect(effect)


class AccentButton(QPushButton):
    def __init__(self, text, variant="blue"):
        super().__init__(text)
        self.setCursor(Qt.PointingHandCursor)
        self.variant = variant
        self.setProperty("variant", variant)
        self.setMinimumHeight(44)


class TitleBar(QFrame):
    def __init__(self, window, icon):
        super().__init__()
        self.window = window
        self._drag_offset = None
        self.setObjectName("titleBar")
        self.setFixedHeight(52)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 8, 10, 8)
        layout.setSpacing(10)

        icon_label = QLabel()
        icon_label.setObjectName("titleBarIcon")
        icon_label.setPixmap(icon.pixmap(18, 18))
        layout.addWidget(icon_label, 0, Qt.AlignVCenter)

        title_label = QLabel("AI Detector UA")
        title_label.setObjectName("titleBarText")
        layout.addWidget(title_label, 0, Qt.AlignVCenter)
        layout.addStretch()

        self.min_button = self._make_button(chr(0x2212), self.window.showMinimized)
        self.max_button = self._make_button(chr(0x25A1), self.toggle_maximized)
        self.close_button = self._make_button(chr(0x00D7), self.window.close, close_button=True)

        layout.addWidget(self.min_button)
        layout.addWidget(self.max_button)
        layout.addWidget(self.close_button)
        self.sync_window_state()

    def _make_button(self, text, handler, close_button=False):
        button = QPushButton(text)
        button.setCursor(Qt.PointingHandCursor)
        button.setObjectName("titleBarCloseButton" if close_button else "titleBarButton")
        button.setFixedSize(42, 32)
        button.clicked.connect(handler)
        return button

    def toggle_maximized(self):
        if self.window.isMaximized():
            self.window.showNormal()
        else:
            self.window.showMaximized()
        self.sync_window_state()

    def sync_window_state(self):
        self.max_button.setText(chr(0x2750) if self.window.isMaximized() else chr(0x25A1))

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.toggle_maximized()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.window.frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self._drag_offset is not None:
            if self.window.isMaximized():
                return
            self.window.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_offset = None
        super().mouseReleaseEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setWindowTitle("AI Detector UA — Studio")
        self.resize(1460, 920)
        self.baseline_texts = []
        self.current_result = None
        self._app_icon = self._build_app_icon()
        self.setWindowIcon(self._app_icon)
        self._build_ui()
        self._build_shortcuts()
        self._apply_styles()

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("appRoot")
        self.setCentralWidget(root)

        main = QVBoxLayout(root)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        self.title_bar = TitleBar(self, self._app_icon)
        main.addWidget(self.title_bar)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(20, 16, 20, 20)
        content_layout.setSpacing(16)
        main.addWidget(content, 1)

        self.hero = GlassCard("heroCard", blur=True)
        hero_layout = QVBoxLayout(self.hero)
        hero_layout.setContentsMargins(24, 22, 24, 22)
        hero_top = QHBoxLayout()
        hero_top.setSpacing(10)

        badge = QLabel("AI DETECTOR • STUDIO")
        badge.setObjectName("badge")
        hero_top.addWidget(badge, 0, Qt.AlignLeft)
        hero_top.addStretch()
        hero_layout.addLayout(hero_top)

        title = QLabel("Перевірка тексту на ознаки ШІ")
        title.setObjectName("heroTitle")
        subtitle = QLabel(
            "Сучасний інтерфейс, стабільні гарячі клавіші, м’яка оцінка підозри, "
            "еталонні тексти для порівняння стилю та красивий робочий простір."
        )
        subtitle.setWordWrap(True)
        subtitle.setObjectName("heroSubtitle")

        title_row = QHBoxLayout()
        title_row.setSpacing(12)
        title_row.addWidget(title, 0, Qt.AlignVCenter)
        title_row.addStretch()
        hero_layout.addLayout(title_row)
        hero_layout.addWidget(subtitle)
        content_layout.addWidget(self.hero)

        body = QHBoxLayout()
        body.setSpacing(16)
        content_layout.addLayout(body, 1)

        left_col = QVBoxLayout()
        left_col.setSpacing(16)
        body.addLayout(left_col, 7)

        right_col = QVBoxLayout()
        right_col.setSpacing(16)
        body.addLayout(right_col, 5)

        # Toolbar
        toolbar = GlassCard()
        toolbar_layout = QVBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(18, 18, 18, 16)
        toolbar_layout.setSpacing(12)

        toolbar_title = QLabel("Інструменти")
        toolbar_title.setObjectName("sectionTitle")
        toolbar_layout.addWidget(toolbar_title)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self.btn_check = AccentButton("Перевірити", "blue")
        self.btn_paste = AccentButton("Вставити", "green")
        self.btn_open = AccentButton("Відкрити", "violet")
        self.btn_baselines = AccentButton("Еталони", "cyan")
        self.btn_clear_baselines = AccentButton("Очистити еталони", "slate")
        self.btn_clear = AccentButton("Очистити", "amber")
        self.btn_save = AccentButton("Зберегти звіт", "rose")

        for btn in [
            self.btn_check, self.btn_paste, self.btn_open,
            self.btn_baselines, self.btn_clear_baselines,
            self.btn_clear, self.btn_save
        ]:
            btn_row.addWidget(btn)

        toolbar_layout.addLayout(btn_row)

        helper = QLabel("Ctrl+A / Ctrl+C / Ctrl+V / Ctrl+X працюють у текстових полях. Ctrl+S — зберегти звіт.")
        helper.setObjectName("helperText")
        helper.setWordWrap(True)
        toolbar_layout.addWidget(helper)
        left_col.addWidget(toolbar)

        # Editor
        editor_card = GlassCard()
        editor_layout = QVBoxLayout(editor_card)
        editor_layout.setContentsMargins(18, 18, 18, 18)
        editor_layout.setSpacing(12)

        editor_header = QHBoxLayout()
        editor_label = QLabel("Робоче поле")
        editor_label.setObjectName("sectionTitle")
        editor_header.addWidget(editor_label)
        editor_header.addStretch()

        chip = QLabel("Встав текст або відкрий файл")
        chip.setObjectName("chip")
        editor_header.addWidget(chip)
        editor_layout.addLayout(editor_header)

        self.text_input = QTextEdit()
        self.text_input.setAcceptRichText(False)
        self.text_input.setPlaceholderText("Почни вводити текст тут... або відкрий .txt / .docx файл.")
        editor_layout.addWidget(self.text_input)
        left_col.addWidget(editor_card, 1)

        # Score card
        self.score_card = GlassCard("scoreCard", blur=True)
        score_layout = QVBoxLayout(self.score_card)
        score_layout.setContentsMargins(20, 20, 20, 18)
        score_layout.setSpacing(10)

        score_row = QHBoxLayout()
        score_text_col = QVBoxLayout()
        self.score_label = QLabel("Підозра: —")
        self.score_label.setObjectName("scoreLabel")
        self.score_hint = QLabel("Після перевірки тут з’явиться короткий висновок.")
        self.score_hint.setWordWrap(True)
        self.score_hint.setObjectName("scoreHint")
        score_text_col.addWidget(self.score_label)
        score_text_col.addWidget(self.score_hint)
        score_row.addLayout(score_text_col, 1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        score_layout.addLayout(score_row)
        score_layout.addWidget(self.progress)
        right_col.addWidget(self.score_card)

        # Baseline small card
        baseline_card = GlassCard()
        baseline_layout = QVBoxLayout(baseline_card)
        baseline_layout.setContentsMargins(18, 16, 18, 16)
        baseline_title = QLabel("Еталонні тексти")
        baseline_title.setObjectName("sectionTitle")
        self.baseline_label = QLabel("Не додано")
        self.baseline_label.setObjectName("baselineText")
        baseline_layout.addWidget(baseline_title)
        baseline_layout.addWidget(self.baseline_label)
        right_col.addWidget(baseline_card)

        # Metrics card
        metrics_card = GlassCard()
        metrics_layout = QVBoxLayout(metrics_card)
        metrics_layout.setContentsMargins(18, 16, 18, 16)
        metrics_title = QLabel("Ключові показники")
        metrics_title.setObjectName("sectionTitle")
        self.metrics_label = QLabel("Показники ще не розраховані.")
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setObjectName("bodyText")
        metrics_layout.addWidget(metrics_title)
        metrics_layout.addWidget(self.metrics_label)
        right_col.addWidget(metrics_card)

        # Report card
        report_card = GlassCard()
        report_layout = QVBoxLayout(report_card)
        report_layout.setContentsMargins(18, 16, 18, 18)
        report_title_row = QHBoxLayout()
        report_title = QLabel("Детальний звіт")
        report_title.setObjectName("sectionTitle")
        report_title_row.addWidget(report_title)
        report_title_row.addStretch()
        report_pill = QLabel("Read-only")
        report_pill.setObjectName("reportPill")
        report_title_row.addWidget(report_pill)
        report_layout.addLayout(report_title_row)

        self.report_box = QTextEdit()
        self.report_box.setReadOnly(True)
        report_layout.addWidget(self.report_box)
        right_col.addWidget(report_card, 1)

        # Signals
        self.btn_check.clicked.connect(self.run_analysis)
        self.btn_paste.clicked.connect(self.paste_from_clipboard)
        self.btn_open.clicked.connect(self.open_file)
        self.btn_baselines.clicked.connect(self.load_baselines)
        self.btn_clear_baselines.clicked.connect(self.clear_baselines)
        self.btn_clear.clicked.connect(self.clear_all)
        self.btn_save.clicked.connect(self.save_report)

        self.text_input.setFocus()

    def _build_app_icon(self):
        pixmap = QPixmap(64, 64)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        gradient = QLinearGradient(8, 8, 56, 56)
        gradient.setColorAt(0.0, QColor("#38bdf8"))
        gradient.setColorAt(0.55, QColor("#2563eb"))
        gradient.setColorAt(1.0, QColor("#0f172a"))
        painter.setBrush(QBrush(gradient))
        painter.setPen(QPen(QColor(255, 255, 255, 36), 1))
        painter.drawRoundedRect(8, 8, 48, 48, 14, 14)

        painter.setBrush(QColor("#ecfeff"))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(18, 17, 20, 24, 5, 5)

        painter.setBrush(QColor("#7dd3fc"))
        painter.drawRoundedRect(22, 22, 12, 3, 1.5, 1.5)
        painter.drawRoundedRect(22, 28, 18, 3, 1.5, 1.5)
        painter.drawRoundedRect(22, 34, 14, 3, 1.5, 1.5)

        painter.setBrush(QColor("#22c55e"))
        painter.drawEllipse(41, 17, 7, 7)
        painter.end()
        return QIcon(pixmap)

    def _build_shortcuts(self):
        save_action = QAction(self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.save_report)
        self.addAction(save_action)

        select_action = QAction(self)
        select_action.setShortcut(QKeySequence.SelectAll)
        select_action.triggered.connect(self.select_all_current)
        self.addAction(select_action)

    def _apply_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background: #070b14;
            }

            QWidget#appRoot {
                background:
                    qradialgradient(cx:0.18, cy:0.08, radius:1.15,
                    fx:0.18, fy:0.08,
                    stop:0 #172554,
                    stop:0.35 #0f172a,
                    stop:1 #070b14);
                color: #e5eefc;
                font-family: "Segoe UI", "Inter", Arial, sans-serif;
                font-size: 14px;
            }

            QFrame#titleBar {
                background:
                    qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(15, 23, 42, 0.98),
                    stop:0.55 rgba(17, 34, 76, 0.98),
                    stop:1 rgba(9, 14, 27, 0.98));
                border-bottom: 1px solid rgba(255,255,255,0.08);
            }

            QLabel#titleBarText {
                color: #dbeafe;
                font-size: 14px;
                font-weight: 600;
                background: transparent;
            }

            QLabel#titleBarIcon {
                background: transparent;
                padding-left: 4px;
            }

            QPushButton#titleBarButton, QPushButton#titleBarCloseButton {
                background: transparent;
                border: none;
                border-radius: 10px;
                color: #d7e6fb;
                font-size: 15px;
                font-weight: 700;
                padding: 0px;
            }

            QPushButton#titleBarButton:hover {
                background: rgba(255,255,255,0.08);
            }

            QPushButton#titleBarCloseButton:hover {
                background: rgba(244, 63, 94, 0.92);
                color: #ffffff;
            }

            QFrame#heroCard {
                background:
                    qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(14, 35, 84, 0.96),
                    stop:0.55 rgba(10, 19, 48, 0.94),
                    stop:1 rgba(9, 14, 27, 0.96));
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 28px;
            }

            QFrame#glassCard, QFrame#scoreCard {
                background:
                    qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(16, 24, 44, 0.94),
                    stop:1 rgba(9, 14, 27, 0.96));
                border: 1px solid rgba(255,255,255,0.06);
                border-radius: 24px;
            }

            QLabel#badge {
                background: rgba(37, 99, 235, 0.18);
                color: #93c5fd;
                border: 1px solid rgba(96, 165, 250, 0.25);
                border-radius: 999px;
                padding: 6px 12px;
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 0.7px;
            }

            QLabel#heroTitle {
                font-size: 38px;
                font-weight: 800;
                color: #ffffff;
                margin-top: 6px;
                background: transparent;
            }

            QLabel#heroSubtitle {
                font-size: 15px;
                line-height: 1.5;
                color: #b8c7df;
                margin-top: 4px;
                background: transparent;
            }

            QLabel#sectionTitle {
                font-size: 20px;
                font-weight: 800;
                color: #f8fbff;
                background: transparent;
            }

            QLabel#helperText {
                color: #96a8c6;
                font-size: 13px;
                background: transparent;
            }

            QLabel#chip, QLabel#reportPill {
                background: transparent;
                color: #8fb5ff;
                border: none;
                padding: 0px;
                font-size: 12px;
                font-weight: 700;
            }

            QLabel#scoreLabel {
                font-size: 28px;
                font-weight: 800;
                color: #f8fbff;
                background: transparent;
            }

            QLabel#scoreHint {
                color: #c3d1e8;
                font-size: 14px;
                background: transparent;
            }

            QLabel#baselineText, QLabel#bodyText {
                color: #d0ddf2;
                font-size: 14px;
                line-height: 1.45;
                background: transparent;
            }

            QPushButton {
                border: none;
                border-radius: 14px;
                padding: 11px 15px;
                color: #ffffff;
                font-weight: 700;
                font-size: 14px;
            }

            QPushButton[variant="blue"] {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #2563eb, stop:1 #1d4ed8);
            }
            QPushButton[variant="green"] {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #10b981, stop:1 #059669);
            }
            QPushButton[variant="violet"] {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #8b5cf6, stop:1 #7c3aed);
            }
            QPushButton[variant="cyan"] {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #06b6d4, stop:1 #0891b2);
            }
            QPushButton[variant="slate"] {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #475569, stop:1 #334155);
            }
            QPushButton[variant="amber"] {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #f59e0b, stop:1 #d97706);
            }
            QPushButton[variant="rose"] {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #f43f5e, stop:1 #e11d48);
            }

            QPushButton:hover {
                border: 1px solid rgba(255,255,255,0.12);
            }

            QTextEdit {
                background:
                    qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(2, 10, 28, 0.98),
                    stop:1 rgba(4, 12, 24, 0.98));
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 18px;
                padding: 14px;
                color: #eef4ff;
                selection-background-color: #2563eb;
                font-size: 15px;
            }

            QTextEdit:focus {
                border: 1px solid rgba(96, 165, 250, 0.55);
            }

            QProgressBar {
                background: rgba(148, 163, 184, 0.18);
                border: 1px solid rgba(255,255,255,0.05);
                border-radius: 12px;
                min-height: 22px;
                max-height: 22px;
            }

            QProgressBar::chunk {
                border-radius: 12px;
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #22d3ee, stop:1 #2563eb);
            }

            QScrollBar:vertical {
                background: transparent;
                width: 12px;
                margin: 6px 4px 6px 4px;
            }

            QScrollBar::handle:vertical {
                background: rgba(148, 163, 184, 0.36);
                min-height: 30px;
                border-radius: 6px;
            }

            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
                background: none;
                border: none;
            }

            QScrollBar:horizontal {
                background: transparent;
                height: 12px;
                margin: 4px 6px 4px 6px;
            }

            QScrollBar::handle:horizontal {
                background: rgba(148, 163, 184, 0.36);
                min-width: 30px;
                border-radius: 6px;
            }

            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
                background: none;
                border: none;
            }
        """)

    def current_text_edit(self):
        fw = QApplication.focusWidget()
        if isinstance(fw, QTextEdit):
            return fw
        return self.text_input

    def select_all_current(self):
        self.current_text_edit().selectAll()

    def update_score_ui(self, r):
        score = int(round(r["risk_score"]))
        self.progress.setValue(score)

        if score < 25:
            color = "#10b981"
            text = "Сильних підстав підозрювати машинну генерацію небагато."
        elif score < 45:
            color = "#f59e0b"
            text = "Є окремі підозрілі сигнали, але вони ще не виглядають переконливо."
        elif score < 65:
            color = "#fb923c"
            text = "Є помітний набір ознак, характерних для генерації або сильного редагування."
        else:
            color = "#f43f5e"
            text = "Сигналів досить багато, але результат усе одно не є доказом."

        if r["false_alarm_flags"]:
            text += " Також є ознаки можливої хибної тривоги."

        self.score_label.setText(f"Підозра: {r['risk_score']} / 100 — {r['label']}")
        self.score_label.setStyleSheet(f"font-size: 28px; font-weight: 800; color: {color};")
        self.score_hint.setText(text)

        self.progress.setStyleSheet(f"""
            QProgressBar {{
                background: rgba(148, 163, 184, 0.18);
                border: 1px solid rgba(255,255,255,0.05);
                border-radius: 12px;
                min-height: 22px;
                max-height: 22px;
            }}
            QProgressBar::chunk {{
                border-radius: 12px;
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {color}, stop:1 #2563eb);
            }}
        """)

        mismatch = "не виконувалась" if r["style_mismatch_score"] is None else f"{r['style_mismatch_score']} / 100"
        self.metrics_label.setText(
            f"Мова: {r['language']}\n"
            f"Слів: {r['words']}\n"
            f"Речень: {r['sentences']}\n"
            f"Різноманітність словника: {r['lexical_diversity']}\n"
            f"Повтори триграм: {r['trigram_repetition']}\n"
            f"Шаблонні фрази: {r['template_phrase_score']}\n"
            f"Стиль vs еталони: {mismatch}\n"
            f"Надійність оцінки: {r['confidence']}"
        )

    def paste_from_clipboard(self):
        self.current_text_edit().paste()
        self.text_input.setFocus()

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Відкрити текст",
            "",
            "Підтримувані файли (*.txt *.docx);;Текстові файли (*.txt);;Документи Word (*.docx)"
        )
        if not path:
            self.text_input.setFocus()
            return
        try:
            self.text_input.setPlainText(read_text_file(path))
        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Не вдалося відкрити файл:\n{e}")
        self.text_input.setFocus()

    def load_baselines(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Додати еталонні тексти",
            "",
            "Підтримувані файли (*.txt *.docx);;Текстові файли (*.txt);;Документи Word (*.docx)"
        )
        if not paths:
            self.text_input.setFocus()
            return
        added = 0
        for path in paths:
            try:
                text = read_text_file(path)
                if tokenize(text):
                    self.baseline_texts.append(text)
                    added += 1
            except Exception:
                pass
        self.baseline_label.setText(f"Додано {len(self.baseline_texts)} файл(ів)")
        if added:
            QMessageBox.information(self, "Готово", f"Додано {added} еталонних файл(ів).")
        self.text_input.setFocus()

    def clear_baselines(self):
        self.baseline_texts = []
        self.baseline_label.setText("Не додано")
        self.text_input.setFocus()

    def clear_all(self):
        self.text_input.clear()
        self.report_box.clear()
        self.current_result = None
        self.progress.setValue(0)
        self.score_label.setText("Підозра: —")
        self.score_label.setStyleSheet("font-size: 28px; font-weight: 800; color: #f8fbff;")
        self.score_hint.setText("Після перевірки тут з’явиться короткий висновок.")
        self.metrics_label.setText("Показники ще не розраховані.")
        self.text_input.setFocus()

    def save_report(self):
        if not self.current_result:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Зберегти звіт", "", "Текстовий файл (*.txt)")
        if not path:
            self.text_input.setFocus()
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(generate_report(self.current_result))
            QMessageBox.information(self, "Готово", "Звіт збережено.")
        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Не вдалося зберегти звіт:\n{e}")
        self.text_input.setFocus()

    def run_analysis(self):
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Увага", "Вставте текст для перевірки.")
            self.text_input.setFocus()
            return
        self.current_result = analyze_text(text, baseline_texts=self.baseline_texts)
        self.update_score_ui(self.current_result)
        self.report_box.setPlainText(generate_report(self.current_result))
        self.text_input.setFocus()


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()