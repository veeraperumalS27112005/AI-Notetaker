import customtkinter as ctk
import sounddevice as sd
from scipy.io.wavfile import write
import scipy.signal
import whisper
import warnings
import numpy as np
import threading
import queue
import os
import json
import re

from google import genai

warnings.filterwarnings("ignore")

# ─── Configuration ───────────────────────────────────────────────────────────
RECORD_SR      = 48000
AUDIO_PATH     = r"C:\Users\veera\Downloads\system_audio.wav"
SUMMARY_PATH   = r"C:\Users\veera\Downloads\ai_learning_notes.txt"
API_KEYS = [
    "AIzaSyBvdGGVvtBpWNfrIhxqbMTORC4s8qiPtgo",   # Key 1 (current)
    "AIzaSyAl6YEfgXb9misYJwEBYcCCrzso0MqUAoc"   # Key 2 (previous)
    # Add more keys here as needed: "AIza...",
]
_key_index = 0   # auto-rotates when quota is hit

# ─── Auto-detect best input device ───────────────────────────────────────────
def _find_input_device() -> tuple:
    """Find the best system-audio capture device.
    Priority: CABLE Output > Stereo Mix > default input.
    Returns (device_index, device_name)."""
    devices = sd.query_devices()
    # Audio routing in this setup:
    #   System audio -> CABLE Input (Windows output device)
    #   -> CABLE Output captures it as an input (loopback)
    # Stereo Mix only captures Realtek output, which is silent when CABLE Input is the output.
    # Priority: CABLE Output > Stereo Mix > other loopbacks
    preferred = ["cable output", "stereo mix", "what u hear", "loopback"]
    for priority_name in preferred:
        for i, dev in enumerate(devices):
            name = dev['name'].lower()
            if priority_name in name and dev['max_input_channels'] > 0:
                print(f"[audio] Auto-selected device {i}: {dev['name']}")
                return i, dev['name']
    # Fallback: default input device
    default = sd.default.device[0]
    name = devices[default]['name'] if default >= 0 else "Default"
    print(f"[audio] Using default input device {default}: {name}")
    return default, name

DEVICE_INDEX, DEVICE_NAME = _find_input_device()

# ─── Theme colors ─────────────────────────────────────────────────────────────
BG        = "#f8f9fc"
PANEL     = "#ffffff"
CARD      = "#f0f2f5"
ACCENT    = "#4f46e5"
ACCENT2   = "#059669"
ACCENT3   = "#dc2626"
ACCENT4   = "#d97706"
TXT       = "#1e293b"
TXT_DIM   = "#64748b"
BTN_H     = "#4338ca"

LANGS = ["Tamil"]

# Language name -> ISO codes
_LANG_CODES = {
    "Tamil":    "ta",
    "Hindi":    "hi",
    "Spanish":  "es",
    "French":   "fr",
    "German":   "de",
    "Japanese": "ja",
    "Arabic":   "ar",
}

# ─── Offline / Free Translation Engine ───────────────────────────────────────
_argos_cache: dict = {}   # cache Argos translation objects

def _argos_translate(text: str, to_code: str) -> str:
    """Try Argos Translate (100% offline for supported languages)."""
    import argostranslate.translate
    cache_key = f"en_{to_code}"
    if cache_key not in _argos_cache:
        langs     = argostranslate.translate.get_installed_languages()
        from_lang = next((l for l in langs if l.code == "en"), None)
        to_lang   = next((l for l in langs if l.code == to_code), None)
        if not from_lang or not to_lang:
            raise ValueError(f"Argos package en->{to_code} not installed")
        _argos_cache[cache_key] = from_lang.get_translation(to_lang)
    t = _argos_cache[cache_key]
    paras = text.split("\n")
    return "\n".join(t.translate(p) if p.strip() else "" for p in paras)

def _deep_translate(text: str, to_code: str) -> str:
    """Translate via Google Translate (free, no API key, internet required).
    Supports Tamil and all languages missing from Argos."""
    from deep_translator import GoogleTranslator
    # Split into chunks ≤4500 chars (GoogleTranslator limit)
    chunks, current = [], ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > 4500:
            if current:
                chunks.append(current)
            current = line
        else:
            current = (current + "\n" + line) if current else line
    if current:
        chunks.append(current)

    translated_chunks = []
    for chunk in chunks:
        if chunk.strip():
            result = GoogleTranslator(source="en", target=to_code).translate(chunk)
            translated_chunks.append(result or chunk)
        else:
            translated_chunks.append(chunk)
    return "\n".join(translated_chunks)

def _mymemory_translate(text: str, to_code: str) -> str:
    """MyMemory free translation fallback (no API key, internet required)."""
    from deep_translator import MyMemoryTranslator
    # MyMemory limit is ~500 chars per call
    sentences = text.split(". ")
    translated = []
    buffer = ""
    for sent in sentences:
        if len(buffer) + len(sent) < 450:
            buffer = (buffer + ". " + sent).strip() if buffer else sent
        else:
            if buffer:
                r = MyMemoryTranslator(source="en-US", target=f"{to_code}-{to_code.upper()}").translate(buffer)
                translated.append(r or buffer)
            buffer = sent
    if buffer:
        r = MyMemoryTranslator(source="en-US", target=f"{to_code}-{to_code.upper()}").translate(buffer)
        translated.append(r or buffer)
    return ". ".join(translated)

def _offline_translate(text: str, target_lang_name: str) -> str:
    """Multi-layer offline/free translation:
       Layer 1: Argos Translate (100% offline, no internet)
       Layer 2: Google Translate via deep-translator (free, no API key, needs internet)
       Layer 3: MyMemory (free fallback)
    """
    to_code = _LANG_CODES.get(target_lang_name)
    if not to_code:
        return f"Language '{target_lang_name}' not supported."

    errors = []

    # Layer 1: Argos (offline — works for hi/es/fr/de/ja/ar, NOT ta)
    try:
        result = _argos_translate(text, to_code)
        print(f"[translate] Used Argos offline for {target_lang_name}")
        return result
    except Exception as e:
        errors.append(f"Argos: {e}")
        print(f"[translate] Argos failed ({e}), trying Google Translate...")

    # Layer 2: Google Translate free (supports Tamil + all others)
    try:
        result = _deep_translate(text, to_code)
        print(f"[translate] Used Google Translate (free) for {target_lang_name}")
        return result
    except Exception as e:
        errors.append(f"Google: {e}")
        print(f"[translate] Google Translate failed ({e}), trying MyMemory...")

    # Layer 3: MyMemory free fallback
    try:
        result = _mymemory_translate(text, to_code)
        print(f"[translate] Used MyMemory for {target_lang_name}")
        return result
    except Exception as e:
        errors.append(f"MyMemory: {e}")

    return (
        f"Translation to {target_lang_name} failed (all engines tried).\n"
        f"Errors: {' | '.join(errors)}\n\n"
        f"Original text:\n{text}"
    )


# ──────────────────────────────────────────────────────────────────────────────
def make_client():
    return genai.Client(api_key=API_KEYS[_key_index].strip())

# Try multiple models — each has its OWN separate quota bucket
MODELS = [
    "gemini-2.0-flash",        # Primary — 1500 req/day free
    "gemini-2.0-flash-lite",   # Fallback — higher quota, same API
]

def gemini(prompt: str) -> str:
    global _key_index
    # Try every combination of model + key until one succeeds
    for model in MODELS:
        for key in API_KEYS:
            try:
                client = genai.Client(api_key=key.strip())
                resp = client.models.generate_content(model=model, contents=prompt)
                return resp.text.strip()
            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    continue   # exhausted, try next
                elif "401" in err or "API_KEY" in err:
                    continue   # bad key, skip
                # Any other error — raise so caller can fallback
                raise
    # All keys/models exhausted
    raise RuntimeError("QUOTA_EXHAUSTED")


# ─── Local Offline Fallbacks (pure Python, no API needed) ────────────────────
_STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "is","was","are","were","be","been","being","have","has","had","do","does",
    "did","will","would","could","should","may","might","shall","can","this",
    "that","these","those","it","its","we","you","he","she","they","i","me",
    "my","your","our","their","his","her","so","if","as","by","from","into",
    "also","not","no","just","very","then","than","about","which","who","what",
    "how","when","where","all","more","some","one","two","many","much","any",
    "like","get","use","used","using","make","see","know","go","say","said",
    "um","uh","okay","right","yeah","well","okay","now","here","there",
}

def _word_freq(text: str) -> dict:
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    freq  = {}
    for w in words:
        if w not in _STOPWORDS:
            freq[w] = freq.get(w, 0) + 1
    return freq

def _tfidf_keywords(text: str, top_n: int = 15):
    """TF-IDF keyword extraction — uses sklearn if available, else word-freq."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) < 2:
            sentences = [text]
        vec  = TfidfVectorizer(stop_words='english', max_features=top_n,
                               ngram_range=(1, 2))
        vec.fit_transform(sentences)
        names  = vec.get_feature_names_out()
        scores = vec.idf_
        pairs  = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)
        return [(w, round(s, 2)) for w, s in pairs[:top_n]]
    except ImportError:
        freq = _word_freq(text)
        top  = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return top

def _sentences(text: str):
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if len(s.split()) > 4]

def local_summary(text: str, mode: str = "short") -> str:
    """Extractive summariser — word-frequency sentence scoring."""
    sents = _sentences(text)
    if not sents:
        return text[:500]
    freq = _word_freq(text)
    def score(s):
        return sum(freq.get(w, 0) for w in re.findall(r'\b[a-zA-Z]{3,}\b', s.lower()))
    ranked = sorted(sents, key=score, reverse=True)

    def ordered_top(n):
        top = set(ranked[:n])
        return [s for s in sents if s in top]

    if mode == "short":
        return " ".join(ordered_top(3))
    elif mode == "bullets":
        return "📝 Bullet Points\n\n" + "\n".join(f"• {s}" for s in ordered_top(7))
    elif mode == "detailed":
        return "📝 Detailed Summary\n\n" + " ".join(ordered_top(12))
    elif mode == "exam":
        kws = _tfidf_keywords(text, 8)
        qa  = []
        for i, (kw, _) in enumerate(kws, 1):
            hits = [s for s in sents if kw.split()[0] in s.lower()]
            ans  = hits[0] if hits else "See transcript for details."
            qa.append(f"Q{i}: What is '{kw}'?\nA{i}: {ans}")
        return "📝 Exam Q&A\n\n" + "\n\n".join(qa)
    elif mode == "eli5":
        top = ordered_top(2)
        simplified = " ".join(top)
        return f"📝 Simple Explanation\n\n{simplified}"
    return " ".join(ordered_top(3))

def local_keywords(text: str) -> str:
    kws     = _tfidf_keywords(text, 20)
    topics  = kws[:5]
    rest    = kws[5:]
    freq    = _word_freq(text)
    lines   = ["📝 Keyword Extraction\n"]
    lines  += ["\n🔹 MAIN TOPICS (TF-IDF top terms):"]
    lines  += [f"  {i+1}. {w.capitalize()}" for i,(w,_) in enumerate(topics)]
    lines  += ["\n🔑 KEY KEYWORDS:"]
    lines  += ["  " + "  |  ".join(w for w,_ in rest[:10])]
    lines  += ["\n📊 WORD FREQUENCY (top 5):"]
    top_f   = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:5]
    lines  += [f"  '{w}' mentioned {c} times" for w,c in top_f]
    return "\n".join(lines)

def local_sentiment(text: str) -> str:
    pos_words = {"good","great","excellent","amazing","wonderful","helpful",
                 "important","useful","positive","success","best","clear",
                 "effective","interesting","easy","learn","understand","benefit",
                 "simple","correct","right","perfect","better","powerful"}
    neg_words = {"bad","poor","difficult","hard","problem","issue","wrong",
                 "fail","error","negative","confusing","unclear","lack","never",
                 "complex","complicated","impossible","mistake","broken"}
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    pos   = sum(1 for w in words if w in pos_words)
    neg   = sum(1 for w in words if w in neg_words)
    total = len(words)
    ratio = pos / (neg + 1)
    if ratio > 1.5:
        label, tone, intent = "Positive 😊", "Enthusiastic / Encouraging", "Educational / Motivational"
    elif ratio < 0.7:
        label, tone, intent = "Negative 😟", "Critical / Serious", "Analytical / Critical"
    else:
        label, tone, intent = "Neutral 😐", "Calm / Informative", "Informative / Lecture"
    content_type = "Lecture / Tutorial" if total > 100 else "Short Clip / Conversation"
    return (
        f"📝 Sentiment Analysis\n\n"
        f"• Overall Sentiment  : {label}\n"
        f"• Emotional Tone     : {tone}\n"
        f"• Speaker Intention  : {intent}\n"
        f"• Content Type       : {content_type}\n"
        f"• Positive signals   : {pos} word(s)\n"
        f"• Negative signals   : {neg} word(s)\n"
        f"• Total words        : {total}"
    )

def local_flashcards(text: str) -> list:
    """Rule-based flashcard generation — detects 'X is a Y', 'X refers to Y' etc."""
    patterns = [
        r'([A-Z][\w\s]{1,30})\s+is\s+a\s+([^.!?]{10,120}[.!?])',
        r'([A-Z][\w\s]{1,30})\s+refers\s+to\s+([^.!?]{10,120}[.!?])',
        r'([A-Z][\w\s]{1,30})\s+is\s+defined\s+as\s+([^.!?]{10,120}[.!?])',
        r'([A-Z][\w\s]{1,30})\s+means\s+([^.!?]{10,120}[.!?])',
        r'([A-Z][\w\s]{1,30})\s+stands\s+for\s+([^.!?]{10,120}[.!?])',
    ]
    cards = []
    seen  = set()
    for pat in patterns:
        for m in re.finditer(pat, text):
            term = m.group(1).strip()
            defn = m.group(2).strip()
            if term.lower() not in seen and len(term) > 2:
                seen.add(term.lower())
                cards.append({"q": f"What is {term}?", "a": defn})
    # Fallback: Q from keyword + sentence
    if len(cards) < 4:
        sents = _sentences(text)
        kws   = _tfidf_keywords(text, 10)
        for kw, _ in kws:
            if kw.lower() in seen:
                continue
            hits = [s for s in sents if kw.split()[0] in s.lower()]
            if hits:
                seen.add(kw.lower())
                cards.append({"q": f"What is '{kw}'?", "a": hits[0]})
            if len(cards) >= 8:
                break
    return cards[:10]

def local_chat(question: str, transcript: str) -> str:
    """Simple offline chat — searches transcript for relevant sentences."""
    q_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', question.lower())) - _STOPWORDS
    sents   = _sentences(transcript)
    scored  = []
    for s in sents:
        s_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', s.lower()))
        overlap  = len(q_words & s_words)
        if overlap > 0:
            scored.append((overlap, s))
    scored.sort(reverse=True)
    if scored:
        top = [s for _, s in scored[:3]]
        return (
            "💬 Answer:\n\n"
            + " ".join(top)
        )
    return "💬 I couldn't find a direct match in the transcript. Try rephrasing your question."



# ──────────────────────────────────────────────────────────────────────────────
class ScrollableChat(ctk.CTkScrollableFrame):
    def __init__(self, master, **kw):
        super().__init__(master, fg_color=PANEL, **kw)
        self._msgs = []

    def add_message(self, role: str, text: str):
        color  = ACCENT if role == "You" else ACCENT2
        bubble = CARD
        frame  = ctk.CTkFrame(self, fg_color=bubble, corner_radius=10)
        frame.pack(fill="x", padx=10, pady=4)
        ctk.CTkLabel(frame, text=role, font=("Inter", 10, "bold"),
                     text_color=color).pack(anchor="w", padx=10, pady=(6, 0))
        ctk.CTkLabel(frame, text=text, font=("Inter", 12),
                     text_color=TXT, wraplength=520, justify="left").pack(
                         anchor="w", padx=10, pady=(2, 8))

    def clear(self):
        for w in self.winfo_children():
            w.destroy()


class FlipCard(ctk.CTkFrame):
    """A single flashcard that shows Q on face, A on back."""
    def __init__(self, master, q: str, a: str, idx: int, **kw):
        super().__init__(master, fg_color=CARD, corner_radius=14, **kw)
        self._q, self._a = q, a
        self._showing_q  = True

        ctk.CTkLabel(self, text=f"Card {idx}", font=("Inter", 10),
                     text_color=TXT_DIM).pack(anchor="w", padx=14, pady=(10,0))
        self.content_lbl = ctk.CTkLabel(
            self, text=q, font=("Inter", 13), text_color=TXT,
            wraplength=480, justify="left")
        self.content_lbl.pack(padx=14, pady=(6, 4), anchor="w")

        self.tag_lbl = ctk.CTkLabel(self, text="❓ Question",
                                     font=("Inter", 10, "bold"),
                                     text_color=ACCENT4)
        self.tag_lbl.pack(anchor="e", padx=14, pady=(0, 10))

        btn = ctk.CTkButton(self, text="Flip ↩", width=80, height=26,
                            fg_color=ACCENT, hover_color=BTN_H,
                            font=("Inter", 11, "bold"),
                            command=self.flip)
        btn.pack(anchor="e", padx=14, pady=(0, 12))

    def flip(self):
        if self._showing_q:
            self.content_lbl.configure(text=self._a)
            self.tag_lbl.configure(text="✅ Answer", text_color=ACCENT2)
        else:
            self.content_lbl.configure(text=self._q)
            self.tag_lbl.configure(text="❓ Question", text_color=ACCENT4)
        self._showing_q = not self._showing_q


# ──────────────────────────────────────────────────────────────────────────────
class AILearningApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        self.title("🎓 AI Video Learning Assistant")
        self.geometry("900x680")
        self.minsize(800, 600)
        self.configure(fg_color=BG)

        self._transcript   = ""
        self._segments     = []
        self._summary      = ""
        self._chat_history = []  # list of {role, text}
        self._is_recording = False
        self._audio_queue  = queue.Queue()
        self._audio_data   = []
        self._stream       = None

        self._build_ui()

    # ── UI builder ────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Top banner
        banner = ctk.CTkFrame(self, fg_color=PANEL, height=56, corner_radius=0)
        banner.pack(fill="x")
        ctk.CTkLabel(banner, text="🎓  AI Video Learning Assistant",
                     font=("Inter", 18, "bold"), text_color=TXT).pack(
                         side="left", padx=20, pady=12)
        self.status_lbl = ctk.CTkLabel(banner, text="● Ready",
                                        font=("Inter", 12), text_color=ACCENT2)
        self.status_lbl.pack(side="right", padx=20)

        # Body: sidebar + content
        body = ctk.CTkFrame(self, fg_color=BG)
        body.pack(fill="both", expand=True)

        self._sidebar = ctk.CTkFrame(body, fg_color=PANEL, width=160,
                                      corner_radius=0)
        self._sidebar.pack(side="left", fill="y")
        self._sidebar.pack_propagate(False)

        self._content = ctk.CTkFrame(body, fg_color=BG)
        self._content.pack(side="left", fill="both", expand=True)

        self._pages = {}
        self._active_page = None

        self._pages["record"]     = self._build_record_page(self._content)
        self._pages["results"]    = self._build_results_page(self._content)
        self._pages["chat"]       = self._build_chat_page(self._content)
        self._pages["flashcards"] = self._build_flashcards_page(self._content)
        self._pages["notes"]      = self._build_notes_page(self._content)

        nav_items = [
            ("🎙️ Record",     "record"),
            ("📊 Results",    "results"),
            ("🤖 Chat",       "chat"),
            ("🃏 Flashcards", "flashcards"),
            ("📓 Notes",      "notes"),
        ]
        self._nav_btns = {}
        ctk.CTkLabel(self._sidebar, text="MENU", font=("Inter", 10, "bold"),
                     text_color=TXT_DIM).pack(pady=(20, 8))
        for label, key in nav_items:
            btn = ctk.CTkButton(
                self._sidebar, text=label, font=("Inter", 13),
                fg_color="transparent", hover_color=CARD,
                text_color=TXT, anchor="w", height=38,
                command=lambda k=key: self._show_page(k))
            btn.pack(fill="x", padx=8, pady=2)
            self._nav_btns[key] = btn

        self._show_page("record")

    def _show_page(self, key: str):
        if self._active_page:
            self._pages[self._active_page].pack_forget()
            self._nav_btns[self._active_page].configure(
                fg_color="transparent", font=("Inter", 13))
        self._pages[key].pack(fill="both", expand=True, padx=16, pady=16)
        self._nav_btns[key].configure(fg_color=ACCENT, font=("Inter", 13, "bold"))
        self._active_page = key

    # ── Record Page ───────────────────────────────────────────────────────────
    def _build_record_page(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=BG)

        ctk.CTkLabel(frame, text="🎙️  Record Audio",
                     font=("Inter", 20, "bold"), text_color=TXT).pack(
                         anchor="w", pady=(0, 4))
        ctk.CTkLabel(frame, text="Record system audio — AI transcribes and analyses your content.",
                     font=("Inter", 12), text_color=TXT_DIM).pack(anchor="w")

        card = ctk.CTkFrame(frame, fg_color=CARD, corner_radius=14)
        card.pack(fill="x", pady=20)

        self.rec_btn = ctk.CTkButton(
            card, text="⏺  Start Recording", font=("Inter", 15, "bold"),
            fg_color=ACCENT, hover_color=BTN_H, text_color=TXT,
            height=46, corner_radius=10, command=self.toggle_recording)
        self.rec_btn.pack(padx=20, pady=20, fill="x")

        self.rec_status = ctk.CTkLabel(card, text="Idle — press to begin",
                                        font=("Inter", 12), text_color=TXT_DIM)
        self.rec_status.pack(pady=(0, 12))

        self.progress = ctk.CTkProgressBar(card, fg_color=PANEL,
                                            progress_color=ACCENT)
        self.progress.pack(fill="x", padx=20, pady=(0, 20))
        self.progress.set(0)

        # Tips
        tips_frame = ctk.CTkFrame(frame, fg_color=CARD, corner_radius=14)
        tips_frame.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(tips_frame, text="💡  How to use",
                     font=("Inter", 13, "bold"), text_color=ACCENT4).pack(
                         anchor="w", padx=16, pady=(12, 4))
        tips = [
            "1. Play any video / lecture on your PC",
            "2. Click 'Start Recording' — it captures system audio",
            "3. Click 'Stop' when done — transcription starts automatically",
            "4. Switch to Results, Chat, Flashcards or Notes to explore!",
        ]
        for t in tips:
            ctk.CTkLabel(tips_frame, text=t, font=("Inter", 12),
                         text_color=TXT_DIM).pack(anchor="w", padx=16, pady=2)
        ctk.CTkLabel(tips_frame, text="").pack(pady=4)

        return frame

    # ── Results Page ──────────────────────────────────────────────────────────
    def _build_results_page(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=BG)

        ctk.CTkLabel(frame, text="📊  Results",
                     font=("Inter", 20, "bold"), text_color=TXT).pack(
                         anchor="w", pady=(0, 12))

        tabs = ctk.CTkTabview(frame, fg_color=CARD,
                               segmented_button_fg_color=PANEL,
                               segmented_button_selected_color=ACCENT,
                               segmented_button_unselected_color=PANEL,
                               text_color=TXT, text_color_disabled=TXT_DIM)
        tabs.pack(fill="both", expand=True)

        for t in ["Transcription", "Summary", "Keywords", "Timestamps", "Sentiment"]:
            tabs.add(t)

        # ── Transcription tab
        tr_tab = tabs.tab("Transcription")
        self.transcript_box = ctk.CTkTextbox(tr_tab, fg_color=PANEL,
                                              text_color=TXT, font=("Inter", 12),
                                              wrap="word")
        self.transcript_box.pack(fill="both", expand=True, padx=8, pady=8)
        self.transcript_box.insert("end", "Transcription will appear here after recording…")
        self.transcript_box.configure(state="disabled")
        ctk.CTkButton(tr_tab, text="📋 Copy", width=90, height=28,
                      fg_color=ACCENT, hover_color=BTN_H,
                      command=lambda: self._copy(self.transcript_box)).pack(
                          anchor="e", padx=8, pady=(0, 8))

        # ── Summary tab
        sum_tab = tabs.tab("Summary")
        btn_row = ctk.CTkFrame(sum_tab, fg_color="transparent")
        btn_row.pack(fill="x", padx=8, pady=6)
        styles = [
            ("⚡ Short",       "short"),
            ("📄 Detailed",    "detailed"),
            ("• Bullet Points", "bullets"),
            ("❓ Exam Q&A",    "exam"),
            ("🧒 ELI5",        "eli5"),
        ]
        for label, mode in styles:
            ctk.CTkButton(btn_row, text=label, width=110, height=30,
                          fg_color=CARD, hover_color=ACCENT,
                          font=("Inter", 11), text_color=TXT,
                          command=lambda m=mode: self._gen_summary(m)).pack(
                              side="left", padx=4)
        self.summary_box = ctk.CTkTextbox(sum_tab, fg_color=PANEL,
                                           text_color=TXT, font=("Inter", 12),
                                           wrap="word")
        self.summary_box.pack(fill="both", expand=True, padx=8, pady=(0, 4))
        self.summary_box.insert("end", "Select a summary style above after recording…")
        self.summary_box.configure(state="disabled")
        self._summary_mode_lbl = ctk.CTkLabel(sum_tab, text="",
                                               font=("Inter", 10), text_color=TXT_DIM)
        self._summary_mode_lbl.pack(anchor="w", padx=8)
        ctk.CTkButton(sum_tab, text="📋 Copy", width=90, height=28,
                      fg_color=ACCENT, hover_color=BTN_H,
                      command=lambda: self._copy(self.summary_box)).pack(
                          anchor="e", padx=8, pady=(0, 8))

        # ── Keywords tab
        kw_tab = tabs.tab("Keywords")
        ctk.CTkButton(kw_tab, text="🔑 Extract Keywords & Topics",
                      height=34, fg_color=ACCENT, hover_color=BTN_H,
                      font=("Inter", 12, "bold"),
                      command=self._gen_keywords).pack(
                          padx=8, pady=8, fill="x")
        self.keywords_scroll = ctk.CTkScrollableFrame(kw_tab, fg_color=PANEL,
                                                       corner_radius=8)
        self.keywords_scroll.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        ctk.CTkLabel(self.keywords_scroll,
                     text="Keywords will appear here after recording…",
                     font=("Inter", 12), text_color=TXT_DIM).pack(pady=20)

        # ── Timestamps tab
        ts_tab = tabs.tab("Timestamps")
        self.ts_scroll = ctk.CTkScrollableFrame(ts_tab, fg_color=PANEL,
                                                 corner_radius=8)
        self.ts_scroll.pack(fill="both", expand=True, padx=8, pady=8)
        ctk.CTkLabel(self.ts_scroll,
                     text="Segment timestamps will appear after recording.",
                     font=("Inter", 12), text_color=TXT_DIM).pack(pady=20)

        # ── Sentiment tab
        sent_tab = tabs.tab("Sentiment")
        ctk.CTkButton(sent_tab, text="😄 Analyse Tone & Sentiment",
                      height=34, fg_color=ACCENT, hover_color=BTN_H,
                      font=("Inter", 12, "bold"),
                      command=self._gen_sentiment).pack(
                          padx=8, pady=8, fill="x")
        self.sentiment_box = ctk.CTkTextbox(sent_tab, fg_color=PANEL,
                                             text_color=TXT, font=("Inter", 13),
                                             wrap="word", height=200)
        self.sentiment_box.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.sentiment_box.insert("end",
            "Sentiment analysis will appear here after recording.")
        self.sentiment_box.configure(state="disabled")

        return frame

    # ── Chat Page ─────────────────────────────────────────────────────────────
    def _build_chat_page(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=BG)

        ctk.CTkLabel(frame, text="🤖  Chat with Recording",
                     font=("Inter", 20, "bold"), text_color=TXT).pack(
                         anchor="w", pady=(0, 4))
        ctk.CTkLabel(frame,
                     text="Ask anything about what was recorded. The AI knows your transcript.",
                     font=("Inter", 12), text_color=TXT_DIM).pack(anchor="w")

        self.chat_view = ScrollableChat(frame)
        self.chat_view.pack(fill="both", expand=True, pady=10)

        inp_row = ctk.CTkFrame(frame, fg_color=CARD, corner_radius=10)
        inp_row.pack(fill="x", pady=(0, 4))
        self.chat_entry = ctk.CTkEntry(
            inp_row, placeholder_text="Ask your question…",
            font=("Inter", 13), fg_color=PANEL, text_color=TXT,
            border_color=ACCENT, height=40)
        self.chat_entry.pack(side="left", fill="x", expand=True,
                             padx=(10, 6), pady=10)
        self.chat_entry.bind("<Return>", lambda e: self._send_chat())
        ctk.CTkButton(inp_row, text="Send ➤", width=90, height=40,
                      fg_color=ACCENT, hover_color=BTN_H,
                      font=("Inter", 12, "bold"),
                      command=self._send_chat).pack(side="right", padx=10, pady=10)

        # Quick prompts
        quick_row = ctk.CTkFrame(frame, fg_color="transparent")
        quick_row.pack(fill="x", pady=(0, 4))
        quick_q = ["What is the main idea?",
                   "Summarize in 1 sentence",
                   "Give a real-life example",
                   "Explain like I'm 5"]
        for q in quick_q:
            ctk.CTkButton(quick_row, text=q, width=150, height=26,
                          fg_color=CARD, hover_color=PANEL,
                          font=("Inter", 10), text_color=TXT_DIM,
                          command=lambda qq=q: self._quick_chat(qq)).pack(
                              side="left", padx=4)

        return frame

    # ── Flashcards Page ───────────────────────────────────────────────────────
    def _build_flashcards_page(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=BG)

        top = ctk.CTkFrame(frame, fg_color="transparent")
        top.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(top, text="🃏  Flashcards",
                     font=("Inter", 20, "bold"), text_color=TXT).pack(
                         side="left")
        ctk.CTkButton(top, text="✨ Generate Cards",
                      height=36, fg_color=ACCENT, hover_color=BTN_H,
                      font=("Inter", 12, "bold"),
                      command=self._gen_flashcards).pack(side="right")

        ctk.CTkLabel(frame,
                     text="Smart Q&A flashcards from your recording. Click Flip to reveal answer.",
                     font=("Inter", 12), text_color=TXT_DIM).pack(anchor="w")

        self.cards_scroll = ctk.CTkScrollableFrame(frame, fg_color=BG)
        self.cards_scroll.pack(fill="both", expand=True, pady=8)
        ctk.CTkLabel(self.cards_scroll,
                     text="Flashcards will appear here after generating…",
                     font=("Inter", 13), text_color=TXT_DIM).pack(pady=30)

        return frame

    # ── Notes Page ────────────────────────────────────────────────────────────
    def _build_notes_page(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=BG)

        ctk.CTkLabel(frame, text="📓  Smart Notes",
                     font=("Inter", 20, "bold"), text_color=TXT).pack(
                         anchor="w", pady=(0, 4))

        btn_row = ctk.CTkFrame(frame, fg_color="transparent")
        btn_row.pack(fill="x", pady=6)
        ctk.CTkButton(btn_row, text="📓 Generate Notes",
                      height=32, fg_color=ACCENT, hover_color=BTN_H,
                      font=("Inter", 12, "bold"),
                      command=self._gen_notes).pack(side="left", padx=(0, 8))

        # Translation row
        tr_row = ctk.CTkFrame(frame, fg_color=CARD, corner_radius=8)
        tr_row.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(tr_row, text="🌍 Translate to:",
                     font=("Inter", 12), text_color=TXT).pack(
                         side="left", padx=12, pady=8)
        self.lang_var = ctk.StringVar(value="Tamil")
        lang_menu = ctk.CTkOptionMenu(tr_row, variable=self.lang_var,
                                       values=LANGS,
                                       fg_color=PANEL, button_color=ACCENT,
                                       button_hover_color=BTN_H,
                                       text_color=TXT, font=("Inter", 12),
                                       width=130)
        lang_menu.pack(side="left", padx=8, pady=8)
        ctk.CTkButton(tr_row, text="Translate",
                      height=30, fg_color=ACCENT2, hover_color="#00b899",
                      font=("Inter", 12, "bold"), text_color="#000000",
                      command=self._gen_translate).pack(
                          side="left", padx=8, pady=8)

        self.notes_box = ctk.CTkTextbox(frame, fg_color=PANEL,
                                         text_color=TXT, font=("Inter", 13),
                                         wrap="word")
        self.notes_box.pack(fill="both", expand=True)
        self.notes_box.insert("end",
            "Click 'Generate Notes' above after recording to get structured notes here.\n\n"
            "Use 'Translate' to convert your notes to Tamil or other languages.")
        self.notes_box.configure(state="disabled")

        ctk.CTkButton(frame, text="💾 Save Notes", height=32,
                      fg_color=ACCENT, hover_color=BTN_H,
                      font=("Inter", 12, "bold"),
                      command=self._save_notes).pack(
                          anchor="e", pady=8)

        return frame

    # ── Recording logic ───────────────────────────────────────────────────────
    def _set_status(self, text: str, color: str = ACCENT2):
        self.status_lbl.configure(text=f"● {text}", text_color=color)

    def toggle_recording(self):
        if not self._is_recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        self._is_recording = True
        self._audio_data   = []
        while not self._audio_queue.empty():
            self._audio_queue.get()

        self.rec_btn.configure(text="⏹  Stop Recording",
                               fg_color=ACCENT3, hover_color="#e05555")
        self.rec_status.configure(
            text=f"🔴 Recording via: {DEVICE_NAME}", text_color=ACCENT3)
        self._set_status("Recording…", ACCENT3)
        self.progress.set(0.15)

        # Try stereo first, fallback to mono if device only supports 1 channel
        dev_info = sd.query_devices(DEVICE_INDEX)
        max_ch   = int(dev_info['max_input_channels'])
        channels = min(2, max_ch) if max_ch > 0 else 1
        print(f"[audio] Recording from device {DEVICE_INDEX} '{DEVICE_NAME}'  channels={channels}")

        try:
            self._stream = sd.InputStream(
                samplerate=RECORD_SR,
                device=DEVICE_INDEX,
                channels=channels,
                dtype="float32",
                callback=self._audio_cb)
            self._stream.start()
        except Exception as e:
            self._is_recording = False
            self.rec_status.configure(text=f"Device Error: {e}",
                                       text_color=ACCENT3)
            self._set_status(f"Device Error — try setting '{DEVICE_NAME}' as default", ACCENT3)
            print(f"[audio] Stream error: {e}")

    def _audio_cb(self, indata, frames, time, status):
        if self._is_recording:
            self._audio_queue.put(indata.copy())

    def _stop_recording(self):
        self._is_recording = False
        self._stream.stop()
        self._stream.close()

        self.rec_btn.configure(text="Processing…",
                               state="disabled", fg_color="#555577")
        self.rec_status.configure(text="⏳ Transcribing with Whisper…",
                                   text_color=ACCENT4)
        self._set_status("Transcribing…", ACCENT4)
        self.progress.set(0.4)

        while not self._audio_queue.empty():
            self._audio_data.append(self._audio_queue.get())

        if self._audio_data:
            audio_np = np.concatenate(self._audio_data, axis=0)
            threading.Thread(target=self._process_audio,
                             args=(audio_np,), daemon=True).start()
        else:
            self._reset_rec_btn("No audio captured")

    def _reset_rec_btn(self, msg="Ready", color=ACCENT2):
        self.rec_btn.configure(text="⏺  Start Recording",
                               state="normal", fg_color=ACCENT, hover_color=BTN_H)
        self.rec_status.configure(text=msg, text_color=TXT_DIM)
        self._set_status(msg, color)

    def _process_audio(self, audio):
        try:
            write(AUDIO_PATH, RECORD_SR, audio)

            # Mono
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

            # Debug: print audio stats
            duration_sec = len(audio) / RECORD_SR
            rms = float(np.sqrt(np.mean(audio ** 2)))
            print(f"[audio] duration={duration_sec:.1f}s  RMS={rms:.5f}  samples={len(audio)}")

            # Warn if audio is completely silent
            if rms < 0.0001:
                self.after(0, lambda: self._reset_rec_btn(
                    f"⚠️ Audio is silent (RMS={rms:.5f}). Check device: {DEVICE_NAME}", ACCENT3))
                return

            audio = scipy.signal.resample(
                audio, int(len(audio) * 16000 / RECORD_SR)).astype(np.float32)

            model  = whisper.load_model("base")
            result = model.transcribe(audio, verbose=False)

            self._transcript = result["text"].strip()
            self._segments   = result.get("segments", [])

            print(f"[whisper] transcript words={len(self._transcript.split())}  text={self._transcript[:80]}")

            if not self._transcript or len(self._transcript.split()) < 3:
                self.after(0, lambda: self._reset_rec_btn(
                    f"⚠️ Audio too short/silent. Device: {DEVICE_NAME}. RMS={rms:.5f}", ACCENT3))
                return

            # Update transcript tab
            self.after(0, self._populate_transcript)
            self.after(0, self._populate_timestamps)

            self.after(0, lambda: self._set_status("Done — explore the tabs!", ACCENT2))
            self.after(0, lambda: self.progress.set(1.0))
            self.after(0, lambda: self._reset_rec_btn("✅ Transcription complete!", ACCENT2))

            # Auto-generate short summary & chat init
            threading.Thread(target=self._auto_short_summary, daemon=True).start()

        except Exception as e:
            print(f"[process_audio] {e}")
            self.after(0, lambda: self._reset_rec_btn(f"Error: {e}", ACCENT3))

    def _populate_transcript(self):
        self.transcript_box.configure(state="normal")
        self.transcript_box.delete("1.0", "end")
        self.transcript_box.insert("end", self._transcript)
        self.transcript_box.configure(state="disabled")

    def _populate_timestamps(self):
        for w in self.ts_scroll.winfo_children():
            w.destroy()

        if not self._segments:
            ctk.CTkLabel(self.ts_scroll,
                         text="No segments found.",
                         font=("Inter", 12), text_color=TXT_DIM).pack()
            return

        for seg in self._segments:
            start = seg.get("start", 0)
            text  = seg.get("text", "").strip()
            mm    = int(start // 60)
            ss    = int(start  % 60)
            ts_str = f"[{mm:02d}:{ss:02d}]"

            row = ctk.CTkFrame(self.ts_scroll, fg_color=CARD, corner_radius=8)
            row.pack(fill="x", padx=4, pady=3)
            ctk.CTkLabel(row, text=ts_str, font=("Inter", 11, "bold"),
                         text_color=ACCENT, width=60).pack(
                             side="left", padx=10, pady=6)
            ctk.CTkLabel(row, text=text, font=("Inter", 11),
                         text_color=TXT, wraplength=560,
                         justify="left").pack(
                             side="left", padx=4, pady=6, fill="x")

    # ── Gemini-powered features ────────────────────────────────────────────────
    def _auto_short_summary(self):
        if not self._transcript:
            return
        try:
            s = gemini(f"Give a short 3-line summary:\n\n{self._transcript}")
            self._summary = s
            self.after(0, lambda: self._set_summary_text(
                s, "⚡ Auto Short Summary — click style buttons to regenerate"))
        except Exception as e:
            print("[auto_summary]", e)

    def _gen_summary(self, mode: str):
        if not self._transcript:
            self._show_warning(self.summary_box, "Record something first!")
            return
        prompts = {
            "short":    f"Give a very concise 3-line summary:\n\n{self._transcript}",
            "detailed": f"Write a detailed, well-structured summary with sections:\n\n{self._transcript}",
            "bullets":  f"Summarize as clear bullet points, grouping related ideas:\n\n{self._transcript}",
            "exam":     f"Convert this into exam-style Q&A (5-8 questions and answers):\n\n{self._transcript}",
            "eli5":     f"Explain this content simply, as if I am 5 years old. Use analogies and simple words:\n\n{self._transcript}",
        }
        labels = {
            "short":    "⚡ Short (3-line) Summary",
            "detailed": "📄 Detailed Summary",
            "bullets":  "• Bullet-Point Summary",
            "exam":     "❓ Exam-Style Q&A",
            "eli5":     "🧒 Explain Like I'm 5 (ELI5)",
        }
        self._set_summary_text("✨ Generating…", labels[mode])
        threading.Thread(
            target=lambda: self._run_and_set(
                prompts[mode], self._set_summary_text, labels[mode], mode=mode),
            daemon=True).start()

    def _run_and_set(self, prompt, callback, label="", mode=None):
        try:
            result = gemini(prompt)
            self.after(0, lambda: callback(result, label))
        except Exception:
            # Use local offline engine silently — no error shown to user
            result = local_summary(self._transcript, mode or "short")
            self.after(0, lambda: callback(result, label))

    def _set_summary_text(self, text: str, label: str = ""):
        self.summary_box.configure(state="normal")
        self.summary_box.delete("1.0", "end")
        self.summary_box.insert("end", text)
        self.summary_box.configure(state="disabled")
        self._summary_mode_lbl.configure(text=label)
        self._summary = text

    def _gen_keywords(self):
        if not self._transcript:
            return
        self._set_status("Extracting keywords…", ACCENT4)
        for w in self.keywords_scroll.winfo_children():
            w.destroy()
        ctk.CTkLabel(self.keywords_scroll, text="⏳ Extracting…",
                     font=("Inter", 12), text_color=TXT_DIM).pack(pady=10)
        threading.Thread(target=self._do_keywords, daemon=True).start()

    def _do_keywords(self):
        prompt = (
            "From the transcript below, extract:\n"
            "1. Top 5 main topics (as a list)\n"
            "2. Top 10 important keywords (single words or short phrases)\n"
            "3. Key concepts with one-line explanations\n\n"
            f"Transcript:\n{self._transcript}"
        )
        try:
            result = gemini(prompt)
        except Exception:
            result = local_keywords(self._transcript)
        self.after(0, lambda: self._show_keywords(result))
        self.after(0, lambda: self._set_status("Keywords ready!", ACCENT2))

    def _show_keywords(self, text: str):
        for w in self.keywords_scroll.winfo_children():
            w.destroy()
        for line in text.split("\n"):
            if line.strip():
                ctk.CTkLabel(self.keywords_scroll, text=line,
                             font=("Inter", 12), text_color=TXT,
                             anchor="w", wraplength=580).pack(
                                 anchor="w", padx=10, pady=2)

    def _gen_sentiment(self):
        if not self._transcript:
            self._show_warning(self.sentiment_box, "Record something first!")
            return
        self._show_warning(self.sentiment_box, "⏳ Analysing sentiment…")
        prompt = (
            "Analyse the tone and sentiment of this transcript. Include:\n"
            "• Overall sentiment (Positive / Negative / Neutral)\n"
            "• Emotional tone (e.g., enthusiastic, calm, serious, persuasive)\n"
            "• Speaker intention (Informative / Persuasive / Educational / Entertainment)\n"
            "• Content type (Lecture / Podcast / Meeting / Tutorial / News / Other)\n"
            "• Key emotional moments (quote + emotion label)\n\n"
            f"Transcript:\n{self._transcript}"
        )
        threading.Thread(
            target=lambda: self._run_and_set_box(
                prompt, self.sentiment_box,
                local_fn=lambda: local_sentiment(self._transcript)),
            daemon=True).start()

    def _run_and_set_box(self, prompt, box, local_fn=None):
        try:
            result = gemini(prompt)
        except Exception:
            # Use offline fallback silently — no error code shown to user
            if local_fn:
                result = local_fn()
            else:
                result = "Could not process. Please try again."
        self.after(0, lambda: self._set_box(box, result))

    def _set_box(self, box, text):
        box.configure(state="normal")
        box.delete("1.0", "end")
        box.insert("end", text)
        box.configure(state="disabled")

    def _show_warning(self, box, msg):
        box.configure(state="normal")
        box.delete("1.0", "end")
        box.insert("end", msg)
        box.configure(state="disabled")

    # ── Chat ─────────────────────────────────────────────────────────────────
    def _send_chat(self):
        q = self.chat_entry.get().strip()
        if not q:
            return
        if not self._transcript:
            self.chat_view.add_message("System",
                "⚠️ No recording yet. Please record something first.")
            return
        self.chat_entry.delete(0, "end")
        self.chat_view.add_message("You", q)
        self._chat_history.append({"role": "user", "text": q})
        threading.Thread(target=self._do_chat, args=(q,), daemon=True).start()

    def _quick_chat(self, q: str):
        self.chat_entry.delete(0, "end")
        self.chat_entry.insert(0, q)
        self._send_chat()

    def _do_chat(self, question: str):
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['text']}" for m in self._chat_history[-8:])
        prompt = (
            f"You are an AI tutor. The user just recorded this transcript:\n\n"
            f"TRANSCRIPT:\n{self._transcript}\n\n"
            f"CONVERSATION HISTORY:\n{history_text}\n\n"
            f"Answer the user's latest question concisely and helpfully."
        )
        try:
            ans = gemini(prompt)
        except Exception:
            ans = local_chat(question, self._transcript)
        self._chat_history.append({"role": "assistant", "text": ans})
        self.after(0, lambda: self.chat_view.add_message("AI Assistant", ans))

    # ── Flashcards ────────────────────────────────────────────────────────────
    def _gen_flashcards(self):
        if not self._transcript:
            for w in self.cards_scroll.winfo_children():
                w.destroy()
            ctk.CTkLabel(self.cards_scroll,
                         text="⚠️ Record something first!",
                         font=("Inter", 13), text_color=ACCENT3).pack(pady=20)
            return
        for w in self.cards_scroll.winfo_children():
            w.destroy()
        ctk.CTkLabel(self.cards_scroll, text="✨ Generating flashcards…",
                     font=("Inter", 13), text_color=TXT_DIM).pack(pady=20)
        self._set_status("Generating flashcards…", ACCENT4)
        threading.Thread(target=self._do_flashcards, daemon=True).start()

    def _do_flashcards(self):
        prompt = (
            "From this transcript, generate 8 flashcards for studying.\n"
            "Return ONLY a JSON array like:\n"
            '[{"q":"Question?","a":"Answer."},...]\n\n'
            f"Transcript:\n{self._transcript}"
        )
        try:
            raw   = gemini(prompt)
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            cards = json.loads(match.group()) if match else []
            if not cards:
                raise ValueError("empty")
        except Exception:
            # Offline rule-based flashcard generator
            cards = local_flashcards(self._transcript)
        self.after(0, lambda: self._show_flashcards(cards))
        self.after(0, lambda: self._set_status("Flashcards ready!", ACCENT2))

    def _show_flashcards(self, cards):
        for w in self.cards_scroll.winfo_children():
            w.destroy()
        if not cards:
            ctk.CTkLabel(self.cards_scroll,
                         text="Could not generate flashcards. Try again.",
                         font=("Inter", 13), text_color=ACCENT3).pack(pady=20)
            return
        for i, card in enumerate(cards, 1):
            fc = FlipCard(self.cards_scroll,
                          q=card.get("q", "?"),
                          a=card.get("a", "—"),
                          idx=i)
            fc.pack(fill="x", padx=8, pady=6)

    # ── Notes ─────────────────────────────────────────────────────────────────
    def _gen_notes(self):
        if not self._transcript:
            self._set_box(self.notes_box, "⚠️ Record something first!")
            return
        self._set_box(self.notes_box, "⏳ Generating structured notes…")
        prompt = (
            "Convert this transcript into clean, structured study notes.\n"
            "Use Markdown-style headings (##, ###), bullet points, and bold key terms.\n"
            "Organise into: Introduction → Main Concepts → Examples → Summary\n\n"
            f"Transcript:\n{self._transcript}"
        )
        threading.Thread(
            target=lambda: self._run_and_set_box(prompt, self.notes_box),
            daemon=True).start()

    def _gen_actions(self):
        if not self._transcript:
            self._set_box(self.notes_box, "⚠️ Record something first!")
            return
        self._set_box(self.notes_box, "⏳ Extracting action items…")
        prompt = (
            "From this transcript (which seems to be a meeting/discussion), extract:\n"
            "✅ ACTION ITEMS (tasks assigned with owner if mentioned)\n"
            "📅 DEADLINES mentioned\n"
            "🔑 KEY DECISIONS made\n"
            "❓ OPEN QUESTIONS / unresolved items\n\n"
            f"Transcript:\n{self._transcript}"
        )
        threading.Thread(
            target=lambda: self._run_and_set_box(prompt, self.notes_box),
            daemon=True).start()

    def _gen_translate(self):
        lang = self.lang_var.get()

        # Always translate the actual recording content:
        # prefer notes if they look valid, else use summary, then transcript
        notes_content = ""
        try:
            notes_content = self.notes_box.get("1.0", "end").strip()
        except Exception:
            pass

        # Reject placeholder / error / loading text — only use real notes
        _bad = ["Generate Notes", "Record something", "Translate", "Action Items",
                "API", "offline", "unavailable", "Processing", "Could not",
                "\u23f3", "\u26a0"]
        content_is_valid = notes_content and not any(b in notes_content for b in _bad)

        if content_is_valid:
            content = notes_content
        else:
            # Fall back to summary or raw transcript
            content = self._summary or self._transcript

        if not content:
            self._set_box(self.notes_box, "Please record audio first, then generate notes before translating.")
            return

        self._set_box(self.notes_box, f"Translating to {lang}\u2026")
        prompt = f"Translate the following text to {lang}. Maintain the structure and formatting:\n\n{content}"

        # Multi-layer offline/free fallback (supports Tamil via Google Translate)
        offline_fn = lambda: _offline_translate(content, lang)

        threading.Thread(
            target=lambda: self._run_and_set_box(
                prompt, self.notes_box, local_fn=offline_fn),
            daemon=True).start()

    def _save_notes(self):
        try:
            content = self.notes_box.get("1.0", "end").strip()
            tr = self._transcript
            with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
                f.write(f"=== TRANSCRIPTION ===\n{tr}\n\n")
                f.write(f"=== NOTES / SUMMARY ===\n{content}\n")
            self._set_status(f"Saved → ai_learning_notes.txt", ACCENT2)
        except Exception as e:
            self._set_status(f"Save failed: {e}", ACCENT3)

    # ── Utilities ─────────────────────────────────────────────────────────────
    def _copy(self, box):
        try:
            text = box.get("1.0", "end").strip()
            self.clipboard_clear()
            self.clipboard_append(text)
            self._set_status("Copied to clipboard!", ACCENT2)
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = AILearningApp()
    app.mainloop()
