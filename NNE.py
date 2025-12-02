import os
import sys
import math
import re
import logging
import time
import random
import threading
import textwrap
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
EMBED_AVAILABLE = False
EMBED_MODEL = None
try:
    from sentence_transformers import SentenceTransformer
    EMBED_AVAILABLE = True
except Exception:
    EMBED_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False
SAMPLE_IMAGE_PATH = "/mnt/data/A_two-part_digital_plot_illustrates_neural_activit.png"
WINDOW_MS = 300
OSCILLOSCOPE_WINDOW_MS = 300
ANIM_FPS = 30
PLOT_UPDATE_INTERVAL_MS = int(1000 / ANIM_FPS)
N_NEURONS = 32
N_CHANNELS = 8
DT = 1.0
INSULT_WORDS = [
    "idiot", "stupid", "dumb", "moron", "hate", "shut up", "trash", "sucks",
    "jerk", "loser", "ugly", "worthless", "fool" ,"annoying", "nonsense", "crap","Fuck","shit","bastard","asshole","Fuck","Fuck you"
]
INSULT_THRESHOLD = 3
NORMALIZATION_MAP = {
    "looser": "loser",
    "definately": "definitely",
    "recieve": "receive",
    "teh": "the",
    "cant": "can't",
    "wont": "won't",
    "im": "i'm",
}
RESPONSE_TEMPLATES = {
    "Joy": [
        "That's wonderful to hear! Tell me more if you like.",
        "Awesome â€” that made my day to hear. Anything else?",
    ],
    "Sadness": [
        "I'm sorry you're feeling down. I'm here to listen.",
        "That sounds really tough. Want to tell me more about it?",
    ],
    "Anger": [
        "I hear you're upset. Let's try to talk calmly about it.",
        "It sounds like this is frustrating â€” I'm listening.",
    ],
    "Boredom": [
        "If you're bored, want a quick game or a fun fact?",
        "I can share a joke or an interesting tidbit â€” which do you prefer?",
    ],
    "Fear": [
        "I'm sorry you're feeling scared. Are you safe right now?",
        "That sounds worrying. Would you like grounding tips or to talk more?",
    ],
    "Neutral": [
        "I see. Tell me more if you want.",
        "Okay â€” what else is on your mind?",
    ]
}
POLITE_FIRM_TEMPLATES = [
    "I won't engage with abusive language â€” let's be respectful.",
    "I understand you're upset, but I can't continue if you use insults.",
    "Please stop using hurtful language. I'm happy to help if we keep it civil.",
]
logger = logging.getLogger("neuronova")
level = logging.DEBUG if os.environ.get("NEURONOVA_DEBUG", "0") == "1" else logging.INFO
logging.basicConfig(level=level, format='[%(levelname)s] %(message)s')
logger.setLevel(level)
USE_EMBEDDINGS = False
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
ENABLE_LLM_FALLBACK = False
CHATBOT_MODEL_NAME = "gpt2"
EMOTIONS = [
    "Admiration", "Adoration", "Aesthetic Appreciation", "Amusement", "Anger",
    "Anxiety", "Awe", "Awkwardness", "Boredom", "Calmness",
    "Confusion", "Craving", "Disappointment", "Disgust", "Empathic Pain",
    "Entrancement", "Excitement", "Fear", "Horror", "Interest",
    "Joy", "Nostalgia", "Romance", "Sadness", "Satisfaction",
    "Sexual Desire", "Sympathy"
]
EMOTION_COLORS = {
    "Admiration": "
    "Amusement": "
    "Awkwardness": "
    "Craving": "
    "Entrancement": "
    "Interest": "
    "Sadness": "
}
EMOTION_PROTOTYPES = {
    "Admiration": ["I'm impressed by your skill", "That performance filled me with respect"],
    "Adoration": ["I feel deep affection for that person", "This is so lovable and warm"],
    "Aesthetic Appreciation": ["That is incredibly beautiful art", "The scenery moved me with its beauty"],
    "Amusement": ["That's funny and makes me laugh", "I find this amusing and humorous"],
    "Anger": ["I'm furious and upset about this", "That makes me so angry"],
    "Anxiety": ["I feel nervous and worried", "I'm anxious about what will happen"],
    "Awe": ["I am filled with wonder and amazement", "This leaves me speechless with awe"],
    "Awkwardness": ["This is embarrassing and I feel awkward", "I cringe and feel socially uncomfortable"],
    "Boredom": ["I'm bored and uninterested", "This is dull and I'm not engaged"],
    "Calmness": ["I feel relaxed and peaceful", "Everything is calm and serene"],
    "Confusion": ["I don't understand what's happening", "This is confusing and unclear to me"],
    "Craving": ["I really want that now", "I crave and desire it strongly"],
    "Disappointment": ["I'm disappointed and let down", "That did not meet my expectations"],
    "Disgust": ["That is disgusting and revolting", "I feel repulsed by that"],
    "Empathic Pain": ["I feel pain for someone else's suffering", "That hurts me to hear about them"],
    "Entrancement": ["I'm mesmerized and completely absorbed", "I am entranced by the experience"],
    "Excitement": ["I'm so excited and thrilled", "I can't wait Ã¢â‚¬â€ I'm pumped"],
    "Fear": ["I'm scared and afraid", "I feel threatened and fearful"],
    "Horror": ["This is horrifying and sickening", "I feel terrified and disgusted"],
    "Interest": ["I'm curious and want to learn more", "This sparks my interest and attention"],
    "Joy": ["I'm happy and full of joy", "That brings me delight and pleasure"],
    "Nostalgia": ["This reminds me of old memories", "I feel bittersweet remembering the past"],
    "Romance": ["I feel warm romantic love", "This feels tender and romantic"],
    "Sadness": ["I'm sad and full of sorrow", "This makes me feel down and lonely"],
    "Satisfaction": ["I'm content and satisfied", "That leaves me feeling fulfilled"],
    "Sexual Desire": ["I feel sexual attraction and desire", "This arouses me physically"],
    "Sympathy": ["I'm sorry to hear that and I care", "I feel compassion for that person"]
}
def tokenize(text):
    if text is None:
        return []
    t = text.lower()
    t = t.replace("i'm", "im").replace("it's", "its").replace("don't", "dont").replace("can't", "cant")
    toks = []
    for w in t.split():
        w2 = ''.join(ch for ch in w if ch.isalnum())
        if w2:
            toks.append(w2)
    return toks
def normalize_text(text: str) -> str:
    if not text:
        return text
    s = text
    for bad, good in NORMALIZATION_MAP.items():
        s = re.sub(r"\b" + re.escape(bad) + r"\b", good, s, flags=re.IGNORECASE)
    return s
def math_log(x):
    import math
    return math.log(1 + x)
if USE_EMBEDDINGS and EMBED_AVAILABLE:
    try:
        EMBED_MODEL = SentenceTransformer(EMBEDDING_MODEL)
        prototype_texts = []
        prototype_map = []
        for emo, texts in EMOTION_PROTOTYPES.items():
            for t in texts:
                prototype_texts.append(t)
                prototype_map.append(emo)
        prototype_embs = EMBED_MODEL.encode(prototype_texts, convert_to_numpy=True, show_progress_bar=False)
    except Exception:
        EMBED_MODEL = None
        EMBED_AVAILABLE = False
        prototype_embs = None
else:
    prototype_texts = []
    prototype_map = []
    prototype_embs = None
class PrototypeSemanticDetector:
    def __init__(self, prototypes_map):
        self.prototypes_map = {k: [p.lower() for p in v] for k, v in prototypes_map.items()}
        vocab = set()
        for proto_list in self.prototypes_map.values():
            for s in proto_list:
                vocab.update(tokenize(s))
        self.vocab = sorted(vocab)
        self.word_to_idx = {w:i for i,w in enumerate(self.vocab)}
        self.prototype_vecs = {}
        for emo, proto_list in self.prototypes_map.items():
            vecs = [self._text_vector(s) for s in proto_list]
            avg = np.mean(vecs, axis=0) if len(vecs)>0 else np.zeros(len(self.vocab))
            norm = np.linalg.norm(avg)
            self.prototype_vecs[emo] = (avg / (norm + 1e-12)) if norm>0 else avg
    def _text_vector(self, text):
        toks = tokenize(text)
        if not toks:
            return np.zeros(len(self.vocab), dtype=float)
        counts = Counter(toks)
        vec = np.zeros(len(self.vocab), dtype=float)
        for w, c in counts.items():
            idx = self.word_to_idx.get(w)
            if idx is not None:
                vec[idx] = 1.0 + math_log(c)
        norm = np.linalg.norm(vec)
        return (vec / (norm + 1e-12)) if norm>0 else vec
    def predict(self, text):
        toks = tokenize(text)
        if not toks:
            return np.zeros(len(EMOTIONS)), "Neutral", {}
        text_vec = self._text_vector(text)
        sims = np.zeros(len(EMOTIONS), dtype=float)
        for i, emo in enumerate(EMOTIONS):
            proto = self.prototype_vecs.get(emo, np.zeros(len(self.vocab)))
            sims[i] = max(0.0, float(np.dot(text_vec, proto)))
        maxv = sims.max()
        if maxv > 0:
            sims = sims / (maxv + 1e-12)
            dominant = EMOTIONS[int(np.argmax(sims))]
        else:
            dominant = "Neutral"
        return sims, dominant, {}
class EmbeddingSemanticDetector:
    def __init__(self, embed_model, prototype_texts, prototype_map, prototype_embs):
        self.model = embed_model
        self.prototype_texts = prototype_texts
        self.prototype_map = prototype_map
        self.prototype_embs = prototype_embs
        self.emotions = EMOTIONS
    def predict(self, text):
        if not text or text.strip()=="":
            return np.zeros(len(self.emotions)), "Neutral", {}
        try:
            q_emb = self.model.encode([text], convert_to_numpy=True)[0]
        except Exception:
            return np.zeros(len(self.emotions)), "Neutral", {}
        q_norm = np.linalg.norm(q_emb) + 1e-12
        q_emb_n = q_emb / q_norm
        prot = self.prototype_embs
        prot_norms = np.linalg.norm(prot, axis=1) + 1e-12
        prot_n = prot / prot_norms[:, None]
        sims = np.dot(prot_n, q_emb_n)
        emo_scores = {emo:0.0 for emo in self.emotions}
        for sim_val, emo in zip(sims, self.prototype_map):
            if sim_val > emo_scores[emo]:
                emo_scores[emo] = float(sim_val)
        vec = np.array([emo_scores.get(e, 0.0) for e in self.emotions], dtype=float)
        maxv = vec.max()
        if maxv > 0:
            vec = vec / (maxv + 1e-12)
            dominant = self.emotions[int(np.argmax(vec))]
        else:
            dominant = "Neutral"
        return vec, dominant, {}
prototype_detector = PrototypeSemanticDetector(EMOTION_PROTOTYPES)
embedding_detector = None
if EMBED_AVAILABLE and prototype_embs is not None:
    embedding_detector = EmbeddingSemanticDetector(EMBED_MODEL, prototype_texts, prototype_map, prototype_embs)
class SpikingEncoder:
    def __init__(self, window_ms=WINDOW_MS, channels=N_CHANNELS, seed=1):
        self.window_ms = window_ms
        self.channels = channels
        self.rng = np.random.RandomState(seed)
    def token_salience(self, token):
        vowel_frac = sum(ch in 'aeiou' for ch in token) / max(1, len(token))
        uncommon = sum(ch in 'qzx' for ch in token)
        return len(token) * (1 + vowel_frac) + 2*uncommon
    def encode(self, text):
        tokens = tokenize(text)
        if not tokens:
            return [[] for _ in range(self.channels)], tokens
        saliences = np.array([self.token_salience(t) for t in tokens], dtype=float)
        if len(saliences) > 1:
            sal_range = np.ptp(saliences)
            sal_range = max(1e-9, sal_range)
            saliences = (saliences - saliences.min()) / sal_range
        else:
            saliences = np.array([0.5])
        channel_trains = [[] for _ in range(self.channels)]
        for i, token in enumerate(tokens):
            ch = (sum(ord(c) for c in token) + i) % self.channels
            t_ms = (1.0 - saliences[i]) * (self.window_ms * 0.8) + self.rng.normal(0, 6.0)
            t_ms = float(max(0.0, min(self.window_ms, t_ms)))
            channel_trains[ch].append(t_ms)
        for ch in range(self.channels):
            channel_trains[ch].sort()
        return channel_trains, tokens
class LIFPopulation:
    def __init__(self, n_neurons=N_NEURONS, n_channels=N_CHANNELS, dt=DT, tau_m=20.0, v_reset=0.0, v_thresh=1.0, seed=2):
        self.n = n_neurons; self.channels = n_channels; self.dt = dt; self.tau = tau_m
        self.v_reset = v_reset; self.v_thresh = v_thresh
        self.rng = np.random.RandomState(seed)
        self.W = self.rng.normal(0.8, 0.3, size=(self.n, self.channels)).clip(0, 2.0)
        self.v = np.zeros(self.n, dtype=float)
        self.spike_times = [[] for _ in range(self.n)]
        self.v_history = np.zeros((0, self.n), dtype=float)
    def reset(self):
        self.v[:] = 0.0
        self.spike_times = [[] for _ in range(self.n)]
        self.v_history = np.zeros((0, self.n), dtype=float)
    def run(self, channel_trains, t_max_ms=WINDOW_MS):
        self.reset()
        n_steps = int(math.ceil(t_max_ms / self.dt))
        spikes = np.zeros((self.channels, n_steps), dtype=bool)
        for ch_idx, times in enumerate(channel_trains):
            for t in times:
                step = int(round(t / self.dt))
                if 0 <= step < n_steps: spikes[ch_idx, step] = True
        v_hist = np.zeros((n_steps, self.n), dtype=float)
        for step in range(n_steps):
            inp = (spikes[:, step].astype(float) @ self.W.T)
            dv = (-self.v / self.tau + inp) * (self.dt)
            self.v += dv
            just_fired = self.v >= self.v_thresh
            if just_fired.any():
                idxs = np.where(just_fired)[0]
                for idx in idxs:
                    self.spike_times[idx].append(step * self.dt)
                    self.v[idx] = self.v_reset
            v_hist[step, :] = self.v
        self.v_history = v_hist
        rates = np.array([len(times) / (t_max_ms / 1000.0) for times in self.spike_times], dtype=float)
        return rates
class DecisionModule:
    def __init__(self, rate_threshold=10.0):
        self.rate_threshold = rate_threshold
        self.positive_set = {"Admiration","Adoration","Aesthetic Appreciation","Amusement","Awe","Calmness","Excitement","Interest","Joy","Satisfaction","Romance"}
        self.negative_set = {"Anger","Anxiety","Confusion","Disappointment","Disgust","Empathic Pain","Fear","Horror","Sadness","Boredom"}
    def decide(self, firing_rates, dominant_emotion, meta=None):
        meta = meta or {}
        mean_rate = float(np.mean(firing_rates))
        if meta.get("self_harm_risk", False):
            return {"mean_rate": mean_rate, "action":"take_action", "tone":"urgent", "dominant": dominant_emotion, "meta": meta}
        if dominant_emotion == "Neutral":
            if mean_rate < self.rate_threshold * 0.6:
                return {"mean_rate": mean_rate, "action":"idle", "tone":"calm", "dominant":"Neutral", "meta": meta}
            elif mean_rate < self.rate_threshold:
                return {"mean_rate": mean_rate, "action":"observe", "tone":"curious", "dominant":"Neutral", "meta": meta}
            else:
                return {"mean_rate": mean_rate, "action":"speak_serious", "tone":"direct", "dominant":"Neutral", "meta": meta}
        if dominant_emotion in self.positive_set:
            return {"mean_rate": mean_rate, "action":"speak_friendly", "tone":"warm", "dominant": dominant_emotion, "meta": meta}
        if dominant_emotion in self.negative_set:
            if dominant_emotion == "Boredom":
                return {"mean_rate": mean_rate, "action":"engage_user", "tone":"uplifting", "dominant": dominant_emotion, "meta": meta}
            if mean_rate > self.rate_threshold * 1.2:
                return {"mean_rate": mean_rate, "action":"take_action", "tone":"urgent", "dominant": dominant_emotion, "meta": meta}
            return {"mean_rate": mean_rate, "action":"speak_serious", "tone":"concerned", "dominant": dominant_emotion, "meta": meta}
        return {"mean_rate": mean_rate, "action":"observe", "tone":"curious", "dominant": dominant_emotion, "meta": meta}
class ResponseGenerator:
    def __init__(self, seed=123):
        self.rng = random.Random(seed)
        self.suggestions = [
            "Let's play a quick word game Ã¢â‚¬â€ I start with a word, you reply with the first thing that comes to your mind.",
            "Want to hear a joke or an interesting fact? (type 'joke' or 'fact')",
            "I can give you a short brain teaser in 30 seconds Ã¢â‚¬â€ want to try?",
            "Tell me a topic you like Ã¢â‚¬â€ I'll give a fun fact or a mini-quiz.",
            "How about a short breathing exercise or a 2-minute stretch? I'll guide you."
        ]
        self.jokes = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "I told my computer I needed a break, and it said: 'No problem Ã¢â‚¬â€ I'll go to sleep.'"
        ]
        self.facts = [
            "Octopuses have three hearts.",
            "Honey never spoils Ã¢â‚¬â€ jars from ancient Egypt are still edible."
        ]
    def generate(self, action, tone, user_text, dominant_emotion, meta=None):
        meta = meta or {}
        ut = (user_text or "").strip()
        logger.debug(f"ResponseGenerator.generate: action={action} tone={tone} dominant={dominant_emotion} meta={meta} user_text='{ut}'")
        if meta.get("self_harm_risk", False):
            logger.info("Response: self-harm risk template returned")
            return ("I'm really sorry you're feeling this way. If you're thinking about harming yourself or are in immediate danger, "
                "please contact local emergency services or a crisis line right now. If you'd like, we can stay here and talk â€” are you safe right now?")
        if dominant_emotion == "Anger" or meta.get("anger_override", False) or meta.get("insult_count", 0) > 0:
            ic = meta.get("insult_count", 0)
            if ic >= INSULT_THRESHOLD * 2:
                logger.info("Response: high insult count -> disengage message")
                return "I've received repeated abusive language. I'm stepping back for now."
            snippet = " ".join(ut.split()[:6])
            tmpl = self.rng.choice(POLITE_FIRM_TEMPLATES)
            reply = f"{tmpl} (I heard: '{snippet}')"
            logger.debug(f"Response chosen (polite firm): {reply}")
            return reply
        ut_l = ut.lower()
        if dominant_emotion == "Sexual Desire":
            return "I can't engage in sexual content. If you'd like, we can talk about relationships, boundaries, or something light-hearted instead."
        if any(kw in ut_l for kw in ("tell me something", "tell me", "can you", "could you", "what", "who", "how", "why", "?")):
            if "joke" in ut_l:
                return self.rng.choice(self.jokes)
            if "fact" in ut_l or "something" in ut_l:
                return self.rng.choice(self.facts)
            if len(ut_l.split()) < 6:
                return "Sure â€” could you give me a topic? Or I can share a fun fact: " + self.rng.choice(self.facts)
            return "That's interesting â€” tell me a bit more about what you want to hear, or I can share a quick fact or joke."
        if action == "engage_user":
            ut_l = ut.lower()
            if "joke" in ut_l:
                return self.rng.choice(self.jokes)
            if "fact" in ut_l:
                return self.rng.choice(self.facts)
            return "You said you're bored Ã¢â‚¬â€ " + self.rng.choice(self.suggestions)
        if dominant_emotion in RESPONSE_TEMPLATES:
            template = self.rng.choice(RESPONSE_TEMPLATES[dominant_emotion])
            echo = " ".join(ut.split()[:8])
            reply = f"{template} ({echo})" if echo else template
            logger.debug(f"Response chosen from templates: {reply}")
            return reply
        if action == "speak_friendly":
            return "Nice! That sounds great â€” want to continue or try something fun?"
        if action == "speak_serious":
            return "I understand. Would you like help or someone to talk to?"
        if action == "take_action":
            return "I'm taking this seriously and prioritizing immediate help. Are you safe right now?"
        if action == "observe":
            return self.rng.choice([
                "Oh â€” what made you feel that way? Tell me more.",
                "I get that â€” would you like a fun fact or we can chat about something you enjoy?",
                "Interesting. Want me to share a quick idea to try right now?"
            ])
        if action == "idle":
            return "Okay â€” I'm here if you need anything."
        logger.debug("ResponseGenerator: falling back to default reply")
        return "Thanks for sharing â€” I'm listening."
class ConversationTracker:
    def __init__(self, emotions):
        self.emotions = emotions
        self.records = []
        self.fig = None
        self.axes = {}
        self.heat_cbar = None
        self._pending_show = threading.Event()
        self._show_lock = threading.Lock()
        self._graph_open = False
        self._auto_update = False
        self._graph_open_time = None
        self._min_display_time = 200.0
        plt.ion()
    @staticmethod
    def _sanitize_text(text):
        if not text:
            return text
        result = ''.join(char if (32 <= ord(char) < 127) or ord(char) > 159 
                         else '' 
                         for char in str(text))
        return result
    def add_entry(self, user_text, emotion_vec, dominant, decision, meta, lif, tokens):
        neg_list = ["Sadness", "Fear", "Empathic Pain", "Anxiety", "Anger", "Disgust", "Horror"]
        try:
            neg_score = float(np.sum([emotion_vec[EMOTIONS.index(e)] for e in neg_list if e in EMOTIONS]))
        except Exception:
            neg_score = 0.0
        meta_safe = meta or {}
        if meta_safe.get("self_harm_risk", False):
            neg_score = max(neg_score, 1.0)
        neg_score = float(max(0.0, min(1.0, neg_score)))
        self.records.append({
            "text": user_text,
            "vec": emotion_vec.copy(),
            "dominant": dominant,
            "decision": decision,
            "meta": meta_safe,
            "mean_rate": decision.get("mean_rate", 0.0),
            "tokens": tokens,
            "spikes": [lst[:] for lst in lif.spike_times],
            "v_history": lif.v_history.copy() if lif.v_history is not None else None,
            "mental_health": neg_score
        })
        if len(self.records) > 200:
            self.records = self.records[-200:]
        if self._graph_open and self._auto_update and self.fig is not None:
            if threading.current_thread() is threading.main_thread():
                try:
                    self._update_plot()
                except Exception as e:
                    logger.debug(f"Auto-update graph error: {e}")
    def print_history(self):
        if not self.records:
            print("[History] No conversation yet.")
            return
        print("\n=== Conversation History ===")
        for idx, rec in enumerate(self.records, 1):
            meta = rec["meta"]
            meta_txt = f" meta={meta}" if meta else ""
            print(f"{idx:02d}. User: {rec['text']}")
            print(f"    Dominant: {rec['dominant']}  mean_rate: {rec['mean_rate']:.2f}{meta_txt}")
        print("============================\n")
    def show_graph(self):
        if not self.records:
            print("[Graph] No data yet. Start chatting to see the graph update live!")
            return
        if threading.current_thread() is threading.main_thread():
            if self.fig is None:
                self._create_figure()
            self._auto_update = True
            self._graph_open = True
            self._graph_open_time = time.time()
            self._update_plot()
            try:
                self.fig.show()
                plt.show(block=False)
                plt.draw()
                plt.pause(0.1)
                print("[Graph] Live graph opened! It will stay open for at least 1 minute and update automatically as you chat.")
            except Exception as e:
                logger.debug(f"show_graph error: {e}")
                try:
                    plt.draw()
                    plt.pause(0.1)
                    print("[Graph] Graph window opened (fallback)")
                except Exception:
                    self._graph_open = False
        else:
            self._pending_show.set()
            print("[Graph] Request queued; graph will open on main thread (minimum 1 minute display time).")
    def _create_figure(self):
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("_create_figure() must be called from main thread")
        fig, axes = plt.subplots(3, 2, figsize=(18, 16), constrained_layout=True)
        fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.08, hspace=0.3, wspace=0.3)
        self.fig = fig
        ax_heat = axes[0, 0]
        ax_mental = axes[0, 1]
        ax_rate = axes[1, 0]
        ax_volt = axes[2, 0]
        try:
            axes[1,1].axis('off')
            axes[2,1].axis('off')
        except Exception:
            pass
        self.axes["heat"] = ax_heat
        self.axes["mental"] = ax_mental
        self.axes["rate"] = ax_rate
        self.axes["volt"] = ax_volt
    def _update_plot(self):
        if threading.current_thread() is not threading.main_thread():
            logger.debug("_update_plot called from non-main thread, skipping")
            return
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
            warnings.filterwarnings('ignore', category=UserWarning, module='tkinter')
        if not self.records or self.fig is None:
            return
        try:
            if not plt.fignum_exists(self.fig.number):
                self._graph_open = False
                return
        except Exception:
            self._graph_open = False
            return
        if self._graph_open_time is not None:
            elapsed_time = time.time() - self._graph_open_time
            if elapsed_time < self._min_display_time:
                pass
        heat_ax = self.axes.get("heat")
        mental_ax = self.axes.get("mental")
        rate_ax = self.axes.get("rate")
        volt_ax = self.axes.get("volt")
        if not all([heat_ax, rate_ax, volt_ax]):
            return
        heat_ax.clear(); rate_ax.clear(); volt_ax.clear()
        if mental_ax is not None:
            mental_ax.clear()
        try:
            data = np.stack([rec["vec"] for rec in self.records], axis=1)
            n_emotions = len(self.emotions)
            im = heat_ax.imshow(data, aspect=0.5, cmap="viridis", interpolation="nearest", vmin=0.0, vmax=1.0)
            heat_ax.set_yticks(range(n_emotions))
            emotion_labels = [f"  {self._sanitize_text(emo)}" for emo in self.emotions]
            heat_ax.set_yticklabels(emotion_labels, fontsize=8, ha='right', va='center')
            heat_ax.tick_params(axis='y', pad=25, length=6, width=1)
            heat_ax.set_ylim(-1.0, n_emotions + 0.5)
            pos = heat_ax.get_position()
            heat_ax.set_position([pos.x0 + 0.02, pos.y0, pos.width * 0.95, pos.height])
            nrec = len(self.records)
            if nrec <= 30:
                xt = list(range(nrec))
                xlabels = [str(i+1) for i in range(nrec)]
            else:
                step = max(1, nrec // 20)
                xt = list(range(0, nrec, step))
                xlabels = [str(i+1) for i in xt]
            heat_ax.set_xticks(xt)
            heat_ax.set_xticklabels(xlabels)
            heat_ax.set_title("Emotion activation (conversation timeline) - LIVE")
            heat_ax.set_xlabel("Turn")
            if self.heat_cbar is not None:
                try:
                    self.heat_cbar.remove()
                except Exception:
                    pass
            self.heat_cbar = self.fig.colorbar(im, ax=heat_ax, fraction=0.015, pad=0.02)
            rates = [rec["mean_rate"] for rec in self.records]
            idxs = list(range(1, len(rates)+1))
            rate_ax.plot(idxs, rates, marker="o", label="mean rate", linewidth=1.5, markersize=4)
            if len(rates) >= 3:
                ma = np.convolve(rates, np.ones(3)/3, mode='same')
                rate_ax.plot(idxs, ma, color='orange', linewidth=1.5, label='MA(3)')
            rate_ax.set_xticks(idxs if len(rates)<=30 else xt)
            rate_ax.set_ylabel("spikes/s"); rate_ax.grid(True, alpha=0.3)
            rate_ax.set_title("Firing rate over time - LIVE")
            rate_ax.legend(loc='upper right', fontsize=8)
            last = self.records[-1]
            v_hist = last.get("v_history")
            dom = last.get('dominant', '-')
            dom_sanitized = self._sanitize_text(dom)
            last_idx = len(self.records) - 1
            if last_idx >= 0:
                heat_ax.axvline(last_idx, color='white', linestyle='--', alpha=0.6, linewidth=2)
            heat_ax.text(0.99, 0.98, f"Last: {dom_sanitized}", transform=heat_ax.transAxes, ha='right', va='top', 
                        color='white', fontsize=9, bbox=dict(facecolor='black', alpha=0.6))
            if v_hist is not None and v_hist.size > 0:
                t = np.arange(v_hist.shape[0])
                nplot = min(12, v_hist.shape[1])
                cmap = plt.get_cmap('tab10')
                for i in range(nplot):
                    color = cmap(i % 10)
                    volt_ax.plot(t, v_hist[:, i], label=f"N{i}", color=color, alpha=0.9 if i%2==0 else 0.6, linewidth=0.8)
                mean_v = v_hist.mean(axis=1)
                volt_ax.plot(t, mean_v, color='k', linewidth=1.5, label='mean')
                volt_ax.legend(loc="upper right", fontsize=7, ncol=2)
            volt_ax.set_title("Most recent membrane potentials (sample neurons) - LIVE")
            volt_ax.set_xlabel("time step"); volt_ax.set_ylabel("voltage (a.u.)")
            volt_ax.grid(True, alpha=0.2)
            if mental_ax is not None:
                mh = [rec.get('mental_health', 0.0) for rec in self.records]
                idxs = list(range(1, len(mh)+1))
                mental_ax.plot(idxs, mh, marker='o', color='C3', linewidth=1.5, markersize=4)
                mental_ax.fill_between(idxs, mh, color='C3', alpha=0.15)
                mental_ax.set_ylim(0.0, 1.0)
                mental_ax.set_title('User mental health (risk) - LIVE')
                mental_ax.set_xlabel('Turn')
                mental_ax.set_ylabel('Risk (0..1)')
                mental_ax.grid(True, alpha=0.25)
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(0.01)
        except Exception as e:
            logger.debug(f"_update_plot error: {e}")
            try:
                if not plt.fignum_exists(self.fig.number):
                    self._graph_open = False
            except Exception:
                self._graph_open = False
    def process_pending_requests(self):
        Called from the GUI/main thread (e.g. from the visualizer animation update)
        to process any queued show_graph() requests coming from background threads.
        if not self._pending_show.is_set():
            return
        with self._show_lock:
            try:
                if self.fig is None:
                    self._create_figure()
                self._auto_update = True
                self._graph_open = True
                if self._graph_open_time is None:
                    self._graph_open_time = time.time()
                self._update_plot()
                try:
                    self.fig.show()
                    plt.show(block=False)
                    plt.draw()
                    plt.pause(0.1)
                    print("[Graph] Graph window opened successfully!")
                except Exception as e:
                    try:
                        plt.draw()
                        plt.pause(0.1)
                        print(f"[Graph] Window opened (fallback method)")
                    except Exception as e2:
                        logger.debug(f"Graph display error: {e2}")
            finally:
                self._pending_show.clear()
    def close_graph(self):
        if self._graph_open_time is not None:
            elapsed_time = time.time() - self._graph_open_time
            remaining_time = self._min_display_time - elapsed_time
            if remaining_time > 0:
                print(f"[Graph] Graph must stay open for at least 1 minute. Please wait {remaining_time:.1f} more seconds.")
                return False
        self._graph_open = False
        self._auto_update = False
        self._graph_open_time = None
        if self.fig is not None:
            if threading.current_thread() is threading.main_thread():
                try:
                    plt.close(self.fig)
                except Exception:
                    pass
            else:
                pass
            self.fig = None
        print("[Graph] Graph closed.")
        return True
class SimpleChatbot:
    def __init__(self, model_name=CHATBOT_MODEL_NAME, max_length=200):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers/torch not available")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.chat_history_ids = None
        self.max_length = max_length
    def chat(self, user_input):
        if not user_input:
            return ""
        new_input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors="pt").to(self.device)
        if self.chat_history_ids is not None:
            bot_input_ids = torch.cat([self.chat_history_ids, new_input_ids], dim=-1)
        else:
            bot_input_ids = new_input_ids
        with torch.no_grad():
            self.chat_history_ids = self.model.generate(
                bot_input_ids,
                max_length=min(self.max_length, bot_input_ids.shape[-1] + 100),
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True, top_k=50, top_p=0.95
            )
        response_ids = self.chat_history_ids[:, bot_input_ids.shape[-1]:]
        response = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)
        return response.strip()
    def reset(self):
        self.chat_history_ids = None
class NeuroVisualizer:
    def __init__(self, lif_pop, encoder, window_ms=OSCILLOSCOPE_WINDOW_MS, tracker=None):
        self.lif = lif_pop
        self.encoder = encoder
        self.window_ms = window_ms
        self.tracker = tracker if tracker is not None else None
        self.latest_lock = threading.Lock()
        self.latest_v_history = None
        self.latest_spike_times = None
        self.latest_tokens = None
        self.fig = None
        self.anim = None
        self._is_setup = False
    def update_latest(self, lif_pop, tokens):
        with self.latest_lock:
            self.latest_v_history = lif_pop.v_history.copy() if lif_pop.v_history is not None else None
            self.latest_spike_times = [list(s) for s in lif_pop.spike_times]
            self.latest_tokens = tokens
    def setup_plot(self):
        if self._is_setup:
            return
        plt.ion()
        self.fig = plt.figure(figsize=(14,8), constrained_layout=True)
        gs = self.fig.add_gridspec(3, 4)
        self.ax_raster = self.fig.add_subplot(gs[0, :3])
        self.ax_mem = self.fig.add_subplot(gs[1:, :3])
        self.ax_3d = self.fig.add_subplot(gs[:, 3])
        self.ax_raster.set_title("Spike raster (neurons) Ã¢â‚¬â€ all neurons")
        self.ax_raster.set_xlabel("time (ms)")
        self.ax_raster.set_ylabel("neuron index")
        self.ax_raster.set_xlim(0, self.window_ms)
        self.ax_raster.set_ylim(-0.5, self.lif.n - 0.5)
        self.ax_mem.set_title("Membrane potentials (ALL neurons) Ã¢â‚¬â€ oscilloscope view")
        self.ax_mem.set_xlabel("time (ms)")
        self.ax_mem.set_ylabel("neuron index -> voltage plotted as offset traces")
        self.ax_mem.set_xlim(0, self.window_ms)
        self.ax_mem.set_ylim(-1.0, self.lif.n + 1.0)
        self.ax_3d.set_title("Firing summary: time (x) vs neuron (y) (color=voltage)")
        self.ax_3d.set_xlabel("time (ms)")
        self.ax_3d.set_ylabel("neuron index")
        self.ax_3d.set_xlim(0, self.window_ms)
        self.ax_3d.set_ylim(-0.5, self.lif.n - 0.5)
        self.mem_lines = [self.ax_mem.plot([], [], linewidth=0.8)[0] for _ in range(self.lif.n)]
        self.raster_scatter = self.ax_raster.scatter([], [], s=8)
        self.summary_scatter = self.ax_3d.scatter([], [], c=[], s=12, cmap='viridis', vmin=-1.5, vmax=2.5)
        self.dom_text = self.ax_raster.text(0.99, 0.02, "Dominant: -", transform=self.ax_raster.transAxes,
                                            fontsize=10, ha='right', va='bottom', bbox=dict(facecolor="
        try:
            img = plt.imread(SAMPLE_IMAGE_PATH)
            ax_img = self.fig.add_axes([0.78, 0.02, 0.18, 0.18], anchor='SE')
            ax_img.imshow(img)
            ax_img.axis('off')
        except Exception:
            pass
        def init():
            for ln in self.mem_lines:
                ln.set_data([], [])
            self.raster_scatter.set_offsets(np.empty((0,2)))
            self.summary_scatter.set_offsets(np.empty((0,2)))
            try:
                self.summary_scatter.set_array(np.array([]))
            except Exception:
                pass
            self.dom_text.set_text("Dominant: -")
            return (self.raster_scatter, self.summary_scatter, *self.mem_lines, self.dom_text)
        def update(frame):
            with self.latest_lock:
                vhist = self.latest_v_history
                spt = self.latest_spike_times
                toks = self.latest_tokens
            if vhist is None or spt is None:
                return (self.raster_scatter, self.summary_scatter, *self.mem_lines, self.dom_text)
            n_steps = vhist.shape[0]
            t_axis = np.arange(n_steps) * DT
            xs=[]; ys=[]
            for n_idx, times_list in enumerate(spt):
                for t in times_list:
                    xs.append(t); ys.append(n_idx)
            if len(xs)>0:
                self.raster_scatter.set_offsets(np.column_stack((xs, ys)))
            else:
                self.raster_scatter.set_offsets(np.empty((0,2)))
            gain = 0.6
            for i in range(self.lif.n):
                y_trace = i + (vhist[:, i] * gain)
                self.mem_lines[i].set_data(t_axis, y_trace)
                self.mem_lines[i].set_color('C0')
                self.mem_lines[i].set_alpha(0.9 if i%2==0 else 0.6)
            spike_x=[]; spike_y=[]; spike_z=[]
            for n_idx, times_list in enumerate(spt):
                for t in times_list:
                    spike_x.append(t); spike_y.append(n_idx); spike_z.append(1.2)
            sample_stride = max(1, int(max(1, n_steps // 200)))
            volt_x=[]; volt_y=[]; volt_z=[]
            for ti in range(0, n_steps, sample_stride):
                for ni in range(0, self.lif.n, max(1, int(self.lif.n // 80))):
                    volt_x.append(t_axis[ti]); volt_y.append(ni); volt_z.append(vhist[ti, ni])
            if spike_x or volt_x:
                xs3 = np.array(spike_x + volt_x)
                ys3 = np.array(spike_y + volt_y)
                zs3 = np.array(spike_z + volt_z)
                offs = np.column_stack((xs3, ys3))
                try:
                    self.summary_scatter.set_offsets(offs)
                    self.summary_scatter.set_array(zs3)
                    self.summary_scatter.set_alpha(0.8)
                except Exception:
                    self.summary_scatter.set_offsets(np.empty((0,2)))
                    try:
                        self.summary_scatter.set_array(np.array([]))
                    except Exception:
                        pass
            dom_label = "-"
            try:
                if toks:
                    joined = " ".join(toks)
                    vecp, dom_label, _meta = prototype_detector.predict(joined)
                else:
                    dom_label = "-"
            except Exception as ex:
                dom_label = "-"
            self.dom_text.set_text(f"Dominant: {dom_label}")
            self.dom_text.set_bbox(dict(facecolor="
            self.ax_raster.set_xlim(0, self.window_ms)
            self.ax_mem.set_xlim(0, self.window_ms)
            self.ax_3d.set_xlim(0, self.window_ms)
            try:
                if getattr(self, 'tracker', None) is not None:
                    try:
                        self.tracker.process_pending_requests()
                        if hasattr(self.tracker, '_graph_open') and self.tracker._graph_open:
                            if not hasattr(self, '_frame_count'):
                                self._frame_count = 0
                            self._frame_count += 1
                            if self._frame_count % 10 == 0:
                                try:
                                    self.tracker._update_plot()
                                except Exception:
                                    pass
                    except Exception:
                        pass
            except Exception:
                pass
            return (self.raster_scatter, self.summary_scatter, *self.mem_lines, self.dom_text)
    def start(self, stop_event=None):
        Keep the GUI alive on main thread until stop_event is set.
        Must be called on the main thread.
        if not self._is_setup:
            self.setup_plot()
        try:
            plt.show(block=True)
            if stop_event is not None:
                try:
                    stop_event.set()
                except Exception:
                    pass
            return
        except Exception:
            try:
                if stop_event is None:
                    plt.show(block=True)
                    return
                while not stop_event.is_set():
                    plt.pause(0.1)
                try:
                    plt.close('all')
                except Exception:
                    pass
            except Exception as e:
                print(f"[Visualizer.start] fallback error: {e}")
class NeuroNovaEngineSemantic:
    def __init__(self):
        self.encoder = SpikingEncoder(window_ms=WINDOW_MS, channels=N_CHANNELS, seed=11)
        self.prototype_detector = prototype_detector
        self.embedding_detector = embedding_detector
        self.lif = LIFPopulation(n_neurons=N_NEURONS, n_channels=N_CHANNELS, dt=DT, tau_m=30.0, v_thresh=1.0, seed=7)
        self.decider = DecisionModule(rate_threshold=8.0)
        self.respgen = ResponseGenerator(seed=42)
        self.tracker = ConversationTracker(EMOTIONS)
        self.chatbot = None
        if TRANSFORMERS_AVAILABLE:
            try:
                self.chatbot = SimpleChatbot(model_name=CHATBOT_MODEL_NAME)
            except Exception:
                self.chatbot = None
        self.GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY') or "AIzaSyD3HXJRZJkt0XZR1pJ-_chZhYlr4ywGiMU"
        self.remote_chat_enabled = bool(self.GEMINI_API_KEY)
        self.conversation_history = []
        self.insult_count = 0
        self.insult_threshold = INSULT_THRESHOLD
        self.insult_words = set(w.lower() for w in INSULT_WORDS)
    def remote_chat(self, user_input, dominant_emotion="Neutral", max_tokens=200):
        if not self.GEMINI_API_KEY:
            return ""
        try:
            import json, urllib.request
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={self.GEMINI_API_KEY}"
            
            system_context = ("You are a warm, empathetic AI assistant integrated with NeuroNovaEngine emotion detection system. "
                           "The system detected the user's current emotion as: " + dominant_emotion + ". "
                           "Respond naturally and conversationally, matching their emotional tone. "
                           "Be empathetic, supportive, and engaging. Keep responses natural and flowing (2-4 sentences). "
                           "Reference previous conversation when relevant to maintain continuity.")
            
            conversation_context = ""
            if len(self.conversation_history) > 0:
                recent_history = self.conversation_history[-4:]
                conversation_context = "\n\nRecent conversation context:\n"
                for entry in recent_history:
                    conversation_context += f"User: {entry.get('user', '')}\nAssistant: {entry.get('assistant', '')}\n"
            
            full_prompt = system_context + conversation_context + "\n\nUser: " + user_input + "\nAssistant:"
            
            body = {
                "contents": [{
                    "parts": [{"text": full_prompt}]
                }],
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": 0.8,
                    "topP": 0.9,
                    "topK": 50
                }
            }
            data = json.dumps(body).encode('utf-8')
            req = urllib.request.Request(url, data=data, headers={
                'Content-Type': 'application/json'
            })
            with urllib.request.urlopen(req, timeout=25) as resp:
                j = json.load(resp)
            candidates = j.get('candidates', [])
            if not candidates:
                return ""
            parts = candidates[0].get('content', {}).get('parts', [])
            if not parts:
                return ""
            msg = parts[0].get('text', '')
            response = (msg or "").strip()
            
            self.conversation_history.append({
                "user": user_input,
                "assistant": response,
                "emotion": dominant_emotion
            })
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            return response
        except Exception as e:
            logger.debug(f"gemini_chat error: {e}")
            return ""
    def detect_emotion(self, text):
        toks = tokenize(text)
        if not toks:
            logger.debug("detect_emotion: empty or no tokens after tokenize")
            return np.zeros(len(EMOTIONS)), "Neutral", {}
        text = text.strip()
        low = text.lower()
        suicidal_phrases = ["i want to die","kill myself","end my life","i cant go on","i'm going to kill myself"]
        for pat in suicidal_phrases:
            if pat in low:
                vec = np.zeros(len(EMOTIONS), dtype=float)
                for e in ("Sadness","Fear","Empathic Pain"):
                    vec[EMOTIONS.index(e)] = 1.0
                vec = vec / (vec.max()+1e-12)
                return vec, "Sadness", {"self_harm_risk": True, "pattern": pat}
        if EMBED_AVAILABLE and self.embedding_detector is not None:
            vec, dom, meta = self.embedding_detector.predict(text)
            logger.debug(f"detect_emotion: embedding detector result dom={dom} meta={meta}")
            return vec, dom, meta
        insult_hits = 0
        try:
            lw = low
            for iw in self.insult_words:
                if iw in lw:
                    insult_hits += lw.count(iw)
        except Exception:
            insult_hits = 0
        logger.debug(f"detect_emotion: insult_hits={insult_hits} current_insult_count={getattr(self,'insult_count',0)}")
        if insult_hits > 0:
            self.insult_count += insult_hits
            meta = {"insult_count": self.insult_count}
            if self.insult_count >= self.insult_threshold:
                vec = np.zeros(len(EMOTIONS), dtype=float)
                anger_idx = EMOTIONS.index("Anger")
                scale = min(1.0, 0.5 + 0.2 * (self.insult_count - self.insult_threshold))
                vec[anger_idx] = scale
                vec = vec / (vec.max() + 1e-12)
                meta["anger_override"] = True
                logger.info(f"detect_emotion: anger_override triggered (insult_count={self.insult_count})")
                return vec, "Anger", meta
        vec, dom, proto_meta = self.prototype_detector.predict(text)
        logger.debug(f"detect_emotion: prototype_detector dom={dom}")
        if insult_hits > 0:
            if not isinstance(proto_meta, dict):
                proto_meta = {}
            proto_meta["insult_count"] = getattr(self, 'insult_count', insult_hits)
        return vec, dom, proto_meta
    def process_text(self, text):
        print("\nInput text:")
        print(textwrap.fill(text, width=80))
        normalized = normalize_text(text)
        if normalized != text:
            print(f"\n[NORMALIZED] {normalized}")
        logger.debug(f"process_text: original='{text}' normalized='{normalized}'")
        vec, dominant, meta = self.detect_emotion(normalized)
        print("\nTop emotion scores (0..1):")
        sorted_idx = np.argsort(-vec)[:6]
        for idx in sorted_idx:
            print(f"  {EMOTIONS[idx]:<22s}: {vec[idx]:.3f}  (color: {EMOTION_COLORS.get(EMOTIONS[idx],'
        dominant_label = "Neutral" if np.max(vec) <= 0 else dominant
        print(f"\nDominant emotion: {dominant_label}  (color: {EMOTION_COLORS.get(dominant_label,'
        if meta:
            print(f"[Meta] {meta}")
        channels, tokens = self.encoder.encode(normalized)
        print(f"\n[SpikingEncoder] tokens: {tokens}")
        for i, ch in enumerate(channels):
            if ch:
                print(f"  channel {i}: spike times (ms) -> {np.array(ch).round(1).tolist()}")
        rates = self.lif.run(channels, t_max_ms=self.encoder.window_ms)
        print(f"\n[LIFPopulation] mean firing rate: {np.mean(rates):.2f} spikes/s  (per neuron)")
        decision = self.decider.decide(rates, dominant_label, meta)
        print(f"[Decision] action: {decision['action']}, tone: {decision['tone']}, mean_rate: {decision['mean_rate']:.2f}")
        reply = ""
        if getattr(self, 'remote_chat_enabled', False) and not meta.get('self_harm_risk', False):
            try:
                reply = self.remote_chat(normalized, dominant_emotion=dominant_label, max_tokens=200)
            except Exception:
                reply = ""
        if not reply:
            reply = self.respgen.generate(decision['action'], decision['tone'], normalized, dominant_label, meta)
        logger.info(f"process_text: dominant={dominant_label} meta={meta} decision={decision} reply='{reply}'")
        print(f"\nRobot> {reply}\n")
        try:
            self.tracker.add_entry(
                user_text=text,
                emotion_vec=vec,
                dominant=dominant_label,
                decision=decision,
                meta=meta,
                lif=self.lif,
                tokens=tokens
            )
        except Exception as e:
            print(f"[Tracker warn] {e}")
        return {"vec": vec, "dominant": dominant_label, "meta": meta, "decision": decision, "reply": reply, "tokens": tokens}
class NeuroNovaCLI:
    def __init__(self, engine, visualizer, stop_event=None):
        self.engine = engine
        self.visualizer = visualizer
        self.stop_event = stop_event or threading.Event()
        self.running = True
    def process_text(self, text):
        res = self.engine.process_text(text)
        if self.visualizer is not None:
            self.visualizer.update_latest(self.engine.lif, res.get("tokens", []))
        return res
    def repl(self):
        print("NeuroNova semantic visual demo")
        print("Commands: exit / quit / history / graph / close graph / bot / plot on / plot off / insults / reset insults")
        print("Tip: Type 'graph' to open a live updating graph that shows emotion reactions in real-time!")
        while self.running:
            try:
                s = input("You> ").strip()
            except (KeyboardInterrupt, EOFError):
                s = "exit"
            if not s:
                continue
            cmd = s.lower().strip()
            if cmd in ("exit", "quit"):
                print("Exiting...")
                self.running = False
                try:
                    self.stop_event.set()
                except Exception:
                    pass
                try: 
                    self.engine.tracker.close_graph()
                    plt.close('all')
                except Exception: pass
                break
            if cmd == "history":
                self.engine.tracker.print_history(); continue
            if cmd == "graph":
                self.engine.tracker.show_graph(); continue
            if cmd in ("close graph", "graph close"):
                self.engine.tracker.close_graph()
                print("[Graph] Closed. Type 'graph' to reopen.")
                continue
            if cmd == "insults":
                print(f"[Insults] current insult count: {getattr(self.engine, 'insult_count', 0)}")
                continue
            if cmd == "reset insults":
                try:
                    self.engine.insult_count = 0
                    print("[Insults] insult counter reset to 0")
                except Exception:
                    print("[Insults] unable to reset")
                continue
            if cmd == "bot":
                if getattr(self.engine, 'remote_chat_enabled', False):
                    print("Remote chatbot session started (GEMINI_API_KEY). Type 'back' to return.")
                    while True:
                        try:
                            u = input("BotUser> ").strip()
                        except (KeyboardInterrupt, EOFError):
                            break
                        if not u: continue
                        if u.lower() in ("back", "exit", "quit"):
                            break
                        resp = self.engine.remote_chat(u)
                        print("Bot> " + (resp or "(no reply)"))
                elif self.engine.chatbot is None:
                    print("[Chatbot unavailable] Install transformers & torch to enable or set GEMINI_API_KEY.")
                else:
                    print("SimpleChatbot session started. Type 'back' to return.")
                    self.engine.chatbot.reset()
                    while True:
                        try:
                            u = input("BotUser> ").strip()
                        except (KeyboardInterrupt, EOFError):
                            break
                        if not u: continue
                        if u.lower() in ("back", "exit", "quit"):
                            break
                        resp = self.engine.chatbot.chat(u)
                        print("Bot> " + resp)
                continue
            if cmd == "plot off":
                print("[Plotting disabled] Closing windows."); 
                try: plt.close('all') 
                except Exception: pass
                continue
            if cmd == "plot on":
                print("[Plotting enabled] (if window closed, restart script to recreate visualizer)"); continue
            self.process_text(s)
def main():
    engine = NeuroNovaEngineSemantic()
    visualizer = NeuroVisualizer(engine.lif, engine.encoder, window_ms=OSCILLOSCOPE_WINDOW_MS, tracker=engine.tracker)
    stop_event = threading.Event()
    cli = NeuroNovaCLI(engine, visualizer, stop_event=stop_event)
    cli_thread = threading.Thread(target=cli.repl, daemon=True)
    cli_thread.start()
    try:
        plt.ion()
        while not stop_event.is_set():
            try:
                if visualizer.tracker is not None:
                    visualizer.tracker.process_pending_requests()
            except Exception:
                pass
            try:
                plt.pause(0.05)
            except Exception:
                pass
            if not cli_thread.is_alive():
                break
    except KeyboardInterrupt:
        stop_event.set()
    except Exception as e:
        print(f"[Main loop error] {e}")
    try:
        if cli_thread.is_alive():
            stop_event.set()
            cli.running = False
            cli_thread.join(timeout=1.0)
    except Exception:
        pass
    print("Shutdown complete.")
if __name__ == "__main__":
    main()
neuronova_full_updated.py
Combined NeuroNovaEngine:
 - Semantic 27-emotion detector (prototype + optional sentence-transformer embeddings)
 - LIF spiking population
 - Spike raster, full-membrane oscilloscope, and 3D firing visualization (animated)
 - Conversation tracker, chatbot (optional) with safer deterministic generation,
   attention_mask use, pad token fallback, and emotion-aware system prompt
 - CLI orchestrator with main-thread plotting
Run:
    pip install numpy matplotlib
Optional (better semantics):
    pip install sentence-transformers
Optional (chatbot):
    pip install transformers torch
import os
import sys
import math
import time
import random
import threading
import textwrap
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
EMBED_AVAILABLE = False
EMBED_MODEL = None
try:
    from sentence_transformers import SentenceTransformer
    EMBED_AVAILABLE = True
except Exception:
    EMBED_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False
SAMPLE_IMAGE_PATH = "/mnt/data/A_two-part_digital_plot_illustrates_neural_activit.png"
WINDOW_MS = 300
OSCILLOSCOPE_WINDOW_MS = 300
ANIM_FPS = 30
PLOT_UPDATE_INTERVAL_MS = int(1000 / ANIM_FPS)
N_NEURONS = 32
N_CHANNELS = 8
DT = 1.0
USE_EMBEDDINGS = False
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
ENABLE_LLM_FALLBACK = False
CHATBOT_MODEL_NAME = "gpt2"
EMOTIONS = [
    "Admiration", "Adoration", "Aesthetic Appreciation", "Amusement", "Anger",
    "Anxiety", "Awe", "Awkwardness", "Boredom", "Calmness",
    "Confusion", "Craving", "Disappointment", "Disgust", "Empathic Pain",
    "Entrancement", "Excitement", "Fear", "Horror", "Interest",
    "Joy", "Nostalgia", "Romance", "Sadness", "Satisfaction",
    "Sexual Desire", "Sympathy"
]
EMOTION_COLORS = {
    "Admiration": "
    "Amusement": "
    "Awkwardness": "
    "Craving": "
    "Entrancement": "
    "Interest": "
    "Sadness": "
}
EMOTION_PROTOTYPES = {
    "Admiration": ["I'm impressed by your skill", "That performance filled me with respect"],
    "Adoration": ["I feel deep affection for that person", "This is so lovable and warm"],
    "Aesthetic Appreciation": ["That is incredibly beautiful art", "The scenery moved me with its beauty"],
    "Amusement": ["That's funny and makes me laugh", "I find this amusing and humorous"],
    "Anger": ["I'm furious and upset about this", "That makes me so angry"],
    "Anxiety": ["I feel nervous and worried", "I'm anxious about what will happen"],
    "Awe": ["I am filled with wonder and amazement", "This leaves me speechless with awe"],
    "Awkwardness": ["This is embarrassing and I feel awkward", "I cringe and feel socially uncomfortable"],
    "Boredom": ["I'm bored and uninterested", "This is dull and I'm not engaged"],
    "Calmness": ["I feel relaxed and peaceful", "Everything is calm and serene"],
    "Confusion": ["I don't understand what's happening", "This is confusing and unclear to me"],
    "Craving": ["I really want that now", "I crave and desire it strongly"],
    "Disappointment": ["I'm disappointed and let down", "That did not meet my expectations"],
    "Disgust": ["That is disgusting and revolting", "I feel repulsed by that"],
    "Empathic Pain": ["I feel pain for someone else's suffering", "That hurts me to hear about them"],
    "Entrancement": ["I'm mesmerized and completely absorbed", "I am entranced by the experience"],
    "Excitement": ["I'm so excited and thrilled", "I can't wait Ã¢â‚¬â€ I'm pumped"],
    "Fear": ["I'm scared and afraid", "I feel threatened and fearful"],
    "Horror": ["This is horrifying and sickening", "I feel terrified and disgusted"],
    "Interest": ["I'm curious and want to learn more", "This sparks my interest and attention"],
    "Joy": ["I'm happy and full of joy", "That brings me delight and pleasure"],
    "Nostalgia": ["This reminds me of old memories", "I feel bittersweet remembering the past"],
    "Romance": ["I feel warm romantic love", "This feels tender and romantic"],
    "Sadness": ["I'm sad and full of sorrow", "This makes me feel down and lonely"],
    "Satisfaction": ["I'm content and satisfied", "That leaves me feeling fulfilled"],
    "Sexual Desire": ["I feel sexual attraction and desire", "This arouses me physically"],
    "Sympathy": ["I'm sorry to hear that and I care", "I feel compassion for that person"]
}
def tokenize(text):
    if text is None:
        return []
    t = text.lower()
    t = t.replace("i'm", "im").replace("it's", "its").replace("don't", "dont").replace("can't", "cant")
    toks = []
    for w in t.split():
        w2 = ''.join(ch for ch in w if ch.isalnum())
        if w2:
            toks.append(w2)
    return toks
def math_log(x):
    import math
    return math.log(1 + x)
if USE_EMBEDDINGS and EMBED_AVAILABLE:
    try:
        EMBED_MODEL = SentenceTransformer(EMBEDDING_MODEL)
        prototype_texts = []
        prototype_map = []
        for emo, texts in EMOTION_PROTOTYPES.items():
            for t in texts:
                prototype_texts.append(t)
                prototype_map.append(emo)
        prototype_embs = EMBED_MODEL.encode(prototype_texts, convert_to_numpy=True, show_progress_bar=False)
    except Exception:
        EMBED_MODEL = None
        EMBED_AVAILABLE = False
        prototype_embs = None
else:
    prototype_texts = []
    prototype_map = []
    prototype_embs = None
class PrototypeSemanticDetector:
    def __init__(self, prototypes_map):
        self.prototypes_map = {k: [p.lower() for p in v] for k, v in prototypes_map.items()}
        vocab = set()
        for proto_list in self.prototypes_map.values():
            for s in proto_list:
                vocab.update(tokenize(s))
        self.vocab = sorted(vocab)
        self.word_to_idx = {w:i for i,w in enumerate(self.vocab)}
        self.prototype_vecs = {}
        for emo, proto_list in self.prototypes_map.items():
            vecs = [self._text_vector(s) for s in proto_list]
            avg = np.mean(vecs, axis=0) if len(vecs)>0 else np.zeros(len(self.vocab))
            norm = np.linalg.norm(avg)
            self.prototype_vecs[emo] = (avg / (norm + 1e-12)) if norm>0 else avg
    def _text_vector(self, text):
        toks = tokenize(text)
        if not toks:
            return np.zeros(len(self.vocab), dtype=float)
        counts = Counter(toks)
        vec = np.zeros(len(self.vocab), dtype=float)
        for w, c in counts.items():
            idx = self.word_to_idx.get(w)
            if idx is not None:
                vec[idx] = 1.0 + math_log(c)
        norm = np.linalg.norm(vec)
        return (vec / (norm + 1e-12)) if norm>0 else vec
    def predict(self, text):
        toks = tokenize(text)
        if not toks:
            return np.zeros(len(EMOTIONS)), "Neutral", {}
        text_vec = self._text_vector(text)
        sims = np.zeros(len(EMOTIONS), dtype=float)
        for i, emo in enumerate(EMOTIONS):
            proto = self.prototype_vecs.get(emo, np.zeros(len(self.vocab)))
            sims[i] = max(0.0, float(np.dot(text_vec, proto)))
        maxv = sims.max()
        if maxv > 0:
            sims = sims / (maxv + 1e-12)
            dominant = EMOTIONS[int(np.argmax(sims))]
        else:
            dominant = "Neutral"
        return sims, dominant, {}
class EmbeddingSemanticDetector:
    def __init__(self, embed_model, prototype_texts, prototype_map, prototype_embs):
        self.model = embed_model
        self.prototype_texts = prototype_texts
        self.prototype_map = prototype_map
        self.prototype_embs = prototype_embs
        self.emotions = EMOTIONS
    def predict(self, text):
        if not text or text.strip()=="":
            return np.zeros(len(self.emotions)), "Neutral", {}
        try:
            q_emb = self.model.encode([text], convert_to_numpy=True)[0]
        except Exception:
            return np.zeros(len(self.emotions)), "Neutral", {}
        q_norm = np.linalg.norm(q_emb) + 1e-12
        q_emb_n = q_emb / q_norm
        prot = self.prototype_embs
        prot_norms = np.linalg.norm(prot, axis=1) + 1e-12
        prot_n = prot / prot_norms[:, None]
        sims = np.dot(prot_n, q_emb_n)
        emo_scores = {emo:0.0 for emo in self.emotions}
        for sim_val, emo in zip(sims, self.prototype_map):
            if sim_val > emo_scores[emo]:
                emo_scores[emo] = float(sim_val)
        vec = np.array([emo_scores.get(e, 0.0) for e in self.emotions], dtype=float)
        maxv = vec.max()
        if maxv > 0:
            vec = vec / (maxv + 1e-12)
            dominant = self.emotions[int(np.argmax(vec))]
        else:
            dominant = "Neutral"
        return vec, dominant, {}
prototype_detector = PrototypeSemanticDetector(EMOTION_PROTOTYPES)
embedding_detector = None
if EMBED_AVAILABLE and prototype_embs is not None:
    embedding_detector = EmbeddingSemanticDetector(EMBED_MODEL, prototype_texts, prototype_map, prototype_embs)
class SpikingEncoder:
    def __init__(self, window_ms=WINDOW_MS, channels=N_CHANNELS, seed=1):
        self.window_ms = window_ms
        self.channels = channels
        self.rng = np.random.RandomState(seed)
    def token_salience(self, token):
        vowel_frac = sum(ch in 'aeiou' for ch in token) / max(1, len(token))
        uncommon = sum(ch in 'qzx' for ch in token)
        return len(token) * (1 + vowel_frac) + 2*uncommon
    def encode(self, text):
        tokens = tokenize(text)
        if not tokens:
            return [[] for _ in range(self.channels)], tokens
        saliences = np.array([self.token_salience(t) for t in tokens], dtype=float)
        if len(saliences) > 1:
            sal_range = np.ptp(saliences)
            sal_range = max(1e-9, sal_range)
            saliences = (saliences - saliences.min()) / sal_range
        else:
            saliences = np.array([0.5])
        channel_trains = [[] for _ in range(self.channels)]
        for i, token in enumerate(tokens):
            ch = (sum(ord(c) for c in token) + i) % self.channels
            t_ms = (1.0 - saliences[i]) * (self.window_ms * 0.8) + self.rng.normal(0, 6.0)
            t_ms = float(max(0.0, min(self.window_ms, t_ms)))
            channel_trains[ch].append(t_ms)
        for ch in range(self.channels):
            channel_trains[ch].sort()
        return channel_trains, tokens
class LIFPopulation:
    def __init__(self, n_neurons=N_NEURONS, n_channels=N_CHANNELS, dt=DT, tau_m=20.0, v_reset=0.0, v_thresh=1.0, seed=2):
        self.n = n_neurons; self.channels = n_channels; self.dt = dt; self.tau = tau_m
        self.v_reset = v_reset; self.v_thresh = v_thresh
        self.rng = np.random.RandomState(seed)
        self.W = self.rng.normal(0.8, 0.3, size=(self.n, self.channels)).clip(0, 2.0)
        self.v = np.zeros(self.n, dtype=float)
        self.spike_times = [[] for _ in range(self.n)]
        self.v_history = np.zeros((0, self.n), dtype=float)
    def reset(self):
        self.v[:] = 0.0
        self.spike_times = [[] for _ in range(self.n)]
        self.v_history = np.zeros((0, self.n), dtype=float)
    def run(self, channel_trains, t_max_ms=WINDOW_MS):
        self.reset()
        n_steps = int(math.ceil(t_max_ms / self.dt))
        spikes = np.zeros((self.channels, n_steps), dtype=bool)
        for ch_idx, times in enumerate(channel_trains):
            for t in times:
                step = int(round(t / self.dt))
                if 0 <= step < n_steps: spikes[ch_idx, step] = True
        v_hist = np.zeros((n_steps, self.n), dtype=float)
        for step in range(n_steps):
            inp = (spikes[:, step].astype(float) @ self.W.T)
            dv = (-self.v / self.tau + inp) * (self.dt)
            self.v += dv
            just_fired = self.v >= self.v_thresh
            if just_fired.any():
                idxs = np.where(just_fired)[0]
                for idx in idxs:
                    self.spike_times[idx].append(step * self.dt)
                    self.v[idx] = self.v_reset
            v_hist[step, :] = self.v
        self.v_history = v_hist
        rates = np.array([len(times) / (t_max_ms / 1000.0) for times in self.spike_times], dtype=float)
        return rates
class DecisionModule:
    def __init__(self, rate_threshold=10.0):
        self.rate_threshold = rate_threshold
        self.positive_set = {"Admiration","Adoration","Aesthetic Appreciation","Amusement","Awe","Calmness","Excitement","Interest","Joy","Satisfaction","Romance"}
        self.negative_set = {"Anger","Anxiety","Confusion","Disappointment","Disgust","Empathic Pain","Fear","Horror","Sadness","Boredom"}
    def decide(self, firing_rates, dominant_emotion, meta=None):
        meta = meta or {}
        mean_rate = float(np.mean(firing_rates))
        if meta.get("self_harm_risk", False):
            return {"mean_rate": mean_rate, "action":"take_action", "tone":"urgent", "dominant": dominant_emotion, "meta": meta}
        if dominant_emotion == "Neutral":
            if mean_rate < self.rate_threshold * 0.6:
                return {"mean_rate": mean_rate, "action":"idle", "tone":"calm", "dominant":"Neutral", "meta": meta}
            elif mean_rate < self.rate_threshold:
                return {"mean_rate": mean_rate, "action":"observe", "tone":"curious", "dominant":"Neutral", "meta": meta}
            else:
                return {"mean_rate": mean_rate, "action":"speak_serious", "tone":"direct", "dominant":"Neutral", "meta": meta}
        if dominant_emotion in self.positive_set:
            return {"mean_rate": mean_rate, "action":"speak_friendly", "tone":"warm", "dominant": dominant_emotion, "meta": meta}
        if dominant_emotion in self.negative_set:
            if dominant_emotion == "Boredom":
                return {"mean_rate": mean_rate, "action":"engage_user", "tone":"uplifting", "dominant": dominant_emotion, "meta": meta}
            if mean_rate > self.rate_threshold * 1.2:
                return {"mean_rate": mean_rate, "action":"take_action", "tone":"urgent", "dominant": dominant_emotion, "meta": meta}
            return {"mean_rate": mean_rate, "action":"speak_serious", "tone":"concerned", "dominant": dominant_emotion, "meta": meta}
        return {"mean_rate": mean_rate, "action":"observe", "tone":"curious", "dominant": dominant_emotion, "meta": meta}
class ResponseGenerator:
    def __init__(self, seed=123):
        self.rng = random.Random(seed)
        self.suggestions = [
            "Let's play a quick word game Ã¢â‚¬â€ I start with a word, you reply with the first thing that comes to your mind.",
            "Want to hear a joke or an interesting fact? (type 'joke' or 'fact')",
            "I can give you a short brain teaser in 30 seconds Ã¢â‚¬â€ want to try?",
            "Tell me a topic you like Ã¢â‚¬â€ I'll give a fun fact or a mini-quiz.",
            "How about a short breathing exercise or a 2-minute stretch? I'll guide you."
        ]
        self.jokes = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "I told my computer I needed a break, and it said: 'No problem Ã¢â‚¬â€ I'll go to sleep.'"
        ]
        self.facts = [
            "Octopuses have three hearts.",
            "Honey never spoils Ã¢â‚¬â€ jars from ancient Egypt are still edible."
        ]
    def generate(self, action, tone, user_text, dominant_emotion, meta=None):
        meta = meta or {}
        ut = (user_text or "").strip()
        ut_l = ut.lower()
        if meta.get("self_harm_risk", False):
            return ("I'm really sorry you're feeling this way. If you're thinking about harming yourself or are in immediate danger, "
                    "please contact local emergency services or a crisis line right now. If you'd like, we can stay here and talk â€” are you safe right now?")
        if dominant_emotion == "Sexual Desire":
            return "I can't engage in sexual content. If you'd like, we can talk about relationships, boundaries, or something light-hearted instead."
        if any(kw in ut_l for kw in ("tell me something", "tell me", "can you", "could you", "what", "who", "how", "why", "?")):
            if "joke" in ut_l:
                return self.rng.choice(self.jokes)
            if "fact" in ut_l or "something" in ut_l:
                return self.rng.choice(self.facts)
            if len(ut_l.split()) < 6:
                return "Sure â€” could you give me a topic? Or I can share a fun fact: " + self.rng.choice(self.facts)
            return "Tell me a bit more about what you want, or I can share a quick fact or joke."
        if action == "engage_user":
            if "joke" in ut_l:
                return self.rng.choice(self.jokes)
            if "fact" in ut_l:
                return self.rng.choice(self.facts)
            return "You said you're bored â€” " + self.rng.choice(self.suggestions)
        if action == "speak_friendly":
            return "Nice! That sounds great â€” want to continue or try something fun?"
        if action == "speak_serious":
            return "I understand. Would you like help or someone to talk to?"
        if action == "take_action":
            return "I'm taking this seriously and prioritizing immediate help. Are you safe right now?"
        if action == "observe":
            return self.rng.choice([
                "Oh â€” what made you feel that way? Tell me more.",
                "I get that â€” would you like a fun fact or we can chat about something you enjoy?",
                "Interesting. Want me to share a quick idea to try right now?"
            ])
        if action == "idle":
            return "Okay â€” I'm here if you need anything."
        return "Alright."
class ConversationTracker:
    def __init__(self, emotions):
        self.emotions = emotions
        self.records = []
        self.fig = None
        self.axes = {}
        self.heat_cbar = None
        self._pending_show = threading.Event()
        self._show_lock = threading.Lock()
        plt.ion()
    def add_entry(self, user_text, emotion_vec, dominant, decision, meta, lif, tokens):
        neg_list = ["Sadness", "Fear", "Empathic Pain", "Anxiety", "Anger", "Disgust", "Horror"]
        try:
            neg_score = float(np.sum([emotion_vec[EMOTIONS.index(e)] for e in neg_list if e in EMOTIONS]))
        except Exception:
            neg_score = 0.0
        meta_safe = meta or {}
        if meta_safe.get("self_harm_risk", False):
            neg_score = max(neg_score, 1.0)
        neg_score = float(max(0.0, min(1.0, neg_score)))
        self.records.append({
            "text": user_text,
            "vec": emotion_vec.copy(),
            "dominant": dominant,
            "decision": decision,
            "meta": meta_safe,
            "mean_rate": decision.get("mean_rate", 0.0),
            "tokens": tokens,
            "spikes": [lst[:] for lst in lif.spike_times],
            "v_history": lif.v_history.copy() if lif.v_history is not None else None,
            "mental_health": neg_score
        })
        if len(self.records) > 200:
            self.records = self.records[-200:]
    def print_history(self):
        if not self.records:
            print("[History] No conversation yet.")
            return
        print("\n=== Conversation History ===")
        for idx, rec in enumerate(self.records, 1):
            meta = rec["meta"]
            meta_txt = f" meta={meta}" if meta else ""
            print(f"{idx:02d}. User: {rec['text']}")
            print(f"    Dominant: {rec['dominant']}  mean_rate: {rec['mean_rate']:.2f}{meta_txt}")
        print("============================\n")
    def show_graph(self):
        if not self.records:
            print("[Graph] No data yet.")
            return
        if threading.current_thread() is threading.main_thread():
            if self.fig is None:
                self._create_figure()
            self._update_plot()
            plt.show(block=True)
            return
        self._pending_show.set()
        print("[Graph] Request queued; will open the graph on the GUI thread.")
    def _create_figure(self):
        fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
        self.fig = fig
        ax_heat = axes[0, 0]
        ax_mental = axes[0, 1]
        ax_rate = axes[1, 0]
        ax_volt = axes[2, 0]
        try:
            axes[1,1].axis('off')
            axes[2,1].axis('off')
        except Exception:
            pass
        self.axes["heat"] = ax_heat
        self.axes["mental"] = ax_mental
        self.axes["rate"] = ax_rate
        self.axes["volt"] = ax_volt
    def _update_plot(self):
        if not self.records:
            return
        heat_ax = self.axes["heat"]
        mental_ax = self.axes.get("mental")
        rate_ax = self.axes["rate"]
        volt_ax = self.axes["volt"]
        heat_ax.clear(); rate_ax.clear(); volt_ax.clear()
        data = np.stack([rec["vec"] for rec in self.records], axis=1)
        im = heat_ax.imshow(data, aspect="auto", cmap="viridis", interpolation="nearest", vmin=0.0, vmax=1.0)
        heat_ax.set_yticks(range(len(self.emotions)))
        heat_ax.set_yticklabels(self.emotions, fontsize=8)
        heat_ax.set_xticks(range(len(self.records)))
        heat_ax.set_xticklabels([str(i+1) for i in range(len(self.records))])
        heat_ax.set_title("Emotion activation (conversation timeline)")
        heat_ax.set_xlabel("Turn")
        if self.heat_cbar is not None:
            self.heat_cbar.remove()
        self.heat_cbar = self.fig.colorbar(im, ax=heat_ax, fraction=0.015, pad=0.02)
        rates = [rec["mean_rate"] for rec in self.records]
        rate_ax.plot(range(1, len(rates)+1), rates, marker="o")
        rate_ax.set_xticks(range(1, len(rates)+1)); rate_ax.set_ylabel("spikes/s"); rate_ax.grid(True, alpha=0.3)
        last = self.records[-1]
        v_hist = last.get("v_history")
        if v_hist is not None and v_hist.size:
            t = np.arange(v_hist.shape[0])
            nplot = min(6, v_hist.shape[1])
            for i in range(nplot):
                volt_ax.plot(t, v_hist[:, i], label=f"Neuron {i}")
            volt_ax.legend(loc="upper right", fontsize=8)
        volt_ax.set_title("Most recent membrane potentials (sample neurons)")
        volt_ax.set_xlabel("time step"); volt_ax.set_ylabel("voltage (a.u.)")
        if mental_ax is not None:
            mental_ax.clear()
            mh = [rec.get('mental_health', 0.0) for rec in self.records]
            idxs = list(range(1, len(mh)+1))
            mental_ax.plot(idxs, mh, marker='o', color='C3')
            mental_ax.fill_between(idxs, mh, color='C3', alpha=0.12)
            mental_ax.set_ylim(0.0, 1.0)
            mental_ax.set_title('User mental health (risk)')
            mental_ax.set_xlabel('Turn')
            mental_ax.set_ylabel('Risk (0..1)')
            mental_ax.grid(True, alpha=0.25)
        self.fig.canvas.draw_idle()
        plt.pause(0.001)
    def process_pending_requests(self):
        if not self._pending_show.is_set():
            return
        with self._show_lock:
            try:
                if self.fig is None:
                    self._create_figure()
                self._update_plot()
                try:
                    plt.show(block=False)
                except Exception:
                    try:
                        plt.draw(); plt.pause(0.001)
                    except Exception:
                        pass
            finally:
                self._pending_show.clear()
def safe_chat(model, tokenizer, user_input, device, dominant_emotion="Neutral", max_new_tokens=120):
    Deterministic / low-temp safe chat call.
    - ensures pad token is set
    - uses attention_mask in generate
    - prepends a short system prompt and selects a style based on dominant_emotion
    - returns a short reply string
    if tokenizer is None or model is None:
        return ""
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    system_prompt = (
        "You are a calm, empathetic virtual assistant. "
        "When the user expresses distress, respond with supportive, non-judgmental language and ask if they are safe. "
        "Keep replies short and comforting."
    )
    style_map = {
        "Sadness":"compassionate and reassuring",
        "Empathic Pain":"compassionate and gentle",
        "Anger":"calm and de-escalating",
        "Boredom":"light and engaging",
        "Excitement":"energetic and encouraging",
        "Neutral":"neutral and helpful",
        "Fear":"reassuring and grounding"
    }
    style = style_map.get(dominant_emotion, "neutral and helpful")
    prompt = f"{system_prompt}\n\nAssistant style: {style}\n\nUser: {user_input}\nAssistant:"
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    try:
        with torch.no_grad():
            out_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.2,
                top_k=40,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
        gen = out_ids[0, input_ids.shape[-1]:]
        reply = tokenizer.decode(gen, skip_special_tokens=True).strip()
        return reply
    except Exception as e:
        return ""
class SimpleChatbot:
    def __init__(self, model_name=CHATBOT_MODEL_NAME, max_length=200):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers/torch not available")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.chat_history_ids = None
        self.max_length = max_length
    def chat(self, user_input, dominant_emotion="Neutral", meta=None):
        Safer chat API:
         - If meta indicates self-harm/emergency/violence, bypass LLM and return rule-based safe reply
         - Otherwise call safe_chat()
        meta = meta or {}
        if meta.get("self_harm_risk", False):
            return ("I'm really sorry you're feeling this way. If you're thinking about harming yourself or are in immediate danger, "
                    "please contact local emergency services or a crisis line right now. If you'd like, we can stay here and talk Ã¢â‚¬â€ are you safe right now?")
        if meta.get("emergency", False):
            return "If this is an emergency, contact local emergency services immediately. Do you need help right now?"
        if meta.get("violence", False):
            return "I hear violent intent in that message. If someone is in danger, contact emergency services. If you're upset, I can help find support or we can talk."
        reply = safe_chat(self.model, self.tokenizer, user_input, self.device, dominant_emotion=dominant_emotion, max_new_tokens=120)
        if not reply:
            return "I understand. Would you like help or someone to talk to?"
        return reply
    def reset(self):
        self.chat_history_ids = None
class NeuroVisualizer:
    def __init__(self, lif_pop, encoder, window_ms=OSCILLOSCOPE_WINDOW_MS, tracker=None):
        self.lif = lif_pop
        self.encoder = encoder
        self.window_ms = window_ms
        self.tracker = tracker if tracker is not None else None
        self.latest_lock = threading.Lock()
        self.latest_v_history = None
        self.latest_spike_times = None
        self.latest_tokens = None
        self.fig = None
        self.anim = None
        self._is_setup = False
    def update_latest(self, lif_pop, tokens):
        with self.latest_lock:
            self.latest_v_history = lif_pop.v_history.copy() if lif_pop.v_history is not None else None
            self.latest_spike_times = [list(s) for s in lif_pop.spike_times]
            self.latest_tokens = tokens
    def setup_plot(self):
        if self._is_setup:
            return
        plt.ion()
        self.fig = plt.figure(figsize=(14,8), constrained_layout=True)
        gs = self.fig.add_gridspec(3, 4)
        self.ax_raster = self.fig.add_subplot(gs[0, :3])
        self.ax_mem = self.fig.add_subplot(gs[1:, :3])
        self.ax_3d = self.fig.add_subplot(gs[:, 3], projection='3d')
        self.ax_raster.set_title("Spike raster (neurons) Ã¢â‚¬â€ all neurons")
        self.ax_raster.set_xlabel("time (ms)")
        self.ax_raster.set_ylabel("neuron index")
        self.ax_raster.set_xlim(0, self.window_ms)
        self.ax_raster.set_ylim(-0.5, self.lif.n - 0.5)
        self.ax_mem.set_title("Membrane potentials (ALL neurons) Ã¢â‚¬â€ oscilloscope view")
        self.ax_mem.set_xlabel("time (ms)")
        self.ax_mem.set_ylabel("neuron index -> voltage plotted as offset traces")
        self.ax_mem.set_xlim(0, self.window_ms)
        self.ax_mem.set_ylim(-1.0, self.lif.n + 1.0)
        self.ax_3d.set_title("Firing summary: time (x) vs neuron (y) (color=voltage)")
        self.ax_3d.set_xlabel("time (ms)")
        self.ax_3d.set_ylabel("neuron index")
        self.ax_3d.set_xlim(0, self.window_ms)
        self.ax_3d.set_ylim(-0.5, self.lif.n - 0.5)
        self.mem_lines = [self.ax_mem.plot([], [], linewidth=0.8)[0] for _ in range(self.lif.n)]
        self.raster_scatter = self.ax_raster.scatter([], [], s=8)
        self.summary_scatter = self.ax_3d.scatter([], [], c=[], s=12, cmap='viridis', vmin=-1.5, vmax=2.5)
        self.dom_text = self.ax_raster.text(0.99, 0.02, "Dominant: -", transform=self.ax_raster.transAxes,
                                            fontsize=10, ha='right', va='bottom', bbox=dict(facecolor="
        try:
            img = plt.imread(SAMPLE_IMAGE_PATH)
            ax_img = self.fig.add_axes([0.78, 0.02, 0.18, 0.18], anchor='SE')
            ax_img.imshow(img)
            ax_img.axis('off')
        except Exception:
            pass
        def init():
            for ln in self.mem_lines:
                ln.set_data([], [])
            self.raster_scatter.set_offsets(np.empty((0,2)))
            self.summary_scatter.set_offsets(np.empty((0,2)))
            try:
                self.summary_scatter.set_array(np.array([]))
            except Exception:
                pass
            self.dom_text.set_text("Dominant: -")
            return (self.raster_scatter, self.summary_scatter, *self.mem_lines, self.dom_text)
        def update(frame):
            with self.latest_lock:
                vhist = self.latest_v_history
                spt = self.latest_spike_times
                toks = self.latest_tokens
            if vhist is None or spt is None:
                return (self.raster_scatter, self.summary_scatter, *self.mem_lines, self.dom_text)
            n_steps = vhist.shape[0]
            t_axis = np.arange(n_steps) * DT
            xs=[]; ys=[]
            for n_idx, times_list in enumerate(spt):
                for t in times_list:
                    xs.append(t); ys.append(n_idx)
            if len(xs)>0:
                self.raster_scatter.set_offsets(np.column_stack((xs, ys)))
            else:
                self.raster_scatter.set_offsets(np.empty((0,2)))
            gain = 0.6
            for i in range(self.lif.n):
                y_trace = i + (vhist[:, i] * gain)
                self.mem_lines[i].set_data(t_axis, y_trace)
                self.mem_lines[i].set_color('C0')
                self.mem_lines[i].set_alpha(0.9 if i%2==0 else 0.6)
            spike_x=[]; spike_y=[]; spike_z=[]
            for n_idx, times_list in enumerate(spt):
                for t in times_list:
                    spike_x.append(t); spike_y.append(n_idx); spike_z.append(1.2)
            sample_stride = max(1, int(max(1, n_steps // 200)))
            volt_x=[]; volt_y=[]; volt_z=[]
            for ti in range(0, n_steps, sample_stride):
                for ni in range(0, self.lif.n, max(1, int(self.lif.n // 80))):
                    volt_x.append(t_axis[ti]); volt_y.append(ni); volt_z.append(vhist[ti, ni])
            if spike_x or volt_x:
                xs3 = np.array(spike_x + volt_x)
                ys3 = np.array(spike_y + volt_y)
                zs3 = np.array(spike_z + volt_z)
                offs = np.column_stack((xs3, ys3))
                try:
                    self.summary_scatter.set_offsets(offs)
                    self.summary_scatter.set_array(zs3)
                    self.summary_scatter.set_alpha(0.8)
                except Exception:
                    self.summary_scatter.set_offsets(np.empty((0,2)))
                    try:
                        self.summary_scatter.set_array(np.array([]))
                    except Exception:
                        pass
            else:
                self.summary_scatter.set_offsets(np.empty((0,2)))
                try:
                    self.summary_scatter.set_array(np.array([]))
                except Exception:
                    pass
            dom_label = "-"
            try:
                if toks:
                    joined = " ".join(toks)
                    vecp, dom_label, _meta = prototype_detector.predict(joined)
                else:
                    dom_label = "-"
            except Exception:
                dom_label = "-"
            self.dom_text.set_text(f"Dominant: {dom_label}")
            self.dom_text.set_bbox(dict(facecolor="
            self.ax_raster.set_xlim(0, self.window_ms)
            self.ax_mem.set_xlim(0, self.window_ms)
            self.ax_3d.set_xlim(0, self.window_ms)
            try:
                if getattr(self, 'tracker', None) is not None:
                    try:
                        self.tracker.process_pending_requests()
                    except Exception:
                        pass
            except Exception:
                pass
            return (self.raster_scatter, self.summary_scatter, *self.mem_lines, self.dom_text)
        self.anim = animation.FuncAnimation(self.fig, update, init_func=init,
                                            interval=PLOT_UPDATE_INTERVAL_MS, blit=False)
        self._is_setup = True
    def start(self, stop_event=None):
        Keep the GUI alive on main thread until stop_event is set.
        Must be called on the main thread.
        if not self._is_setup:
            self.setup_plot()
        try:
            plt.show(block=True)
            if stop_event is not None:
                try:
                    stop_event.set()
                except Exception:
                    pass
            return
        except Exception:
            try:
                if stop_event is None:
                    plt.show(block=True)
                    return
                while not stop_event.is_set():
                    plt.pause(0.1)
                try:
                    plt.close('all')
                except Exception:
                    pass
            except Exception as e:
                print(f"[Visualizer.start] fallback error: {e}")
class NeuroNovaEngineSemantic:
    def __init__(self):
        self.encoder = SpikingEncoder(window_ms=WINDOW_MS, channels=N_CHANNELS, seed=11)
        self.prototype_detector = prototype_detector
        self.embedding_detector = embedding_detector
        self.lif = LIFPopulation(n_neurons=N_NEURONS, n_channels=N_CHANNELS, dt=DT, tau_m=30.0, v_thresh=1.0, seed=7)
        self.decider = DecisionModule(rate_threshold=8.0)
        self.respgen = ResponseGenerator(seed=42)
        self.tracker = ConversationTracker(EMOTIONS)
        self.chatbot = None
        if TRANSFORMERS_AVAILABLE:
            try:
                self.chatbot = SimpleChatbot(model_name=CHATBOT_MODEL_NAME)
            except Exception:
                self.chatbot = None
        self.GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY') or "AIzaSyD3HXJRZJkt0XZR1pJ-_chZhYlr4ywGiMU"
        self.remote_chat_enabled = bool(self.GEMINI_API_KEY)
        self.conversation_history = []
    def detect_emotion(self, text):
        toks = tokenize(text)
        if not toks:
            return np.zeros(len(EMOTIONS)), "Neutral", {}
        text = text.strip()
        low = text.lower()
        suicidal_phrases = ["i want to die","kill myself","end my life","i cant go on","i'm going to kill myself"]
        for pat in suicidal_phrases:
            if pat in low:
                vec = np.zeros(len(EMOTIONS), dtype=float)
                for e in ("Sadness","Fear","Empathic Pain"):
                    vec[EMOTIONS.index(e)] = 1.0
                vec = vec / (vec.max()+1e-12)
                return vec, "Sadness", {"self_harm_risk": True, "pattern": pat}
        if EMBED_AVAILABLE and self.embedding_detector is not None:
            vec, dom, meta = self.embedding_detector.predict(text)
            return vec, dom, meta
        vec, dom, meta = self.prototype_detector.predict(text)
        return vec, dom, meta
    def remote_chat(self, user_input, dominant_emotion="Neutral", max_tokens=200):
        if not getattr(self, 'GEMINI_API_KEY', None):
            return ""
        try:
            import json, urllib.request
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={self.GEMINI_API_KEY}"
            
            system_context = ("You are a warm, empathetic AI assistant integrated with NeuroNovaEngine emotion detection system. "
                           "The system detected the user's current emotion as: " + dominant_emotion + ". "
                           "Respond naturally and conversationally, matching their emotional tone. "
                           "Be empathetic, supportive, and engaging. Keep responses natural and flowing (2-4 sentences). "
                           "Reference previous conversation when relevant to maintain continuity.")
            
            conversation_context = ""
            if hasattr(self, 'conversation_history') and len(self.conversation_history) > 0:
                recent_history = self.conversation_history[-4:]
                conversation_context = "\n\nRecent conversation context:\n"
                for entry in recent_history:
                    conversation_context += f"User: {entry.get('user', '')}\nAssistant: {entry.get('assistant', '')}\n"
            
            full_prompt = system_context + conversation_context + "\n\nUser: " + user_input + "\nAssistant:"
            
            body = {
                "contents": [{
                    "parts": [{"text": full_prompt}]
                }],
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": 0.8,
                    "topP": 0.9,
                    "topK": 50
                }
            }
            data = json.dumps(body).encode('utf-8')
            req = urllib.request.Request(url, data=data, headers={
                'Content-Type': 'application/json'
            })
            with urllib.request.urlopen(req, timeout=25) as resp:
                j = json.load(resp)
            candidates = j.get('candidates', [])
            if not candidates:
                return ""
            parts = candidates[0].get('content', {}).get('parts', [])
            if not parts:
                return ""
            msg = parts[0].get('text', '')
            response = (msg or "").strip()
            
            if not hasattr(self, 'conversation_history'):
                self.conversation_history = []
            self.conversation_history.append({
                "user": user_input,
                "assistant": response,
                "emotion": dominant_emotion
            })
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            return response
        except Exception as e:
            logger.debug(f"gemini_chat error: {e}")
            return ""
    def process_text(self, text):
        print("\nInput text:")
        print(textwrap.fill(text, width=80))
        vec, dominant, meta = self.detect_emotion(text)
        print("\nTop emotion scores (0..1):")
        sorted_idx = np.argsort(-vec)[:6]
        for idx in sorted_idx:
            print(f"  {EMOTIONS[idx]:<22s}: {vec[idx]:.3f}  (color: {EMOTION_COLORS.get(EMOTIONS[idx],'
        dominant_label = "Neutral" if np.max(vec) <= 0 else dominant
        print(f"\nDominant emotion: {dominant_label}  (color: {EMOTION_COLORS.get(dominant_label,'
        if meta:
            print(f"[Meta] {meta}")
        channels, tokens = self.encoder.encode(text)
        print(f"\n[SpikingEncoder] tokens: {tokens}")
        for i, ch in enumerate(channels):
            if ch:
                print(f"  channel {i}: spike times (ms) -> {np.array(ch).round(1).tolist()}")
        rates = self.lif.run(channels, t_max_ms=self.encoder.window_ms)
        print(f"\n[LIFPopulation] mean firing rate: {np.mean(rates):.2f} spikes/s  (per neuron)")
        decision = self.decider.decide(rates, dominant_label, meta)
        print(f"[Decision] action: {decision['action']}, tone: {decision['tone']}, mean_rate: {decision['mean_rate']:.2f}")
        reply = ""
        if getattr(self, 'remote_chat_enabled', False) and not meta.get('self_harm_risk', False):
            try:
                reply = self.remote_chat(text, dominant_emotion=dominant_label, max_tokens=200)
            except Exception:
                reply = ""
        if not reply:
            reply = self.respgen.generate(decision['action'], decision['tone'], text, dominant_label, meta)
        print(f"\nRobot> {reply}\n")
        try:
            self.tracker.add_entry(
                user_text=text,
                emotion_vec=vec,
                dominant=dominant_label,
                decision=decision,
                meta=meta,
                lif=self.lif,
                tokens=tokens
            )
        except Exception as e:
            print(f"[Tracker warn] {e}")
        return {"vec": vec, "dominant": dominant_label, "meta": meta, "decision": decision, "reply": reply, "tokens": tokens}
class NeuroNovaCLI:
    def __init__(self, engine, visualizer, stop_event=None):
        self.engine = engine
        self.visualizer = visualizer
        self.stop_event = stop_event or threading.Event()
        self.running = True
    def process_text(self, text):
        res = self.engine.process_text(text)
        if self.visualizer is not None:
            self.visualizer.update_latest(self.engine.lif, res.get("tokens", []))
        return res
    def repl(self):
        print("NeuroNova semantic visual demo")
        print("Commands: exit / quit / history / graph / oscilloscope / bot / plot off")
        print("Note: Graphs only open when you explicitly request them with 'graph' or 'oscilloscope' commands.")
        while self.running:
            try:
                s = input("You> ").strip()
            except (KeyboardInterrupt, EOFError):
                s = "exit"
            if not s:
                continue
            cmd = s.lower().strip()
            if cmd in ("exit", "quit"):
                print("Exiting...")
                self.running = False
                try:
                    self.stop_event.set()
                except Exception:
                    pass
                try: plt.close('all')
                except Exception: pass
                break
            if cmd == "history":
                self.engine.tracker.print_history(); continue
            if cmd == "graph":
                self.engine.tracker.show_graph()
                time.sleep(0.5)
                continue
            if cmd in ("oscilloscope", "osc", "visualizer"):
                print("[Oscilloscope] Neural activity visualizer not available. Use 'graph' to see conversation emotion graphs.")
                continue
            if cmd == "bot":
                if self.engine.chatbot is None:
                    print("[Chatbot unavailable] Install transformers & torch to enable.")
                else:
                    print("SimpleChatbot session started. Type 'back' to return.")
                    self.engine.chatbot.reset()
                    while True:
                        try:
                            u = input("BotUser> ").strip()
                        except (KeyboardInterrupt, EOFError):
                            break
                        if not u: continue
                        if u.lower() in ("back", "exit", "quit"):
                            break
                        resp = self.engine.chatbot.chat(u, dominant_emotion="Neutral", meta={})
                        print("Bot> " + resp)
                continue
            if cmd == "plot off":
                print("[Plotting disabled] Closing windows.");
                try: plt.close('all')
                except Exception: pass
                continue
            if cmd == "plot on":
                print("[Plotting enabled] (if window closed, restart script to recreate visualizer)"); continue
            self.process_text(s)
def main():
    engine = NeuroNovaEngineSemantic()
    visualizer = NeuroVisualizer(engine.lif, engine.encoder, window_ms=OSCILLOSCOPE_WINDOW_MS, tracker=engine.tracker)
    stop_event = threading.Event()
    cli = NeuroNovaCLI(engine, visualizer, stop_event=stop_event)
    cli_thread = threading.Thread(target=cli.repl, daemon=True)
    cli_thread.start()
    try:
        plt.ioff()
        while not stop_event.is_set():
            try:
                if visualizer.tracker is not None:
                    visualizer.tracker.process_pending_requests()
            except Exception:
                pass
            try:
                plt.pause(0.05)
            except Exception:
                pass
            if not cli_thread.is_alive():
                break
    except KeyboardInterrupt:
        stop_event.set()
    except Exception as e:
        print(f"[Main loop error] {e}")
    try:
        if cli_thread.is_alive():
            stop_event.set()
            cli.running = False
            cli_thread.join(timeout=1.0)
    except Exception:
        pass
    print("Shutdown complete.")
if __name__ == "__main__":
    main()