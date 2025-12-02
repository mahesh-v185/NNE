import os

SAMPLE_IMAGE_PATH = os.environ.get("NEURONOVA_SAMPLE_IMAGE", "assets/screenshot1.png")
DEMO_MODE = os.environ.get("NEURONOVA_DEMO", "0") == "1"
ENABLE_GUI = os.environ.get("NEURONOVA_GUI", "1") == "1"

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY') or "AIzaSyD3HXJRZJkt0XZR1pJ-_chZhYlr4ywGiMU"
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY') or os.environ.get('OPENAI_API')

USE_EMBEDDINGS = os.environ.get("NEURONOVA_USE_EMBEDDINGS", "0") == "1"
EMBEDDING_MODEL = os.environ.get("NEURONOVA_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

ENABLE_LLM_FALLBACK = os.environ.get("NEURONOVA_LLM_FALLBACK", "0") == "1"
CHATBOT_MODEL_NAME = os.environ.get("NEURONOVA_CHATBOT_MODEL", "gpt2")

WINDOW_MS = 300
OSCILLOSCOPE_WINDOW_MS = 300
ANIM_FPS = 30
PLOT_UPDATE_INTERVAL_MS = int(1000 / ANIM_FPS)
N_NEURONS = 32
N_CHANNELS = 8
DT = 1.0

INSULT_THRESHOLD = 3

EMOTIONS = [
    "Admiration", "Adoration", "Aesthetic Appreciation", "Amusement", "Anger",
    "Anxiety", "Awe", "Awkwardness", "Boredom", "Calmness",
    "Confusion", "Craving", "Disappointment", "Disgust", "Empathic Pain",
    "Entrancement", "Excitement", "Fear", "Horror", "Interest",
    "Joy", "Nostalgia", "Romance", "Sadness", "Satisfaction",
    "Sexual Desire", "Sympathy"
]

EMOTION_COLORS = {
    "Admiration": "#FF6B6B",
    "Amusement": "#4ECDC4",
    "Awkwardness": "#95E1D3",
    "Craving": "#F38181",
    "Entrancement": "#AA96DA",
    "Interest": "#FCBAD3",
    "Sadness": "#3D5A80"
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
    "Excitement": ["I'm so excited and thrilled", "I can't wait — I'm pumped"],
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

def load_insult_words():
    try:
        with open('data/insult_words.txt', 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f if line.strip()]
            return words if words else ["idiot", "stupid", "dumb", "moron", "hate", "shut up", "trash", "sucks",
                                       "jerk", "loser", "ugly", "worthless", "fool", "annoying", "nonsense", "crap"]
    except Exception:
        return ["idiot", "stupid", "dumb", "moron", "hate", "shut up", "trash", "sucks",
                "jerk", "loser", "ugly", "worthless", "fool", "annoying", "nonsense", "crap"]

INSULT_WORDS = load_insult_words()

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
        "Awesome — that made my day to hear. Anything else?",
    ],
    "Sadness": [
        "I'm sorry you're feeling down. I'm here to listen.",
        "That sounds really tough. Want to tell me more about it?",
    ],
    "Anger": [
        "I hear you're upset. Let's try to talk calmly about it.",
        "It sounds like this is frustrating — I'm listening.",
    ],
    "Boredom": [
        "If you're bored, want a quick game or a fun fact?",
        "I can share a joke or an interesting tidbit — which do you prefer?",
    ],
    "Fear": [
        "I'm sorry you're feeling scared. Are you safe right now?",
        "That sounds worrying. Would you like grounding tips or to talk more?",
    ],
    "Neutral": [
        "I see. Tell me more if you want.",
        "Okay — what else is on your mind?",
    ]
}

POLITE_FIRM_TEMPLATES = [
    "I won't engage with abusive language — let's be respectful.",
    "I understand you're upset, but I can't continue if you use insults.",
    "Please stop using hurtful language. I'm happy to help if we keep it civil.",
]

SAFETY_DISCLAIMER = """
⚠️  IMPORTANT SAFETY NOTICE:
This tool contains a non-clinical heuristic model for emotion detection and mental-health risk estimation.
It is EXPERIMENTAL and NOT a medical or diagnostic tool.

If you or someone you know is in immediate danger or exhibiting self-harm intent:
- Contact local emergency services immediately
- Reach out to a crisis helpline
- Seek professional mental health support

This system may produce false positives or false negatives. Do not rely on it for clinical decisions.
"""

