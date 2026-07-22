import re
from collections import Counter
from datetime import datetime, timezone

STOPWORDS = {
    "de", "la", "el", "en", "y", "que", "los", "las", "un", "una", "es", "por", "para", "con", "del", "se", "como", "mi", "me", "hay",
    "esta", "está", "son", "una", "unas", "unos", "pero", "más", "este", "esto", "tengo", "quiero", "puedo", "cómo", "cuál", "qué", "donde", "dónde"
}

def compute_metrics(messages: list[dict]) -> dict:
    sessions = {m["session_id"] for m in messages}
    user_msgs = [m for m in messages if m["role"] == "user"]
    # dígitos excluidos deliberadamente: es una métrica de términos temáticos, no de identificadores
    words = [w for m in user_msgs for w in re.findall(r"[a-záéíóúüñ]+", m["content"].lower())
             if w not in STOPWORDS and len(w) > 3]
    per_day = Counter(
        datetime.fromtimestamp(m["created_at"], tz=timezone.utc).strftime("%Y-%m-%d")
        for m in messages
    )
    return {
        "total_sessions": len(sessions),
        "total_messages": len(messages),
        "user_messages": len(user_msgs),
        "avg_messages_per_session": round(len(messages) / len(sessions), 2) if sessions else 0.0,
        "avg_user_msg_length": round(sum(len(m["content"]) for m in user_msgs) / len(user_msgs), 1) if user_msgs else 0.0,
        "top_terms": Counter(words).most_common(10),
        "messages_per_day": dict(per_day),
    }
