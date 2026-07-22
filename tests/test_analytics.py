from src.analytics.metrics import compute_metrics

def test_metricas_basicas():
    msgs = [
        {"session_id": "a", "role": "user", "content": "hola tarjetas credito", "created_at": 1.0},
        {"session_id": "a", "role": "assistant", "content": "claro", "created_at": 2.0},
        {"session_id": "b", "role": "user", "content": "tarjetas debito", "created_at": 3.0},
    ]
    m = compute_metrics(msgs)
    assert m["total_sessions"] == 2
    assert m["total_messages"] == 3
    assert m["user_messages"] == 2
    assert "tarjetas" in dict(m["top_terms"])
