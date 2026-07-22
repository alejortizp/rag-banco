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

def test_metricas_con_lista_vacia():
    m = compute_metrics([])
    assert m["total_sessions"] == 0 and m["total_messages"] == 0
    assert m["avg_messages_per_session"] == 0.0 and m["avg_user_msg_length"] == 0.0
    assert m["top_terms"] == [] and m["messages_per_day"] == {}

def test_top_terms_ignora_puntuacion():
    msgs = [
        {"session_id": "a", "role": "user", "content": "¿Tarjetas? credito, credito!", "created_at": 1.0},
        {"session_id": "a", "role": "user", "content": "tarjetas credito", "created_at": 2.0},
    ]
    terms = dict(compute_metrics(msgs)["top_terms"])
    assert terms["tarjetas"] == 2
    assert terms["credito"] == 3
