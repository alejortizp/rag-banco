from src.config import get_settings

def test_settings_es_singleton():
    assert get_settings() is get_settings()

def test_settings_tiene_defaults():
    s = get_settings()
    assert s.chunk_size > 0
    assert s.history_window_n > 0
