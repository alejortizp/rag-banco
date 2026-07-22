from src.memory.history import ConversationRepository

def test_persiste_y_recupera_ultimos_n(tmp_path):
    repo = ConversationRepository(db_path=str(tmp_path / "t.db"))
    for i in range(10):
        repo.add_message("sesion-1", "user", f"msg {i}")
    ultimos = repo.get_last_n("sesion-1", n=3)
    assert [m["content"] for m in ultimos] == ["msg 7", "msg 8", "msg 9"]

def test_sesiones_aisladas(tmp_path):
    repo = ConversationRepository(db_path=str(tmp_path / "t.db"))
    repo.add_message("a", "user", "hola")
    assert repo.get_last_n("b", n=5) == []
