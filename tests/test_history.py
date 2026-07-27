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


def test_list_sessions_ordena_por_actividad_reciente(tmp_path):
    repo = ConversationRepository(db_path=str(tmp_path / "t.db"))
    repo.add_message("vieja", "user", "hola")
    repo.add_message("activa", "user", "hola")
    repo.add_message("activa", "assistant", "respuesta")
    repo.add_message("vieja", "user", "otra mas")  # 'vieja' vuelve a ser la mas reciente
    sessions = repo.list_sessions()
    assert [s["session_id"] for s in sessions] == ["vieja", "activa"]
    assert sessions[0]["message_count"] == 2
    assert sessions[1]["message_count"] == 2
    assert sessions[0]["last_activity"] >= sessions[1]["last_activity"]


def test_list_sessions_sin_datos_devuelve_vacio(tmp_path):
    repo = ConversationRepository(db_path=str(tmp_path / "t.db"))
    assert repo.list_sessions() == []


def test_delete_session_elimina_solo_esa_sesion(tmp_path):
    repo = ConversationRepository(db_path=str(tmp_path / "t.db"))
    repo.add_message("borrar", "user", "hola")
    repo.add_message("borrar", "assistant", "respuesta")
    repo.add_message("conservar", "user", "hola")
    repo.delete_session("borrar")
    assert repo.get_last_n("borrar", 10) == []
    assert len(repo.get_last_n("conservar", 10)) == 1
    assert [s["session_id"] for s in repo.list_sessions()] == ["conservar"]


def test_delete_session_inexistente_no_falla(tmp_path):
    repo = ConversationRepository(db_path=str(tmp_path / "t.db"))
    repo.delete_session("no-existe")  # no debe lanzar
    assert repo.list_sessions() == []
