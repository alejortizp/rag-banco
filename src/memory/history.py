import sqlite3, time

class ConversationRepository:
    """Patrón Repository: abstrae la persistencia del historial de conversación."""
    def __init__(self, db_path: str):
        self.db_path = db_path
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at REAL NOT NULL)""")

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def add_message(self, session_id: str, role: str, content: str) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
                      (session_id, role, content, time.time()))

    def get_last_n(self, session_id: str, n: int) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT role, content, created_at FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (session_id, n)).fetchall()
        return [{"session_id": session_id, "role": r, "content": co, "created_at": t}
                for r, co, t in reversed(rows)]

    def all_messages(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT session_id, role, content, created_at FROM messages ORDER BY id").fetchall()
        return [{"session_id": s, "role": r, "content": co, "created_at": t} for s, r, co, t in rows]
