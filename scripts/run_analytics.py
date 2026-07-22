import json
from src.config import get_settings
from src.memory.history import ConversationRepository
from src.analytics.metrics import compute_metrics

if __name__ == "__main__":
    repo = ConversationRepository(get_settings().db_path)
    print(json.dumps(compute_metrics(repo.all_messages()), indent=2, ensure_ascii=False))
