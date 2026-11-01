"""Runtime configuration, read from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader so the project has no extra dependency for it."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("CLAUDE_MODEL", "claude-haiku-5-5"))
    kb_dir: Path = field(default_factory=lambda: Path(os.getenv("KB_DIR", ROOT / "knowledge_base")))
    db_path: Path = field(default_factory=lambda: Path(os.getenv("DB_PATH", ROOT / "data" / "support.db")))
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "4")))
    # Minimum share of the question's key terms that must appear in the retrieved
    # context before the bot is allowed to answer instead of handing off.
    min_coverage: float = field(default_factory=lambda: float(os.getenv("MIN_COVERAGE", "0.5")))
    min_score: float = field(default_factory=lambda: float(os.getenv("MIN_SCORE", "2.0")))
    store_name: str = field(default_factory=lambda: os.getenv("STORE_NAME", "Brewline Coffee Supply"))

    @property
    def mock_mode(self) -> bool:
        """Without an API key the app still works using an extractive answerer."""
        return not self.anthropic_api_key


settings = Settings()
