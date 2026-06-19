import json
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime, timezone


LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "claim_id"):
            obj["claim_id"] = record.claim_id
        if hasattr(record, "provider"):
            obj["provider"] = record.provider
        if hasattr(record, "duration_ms"):
            obj["duration_ms"] = record.duration_ms
        if hasattr(record, "cache_hit"):
            obj["cache_hit"] = record.cache_hit
        if hasattr(record, "model"):
            obj["model"] = record.model
        if record.exc_info and record.exc_info[0]:
            obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(obj, default=str)


def setup_logging(name: str = "evidence-review") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "evidence-review.jsonl",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JSONFormatter())

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def get_logger(name: str = "evidence-review") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logging(name)
    return logger
