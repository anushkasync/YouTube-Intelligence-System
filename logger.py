import os
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

from config.config import LOG_DIR

os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("rag_system")

logger.setLevel(logging.INFO)

logger.propagate = False

if not logger.handlers:

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(trace_id)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()

    console_handler.setLevel(logging.INFO)

    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        filename=os.path.join(LOG_DIR, "app.log"),
        maxBytes=1_000_000,
        backupCount=5,
        encoding="utf-8"
    )

    file_handler.setLevel(logging.INFO)

    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

class TraceLoggerAdapter(logging.LoggerAdapter):

    def process(self, msg, kwargs):

        extra = kwargs.setdefault("extra", {})

        extra.setdefault("trace_id", self.extra.get("trace_id", "NO_TRACE"))

        return msg, kwargs


def get_logger(trace_id=None):

    return TraceLoggerAdapter(
        logger,
        {"trace_id": trace_id or "NO_TRACE"}
    )

EVENT_LOG_FILE = os.path.join(LOG_DIR, "events.jsonl")


def log_event(event, path=None):

    log_file = path or EVENT_LOG_FILE

    event["timestamp"] = (
        datetime.utcnow().isoformat() + "Z"
    )

    with open(log_file, "a", encoding="utf-8") as stream:

        json.dump(event, stream)

        stream.write("\n")