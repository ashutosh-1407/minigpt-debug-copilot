import logging
import sys
import time
import uuid
import json


logger = logging.getLogger("mini_debug_copilot")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler = handler

def generate_request_id() -> str:
    return str(uuid.uuid4())

def log_event(event_name: str, **kwargs):
    payload = {
        "event": event_name,
        "timestamp": time.time(),
        **kwargs
    }
    logger.info(json.dumps(payload))
