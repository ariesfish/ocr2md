from __future__ import annotations

import argparse
import sys

import uvicorn

from ocr2md import load_config
from ocr2md.utils.logging import ensure_logging_configured


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ocr2md FastAPI server.")
    parser.add_argument("--config", "-c", type=str, default=None)
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_logging_configured(level=cfg.logging.level, format_string=cfg.logging.format)

    host = args.host or cfg.web.host
    port = int(args.port or cfg.web.port)
    uvicorn.run("ocr2md.web.app:app", host=host, port=port, reload=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
