"""Load a config file and apply command-line switches. Shared by app.py and web.py."""

import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def build_config(config_file: str = "config.json", with_evil: bool = False,
                 no_safety: bool = False) -> Path:
    path = Path(config_file)
    if not path.is_absolute():
        path = ROOT / "assistant" / config_file
    config = json.loads(path.read_text(encoding="utf-8"))
    for s in config["servers"]:
        if s["name"] == "evil" and with_evil:
            s["enabled"] = True
    if no_safety:
        config["safety"]["quarantine_suspicious_tools"] = False
    out = Path(tempfile.gettempdir()) / f"mcp_assistant_{path.stem}.json"
    out.write_text(json.dumps(config))
    return out
