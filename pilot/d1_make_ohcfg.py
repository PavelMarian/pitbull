"""Create an isolated OpenHarness config dir for the D1 skill-parity run.

Why a separate dir: OpenHarness 0.1.4 ignores `--settings`/`-k` for profiles
with a credential slot and loads user skills only from `<config>/skills/`.
An isolated OPENHARNESS_CONFIG_DIR gives us (a) the OpenRouter key from .env
without exporting it as an environment variable (the run must keep
OPENROUTER_API_KEY out of the environment of the process that executes
generated code) and (b) the `pit-repair` skill visible to the harness via a
symlink to `.claude/skills/pit-repair`.

Usage: python pilot/d1_make_ohcfg.py   (writes ~/.openharness-d1, mode 700)
"""
import json
import os
import sys
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
CFG = Path(os.environ.get("D1_OH_CONFIG_DIR", Path.home() / ".openharness-d1"))
MODEL = os.environ.get("D1_MODEL", "z-ai/glm-5.3-flash")
MAX_TOKENS = int(os.environ.get("D1_MAX_TOKENS", "14000"))  # as in A2 fetch


def main():
    key = dotenv_values(ROOT / ".env").get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY not found in .env")
    os.umask(0o077)
    (CFG / "skills").mkdir(parents=True, exist_ok=True)
    (CFG / "plugins").mkdir(exist_ok=True)
    link = CFG / "skills" / "pit-repair"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(ROOT / ".claude" / "skills" / "pit-repair", target_is_directory=True)
    settings = {
        "active_profile": "d1-openrouter", "api_key": key, "model": MODEL, "max_tokens": MAX_TOKENS,
        "profiles": {"d1-openrouter": {
            "label": "D1 OpenRouter", "provider": "openai", "api_format": "openai",
            "auth_source": "openai_api_key", "default_model": MODEL,
            "base_url": "https://openrouter.ai/api/v1", "credential_slot": None, "allowed_models": []}},
    }
    (CFG / "settings.json").write_text(json.dumps(settings, indent=1), encoding="utf-8")
    os.chmod(CFG / "settings.json", 0o600)
    print("config dir:", CFG, "| model:", MODEL, "| skill link ->", link.resolve())


if __name__ == "__main__":
    main()
