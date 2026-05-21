"""Node configuration management."""
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

DEFAULT_CONFIG_DIR = Path.home() / ".gitblock-node"
DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "config.json"

@dataclass
class NodeConfig:
    """Configuration for a GitBlock node."""
    wallet_address: str = ""
    api_url: str = "https://api.gitblock.io"
    listen_host: str = "0.0.0.0"
    listen_port: int = 8000
    models: list[str] = field(default_factory=lambda: ["llama-3.3-70b"])
    max_concurrent: int = 4
    reputation_threshold: float = 0.5
    heartbeat_interval: int = 30
    log_level: str = "INFO"
    data_dir: str = str(DEFAULT_CONFIG_DIR / "data")

    @classmethod
    def from_env(cls) -> "NodeConfig":
        """Load config from environment variables."""
        cfg = cls()
        for field_name in cls.__dataclass_fields__:
            env_key = f"GBNODE_{field_name.upper()}"
            val = os.environ.get(env_key)
            if val is not None:
                fld = cls.__dataclass_fields__[field_name]
                if fld.type == "list[str]":
                    setattr(cfg, field_name, val.split(","))
                elif fld.type == "int":
                    setattr(cfg, field_name, int(val))
                elif fld.type == "float":
                    setattr(cfg, field_name, float(val))
                else:
                    setattr(cfg, field_name, val)
        return cfg

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "NodeConfig":
        """Load config from file, falling back to env vars."""
        path = path or DEFAULT_CONFIG_FILE
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            cfg = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        else:
            cfg = cls.from_env()
        return cfg

    def save(self, path: Optional[Path] = None) -> None:
        """Save config to file."""
        path = path or DEFAULT_CONFIG_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
