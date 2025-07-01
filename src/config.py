import os
from typing import Any, Dict

import torch
import yaml


class Config:
    """Configuration manager for AlphaZero training."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_device()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        return config

    def _setup_device(self):
        """Setup device based on configuration."""
        device_config = self.config.get("device", "auto")

        if device_config == "auto":
            if torch.cuda.is_available():
                self.config["device"] = "cuda"
            elif torch.backends.mps.is_available():
                self.config["device"] = "mps"
            else:
                self.config["device"] = "cpu"
        else:
            self.config["device"] = device_config

    def get(self, key: str, default=None):
        """Get configuration value using dot notation."""
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_network_config(self) -> Dict[str, Any]:
        """Get network configuration."""
        return self.config.get("network", {})

    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.config.get("training", {})

    def get_scheduler_config(self) -> Dict[str, Any]:
        """Get scheduler configuration."""
        return self.config.get("scheduler", {})

    def get_self_play_config(self) -> Dict[str, Any]:
        """Get self-play configuration."""
        return self.config.get("self_play", {})

    def get_dirichlet_config(self) -> Dict[str, Any]:
        """Get Dirichlet noise configuration."""
        return self.config.get("dirichlet", {})

    def get_training_loop_config(self) -> Dict[str, Any]:
        """Get training loop configuration."""
        return self.config.get("training_loop", {})

    def get_directories_config(self) -> Dict[str, Any]:
        """Get directories configuration."""
        return self.config.get("directories", {})

    def get_device(self) -> str:
        """Get device configuration."""
        return self.config.get("device", "auto")

    def update(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        for key, value in updates.items():
            keys = key.split(".")
            config = self.config

            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]

            config[keys[-1]] = value

    def set(self, key: str, value: Any):
        """Set a configuration value using dot notation."""
        keys = key.split(".")
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save(self, path: str = None):
        """Save current configuration to file."""
        if path is None:
            path = self.config_path

        with open(path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)

    def print_config(self):
        """Current configuration."""
        print("Current Configuration:")
        print("=" * 60)
        print(yaml.dump(self.config, default_flow_style=False, indent=2))
        print("=" * 60)
