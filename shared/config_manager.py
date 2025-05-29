"""
Configuration Manager for AI Pipeline

This module provides a centralized way to manage configuration settings
using YAML files. It supports loading, saving, and updating configurations
with proper validation and type handling.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import copy

class ConfigManager:
    """Manages configuration loading and saving from YAML files."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        self._config = {}
        self.load()
    
    def load(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Dictionary containing configuration settings
        """
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f) or {}
        else:
            self._config = {}
        return self._config
    
    def save(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration dictionary to save. If None, saves current config.
        """
        if config is not None:
            self._config = config
        
        with open(self.config_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'data.path' or 'costs.false_positive')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'data.path')
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config or not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update multiple configuration values.
        
        Args:
            updates: Dictionary of key-value pairs to update
        """
        for key, value in updates.items():
            self.set(key, value)
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire configuration section.
        
        Args:
            section: Section name (e.g., 'data', 'models')
            
        Returns:
            Dictionary containing the section
        """
        return self.get(section, {})
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get the entire configuration as a dictionary.
        
        Returns:
            Complete configuration dictionary
        """
        return copy.deepcopy(self._config)
    
    def validate(self) -> bool:
        """
        Validate the configuration.
        
        Returns:
            True if configuration is valid
        """
        required_sections = ['data', 'costs', 'output']
        for section in required_sections:
            if section not in self._config:
                return False
        return True
    
    # Convenience methods for common config access patterns
    @property
    def data_path(self) -> str:
        """Get the data path."""
        return self.get('data.path', 'data/training_data.csv')
    
    @property
    def target_column(self) -> str:
        """Get the target column name."""
        return self.get('data.target_column', 'GT_Label')
    
    @property
    def output_dir(self) -> str:
        """Get the output directory."""
        return self.get('output.base_dir', 'output')
    
    @property
    def model_dir(self) -> str:
        """Get the model directory."""
        base_dir = self.output_dir
        models_subdir = self.get('output.subdirs.models', 'models')
        return os.path.join(base_dir, models_subdir)
    
    @property
    def plots_dir(self) -> str:
        """Get the plots directory."""
        base_dir = self.output_dir
        plots_subdir = self.get('output.subdirs.plots', 'plots')
        return os.path.join(base_dir, plots_subdir)
    
    @property
    def predictions_dir(self) -> str:
        """Get the predictions directory."""
        base_dir = self.output_dir
        predictions_subdir = self.get('output.subdirs.predictions', 'predictions')
        return os.path.join(base_dir, predictions_subdir)
    
    @property
    def cost_fp(self) -> float:
        """Get the false positive cost."""
        return self.get('costs.false_positive', 1)
    
    @property
    def cost_fn(self) -> float:
        """Get the false negative cost."""
        return self.get('costs.false_negative', 30)
    
    @property
    def use_kfold(self) -> bool:
        """Get whether to use k-fold cross validation."""
        return self.get('training.use_kfold', True)
    
    @property
    def n_splits(self) -> int:
        """Get the number of k-fold splits."""
        return self.get('training.n_splits', 5)

# Global configuration manager instance
config_manager = ConfigManager()

def load_config(config_path: str = "config.yaml") -> ConfigManager:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        ConfigManager instance
    """
    return ConfigManager(config_path)

def get_config() -> ConfigManager:
    """
    Get the global configuration manager instance.
    
    Returns:
        Global ConfigManager instance
    """
    return config_manager 