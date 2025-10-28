# almeval/models/__init__.py
"""
Initializes the model registry and provides the `build_model` factory function.

This module performs two main actions on startup:
1.  **Auto-Discovery**: It scans all `.py` files in the `models/` directory,
    finds all classes that inherit from `BaseModel`, and registers them
    based on their unique `NAME` attribute.
2.  **Configuration Loading**: It loads and parses `models.yaml` to understand
    the user-defined aliases and their corresponding configurations.

The public `build_model` function then uses this information to instantiate
the correct model with the correct parameters.
"""

import os
import importlib
import inspect
import pkgutil
import yaml
from .base import BaseModel

# --- Part 1: Auto-discover all BaseModel subclasses ---

def _discover_models() -> dict:
    """
    Finds and returns all subclasses of BaseModel within this package.

    It creates a dictionary mapping the class's `NAME` attribute to the class
    object itself (e.g., {'audio-flamingo-3': <class 'AudioFlamingo3'>}).

    Returns:
        A dictionary of registered model classes.
    """
    package_name = __name__
    package = importlib.import_module(package_name)
    
    model_classes = {}
    
    # Iterate over all modules in the 'models' directory (e.g., audio_flamingo.py)
    for _, module_name, _ in pkgutil.iter_modules(package.__path__):
        # Exclude special files (like __init__.py) and the base class file
        if not module_name.startswith('_') and module_name != 'base':
            try:
                # Dynamically import the module (e.g., 'almeval.models.audio_flamingo')
                module = importlib.import_module(f'{package_name}.{module_name}')
                
                # Find all classes in the module that are valid subclasses of BaseModel
                for _, cls in inspect.getmembers(module, inspect.isclass):
                    is_valid_model = (
                        issubclass(cls, BaseModel) and
                        cls is not BaseModel and
                        hasattr(cls, 'NAME') and
                        isinstance(cls.NAME, str)
                    )
                    if is_valid_model:
                        if cls.NAME in model_classes:
                            # Ensure no two classes have the same NAME
                            raise NameError(
                                f"Duplicate model NAME '{cls.NAME}' found in module "
                                f"'{module_name}'. Model names must be unique."
                            )
                        # Register the class using its unique NAME attribute
                        model_classes[cls.NAME] = cls
            except Exception as e:
                print(f"Warning: Failed to auto-discover models in module '{module_name}': {e}")
                
    return model_classes

# This dictionary is populated on startup and holds all discovered model classes.
_SUPPORTED_BASE_MODELS = _discover_models()

# --- Part 2: Load model configurations from the YAML file ---

def _load_model_configs() -> dict:
    """
    Loads and parses the 'models.yaml' file.
    
    Returns:
        A dictionary containing the configurations for all model aliases.
    """
    config_path = os.path.join(os.path.dirname(__file__), 'models.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            "Configuration file 'models.yaml' not found. Please create it in the "
            "'models' directory to define your model aliases."
        )
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# This dictionary is populated on startup and holds all alias configurations.
_MODEL_CONFIGS = _load_model_configs()

# --- Part 3: The public build_model factory function ---

def build_model(name: str, **kwargs) -> BaseModel:
    """
    Builds a model instance from a user-friendly alias defined in models.yaml.

    This factory function is the main entry point for creating models.

    Args:
        name (str): The alias of the model to build (e.g., 'af3', 'af3_think').
                    This must be a key in the `models.yaml` file.
        **kwargs: Additional arguments that will override or be added to the
                  arguments defined in the YAML config. This allows for dynamic
                  parameter changes at runtime.

    Returns:
        An instantiated model object that is a subclass of BaseModel.
    """
    # 1. Look up the alias in our loaded YAML configurations
    config = _MODEL_CONFIGS.get(name)
    if config is None:
        raise ValueError(
            f"Model alias '{name}' not found in models.yaml. "
            f"Available aliases are: {list(_MODEL_CONFIGS.keys())}"
        )

    # 2. Find the corresponding Python class via its unique NAME
    base_model_name = config.get('base_model')
    ModelClass = _SUPPORTED_BASE_MODELS.get(base_model_name)
    if ModelClass is None:
        raise ValueError(
            f"The base model '{base_model_name}' (defined for alias '{name}') "
            f"is not a registered model class. Check the 'NAME' attribute in your model files. "
            f"Available base models: {list(_SUPPORTED_BASE_MODELS.keys())}"
        )
    
    # 3. Prepare the final set of arguments for the class's __init__ method
    # Start with the default arguments from the YAML file's 'init_args'
    init_args = config.get('init_args', {})
    # Allow the user to override YAML arguments with any kwargs passed to this function
    init_args.update(kwargs)

    # 4. Instantiate the model and return it
    try:
        print(f"Building model '{ModelClass.__name__}' for alias '{name}' with args: {init_args}")
        model_instance = ModelClass(**init_args)
        return model_instance
    except Exception as e:
        print(f"ERROR: Failed to instantiate model '{ModelClass.__name__}' with args: {init_args}")
        # Reraise the exception to provide a full traceback to the user
        raise e