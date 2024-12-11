import yaml
from types import SimpleNamespace

def load_config_as_namespace(config_file):
    with open(config_file, "r") as file:
        config_dict = yaml.safe_load(file)
    return convert_dict_to_namespace(config_dict)

def convert_dict_to_namespace(d):
    """Recursively converts a dictionary into a SimpleNamespace."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: convert_dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [convert_dict_to_namespace(item) for item in d]
    else:
        return d
