import json
from robomimic.config import config_factory


HOME_DIR = "/home/wjung85/Repo/projects/robomimic-safediffusion/safediffusion"
RESULT_DIR = f"{HOME_DIR}/results"


def load_config_from_json(json_path):
    """
    Load the Config object from the json file.

    It does by loading the base config using algo_name and overwrite the settings with the json file.
    """
    ext_cfg = json.load(open(json_path, "r"))
    config = config_factory(ext_cfg["algo_name"])
    with config.values_unlocked():
        config.update(ext_cfg)

    return config