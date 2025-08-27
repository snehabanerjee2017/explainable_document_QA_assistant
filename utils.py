import yaml
from typing import Dict

def load_config(path_to_config:str)->Dict:
    with open(path_to_config, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    
    return data