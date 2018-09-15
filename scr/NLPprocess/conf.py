import os
import yaml

root_dir = os.path.dirname(__file__)

preprocess_yaml = yaml.load(open(os.path.join(root_dir, './preprocess.yml'), 'r', encoding='utf-8'))

class PreprocessConfig:
    def __init__(self):
        self.result = preprocess_yaml['result']
        self.str_process = preprocess_yaml['str_process']
        self.short_str_process = preprocess_yaml['short_str_process']
        self.str_to_enum = preprocess_yaml['str_to_enum']
        self.str_drop = preprocess_yaml['str_drop']
        self.float_process = preprocess_yaml['float_process']

preprocess_config = PreprocessConfig()

if __name__ == '__main__':
    pass