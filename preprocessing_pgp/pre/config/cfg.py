import configparser
import os
import json

CONFIG_PATH = '/bigdata/fdp/cdp/script/config'

def build_path(file):
    return os.path.join(CONFIG_PATH, file)

class Config:
    def __init__(self):
        # TODO: read ini files using glob to exclude .ini.swp
        self._cfg_files = [file for file in os.listdir(CONFIG_PATH) if 'ini' in file]
        self.cfg = {}
        
        self.__load_builtin_cfgs()
        
    @property
    def cfg_files(self):
        return self._cfg_files
    
    def load_cfg(self, config_file: str, config_name: str):
        try:
            cfg = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
            cfg.read(config_file)
        except UnicodeDecodeError:
            raise Exception(f'Error when loading {config_file}')
        self.cfg[config_name] = cfg
    
    def __load_builtin_cfgs(self):
        for config_file in self.cfg_files:
            config_name = config_file.replace('.ini', '')
            self.load_cfg(build_path(config_file), config_name)  
    
    def get(self, config_name:str, section:str=None, option:str=None, default:str=None):
        if option is None:
            if section is None:
                return [{s: [o for o in self.cfg[config_name][s]]} for s in self.cfg[config_name] if s != 'DEFAULT']
            return [o for o in self.cfg[config_name][section]]
                
        try:
            result = json.loads(self.cfg[config_name].get(section, option, fallback=default)) 
        except:
            result = self.cfg[config_name].get(section, option, fallback=default)
        return result
        
   
