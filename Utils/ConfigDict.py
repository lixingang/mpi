import yaml
class ConfigDict(dict):
    """
    Makes a  dictionary behave like an object,with attribute-style access.
    """
    def __init__(self, yaml_config):
        config = None
        with open(yaml_config, 'r') as f:
            config = yaml.load(f,Loader=yaml.Loader)
        for key in config:
            setattr(self, key, config[key])
            
    def __getattr__(self,name):
        try:
            return self[name]
        except:
            raise AttributeError(name)
    def __setattr__(self,name,value):
        self[name]=value

