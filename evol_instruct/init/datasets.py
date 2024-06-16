import configparser
import json
from pyprojroot import here


from evol_instruct.init.logger import logger

config = configparser.ConfigParser()
config.read(here('evol_instruct/config/config.ini'))

datasets = json.loads(config['datasets']['seed'])

logger.info('Seed dataset configuration loaded')