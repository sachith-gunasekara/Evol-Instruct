import configparser
import json

from evol_instruct.init.logger import logger
from evol_instruct.helpers.ei_os import get_path

config = configparser.ConfigParser()
config.read(get_path('config/config.ini'))

datasets = json.loads(config['datasets']['seed'])

logger.info('Seed dataset configuration loaded')