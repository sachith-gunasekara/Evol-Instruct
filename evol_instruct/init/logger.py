import logging
import configparser
from pyprojroot import here

config = configparser.ConfigParser()
config.read(here('evol_instruct/config/config.ini'))

logger = logging.getLogger(config['logger']['name'])

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler(config['logger']['file'])

c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

logger.setLevel(logging.DEBUG)

logger.info('Logger initialized')