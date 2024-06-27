import os
import configparser
from pyprojroot import here
from huggingface_hub import hf_hub_download

from evol_instruct.init.logger import logger

config = configparser.ConfigParser()
config.read(here('evol_instruct/config/config.ini'))    

logger.info('Downloading generator model file %s from %s', config['model']['GeneratorModelGGMLFileName'], config['model']['GeneratorModel'])

# Model used for generating the dataset
generator_model_path = hf_hub_download(
    config['model']['GeneratorModel'],
    config['model']['GeneratorModelGGMLFileName']
)

logger.info('Generator model %s downloaded and is available at %s', config['model']['GeneratorModel'], generator_model_path)

# This is imported in the main file
evaluator_model_ggml_path = ""

# Model used for evaluating evolved instructions
if config.getboolean('model', 'isEvaluatorModelGGML'):
    logger.info('Downloading GGML evaluator model file %s from %s', config['model']['EvaluatorModelGGMLFileName'], config['model']['EvaluatorModel'])

    evaluator_model_ggml_path = hf_hub_download(
        config['model']['EvaluatorModel'],
        config['model']['EvaluatorModelGGMLFileName']
    )

    evaluator_model_gguf_path = os.path.join(os.path.dirname(evaluator_model_ggml_path), config['model']['EvaluatorModelGGUFFileName'])
else:
    logger.info('Downloading GGUF evaluator model file %s from %s', config['model']['EvaluatorModelGGUFFileName'], config['model']['EvaluatorModel'])
    
    evaluator_model_gguf_path = hf_hub_download(
        config['model']['EvaluatorModel'],
        config['model']['EvaluatorModelGGUFFileName']
    )

logger.info('Evaluator model %s downloaded and is available at %s', config['model']['EvaluatorModel'], os.path.dirname(evaluator_model_gguf_path))