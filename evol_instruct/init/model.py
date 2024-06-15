import os
import configparser
from huggingface_hub import hf_hub_download

from evol_instruct.init.logger import logger
from evol_instruct.helpers.ei_os import get_path

config = configparser.ConfigParser()
config.read(get_path('config/config.ini'))

# Model used for generating the dataset
generator_model_path = hf_hub_download(
    config['model']['GeneratorModel'],
    config['model']['GeneratorModelGGMLFileName']
)

# Model used for evaluating evolved instructions
evaluator_model_ggml_path = hf_hub_download(
    config['model']['EvaluatorModel'],
    config['model']['EvaluatorModelGGMLFileName']
)

evaluator_model_gguf_path = os.path.join(os.path.dirname(evaluator_model_ggml_path), config['model']['EvaluatorModelGGUFFileName'])