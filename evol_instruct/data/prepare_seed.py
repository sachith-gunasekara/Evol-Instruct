
import configparser
from pyprojroot import here
from transformers import AutoTokenizer
from datasets import load_dataset
import nltk

from evol_instruct.init.logger import logger
from evol_instruct.init.datasets import datasets

logger.info('Downloading punkt and stopwords from nltk')
nltk.download('punkt')
nltk.download('stopwords')

def prepare_seed_datasets():
    config = configparser.ConfigParser()
    config.read(here('evol_instruct/config/config.ini'))

    logger.info('Loading tokenizer: %s', config['tokenizer']['GeneratorTokenizer'])
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer']['GeneratorTokenizer'])

    def preprocess_dataset(example):
        instruction = f"{example['instruction']}\n\n{example['context']}" if example['context'] else f"{example['instruction']}"
        return {'instruction': tokenizer.decode(tokenizer.encode(instruction, max_length=1024, truncation=True))}

    logger.info('Loading dataset: %s. Truncating to 1024 tokens', datasets[0])
    data = load_dataset(
        datasets[0],
        split="train",
    ).map(
        preprocess_dataset,
        remove_columns=['context', 'response'],
        num_proc=2
    )

    return data