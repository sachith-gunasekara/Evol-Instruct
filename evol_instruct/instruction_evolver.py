import os
import random
import string
import json
import configparser
from time import time

import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
from pyprojroot import here

from evol_instruct.helpers.bash import clear_terminal
from evol_instruct.init.logger import logger
from evol_instruct.helpers.generate import generate_from_generator_model, generate_from_evaluator_model
from evol_instruct.helpers.prompts import (
    get_in_depth_evolving_prompt_with_add_constraint_operation,
    get_in_depth_evolving_prompt_with_deepening_operation,
    get_in_depth_evolving_prompt_with_concretizing_operation,
    get_in_depth_evolving_prompt_with_increase_reasoning_steps_operation,
    get_in_breadth_evolving_base_prompt,
    get_equality_check_prompt
)
from evol_instruct.helpers.evolver import dump_enevolved_instructions
from evol_instruct.dataset import Dataset



evol_config = configparser.ConfigParser()
evol_config.read(here('evol_instgruct/config/config.ini'))

class InstructionEvolver:
    def __init__(self, initial_instructions: list[str]):
        self.pool = initial_instructions
        self.evolved_dataset = None # Will initialize later on
        self.config = {
            "strategy": None,
            "in_depth_evolution_operation": None,
            "prompt": None
        }

    def select_evolution_strategy(self):
        logger.info("Selecting evolution strategy")
        strategies = [
            (0, "in-depth-evolving"),
            (1, "in-breadth-evolving"),
        ]

        self.config["strategy"] = random.choice(strategies)
        logger.info("Evolution strategy: %s", self.config['strategy'][1])

        return self

    def select_in_depth_evolution_operation(self):
        logger.info("Selecting in-depth evolution operation")
        operations = [
            (0, "add-constraints"),
            (1, "deepening"),
            (2, "concretizing"),
            (3, "increase-reasoning-steps")
        ]

        self.config["in_depth_evolution_operation"] = random.choice(operations)
        logger.info("In-depth evolution operation: %s", self.config['in_depth_evolution_operation'][1])

        return self

    def generate_prompt(self, instruction):
        match self.config["strategy"][0]:
            case 0:
                match self.config['in_depth_evolution_operation'][0]:
                    case 0:
                        self.config["prompt"] = get_in_depth_evolving_prompt_with_add_constraint_operation()
                    case 1:
                        self.config["prompt"] = get_in_depth_evolving_prompt_with_deepening_operation()
                    case 2:
                        self.config["prompt"] = get_in_depth_evolving_prompt_with_concretizing_operation()
                    case 3:
                        self.config["prompt"] = get_in_depth_evolving_prompt_with_increase_reasoning_steps_operation()

            case 1:
                self.config["prompt"] = get_in_breadth_evolving_base_prompt()

        self.config["prompt"] = self.config["prompt"].format(instruction=instruction)

        logger.debug('Prompt: %s', self.config["prompt"])
        return self

    def generate_example(self):
        logger.info("Generating example")

        instruction = generate_from_generator_model(self.config["prompt"]) \
            .replace("#Rewritten Prompt#:", "") \
            .replace("#Created Prompt#:", "")\
            .strip().strip('\n')
        
        logger.info('Instruction has been generated')
        logger.debug('Instruction: %s', instruction)
        logger.info("Generating response")

        response = generate_from_generator_model(f"<human>: {instruction}\n<bot>: ")

        logger.info('Response has been generated')
        logger.debug('Response: %s', response)

        return instruction, response

    def evolve_instruction(self, instruction):
        return self \
            .generate_prompt(instruction) \
            .generate_example()

    def has_instruction_evolved(self, original_instruction, evolved_instruction, response):

        def has_information_gain(original_instruction, evolved_instruction, counter=0):
            if counter > 5:
                return False

            logger.info("Checking for information gain in the evolved instruction")

            equality_check_prompt = get_equality_check_prompt(original_instruction, evolved_instruction)

            logger.debug('Equality check prompt: %s', equality_check_prompt)

            model_output = generate_from_evaluator_model(equality_check_prompt).lower()

            logger.debug('Model output: %s', model_output)

            if "not equal" in model_output:
                logger.info("Evolved instruction has information gain. Deduced in %i attempts", counter+1)
                return True
            elif "equal" in nltk.word_tokenize(model_output):
                logger.info("Evolved instruction has no information gain")
                return False
            else:
                logger.info("Could not deduce information gain. Checking again. Attempt: %i", counter+2)
                return has_information_gain(original_instruction, evolved_instruction, counter+1)


        def is_response_difficult(response):
            return 'sorry' in response and len(nltk.word_tokenize(response)) < 80

        def response_contains_only_punctuation_and_stop_words(response):
            stop_words = set(stopwords.words('english'))
            words = nltk.word_tokenize(response)
            return all(word in stop_words or word in string.punctuation for word in words)

        def instruction_contains_disallowed_phrases(instruction):
            disallowed_phrases = [
                "#Given Prompt#", "#Created Prompt#", "#Rewritten Prompt#",
                "given prompt", "created prompt", "rewritten prompt"]

            return any(phrase.lower() in instruction.lower() for phrase in disallowed_phrases)

        if \
        not is_response_difficult(response) and \
        not response_contains_only_punctuation_and_stop_words(response) and \
        not instruction_contains_disallowed_phrases(evolved_instruction) and\
        has_information_gain(original_instruction, evolved_instruction):
            return True
        else:
            return False

    def evolve(
            self,
            epochs,
            category, 
            file_name_manual_epoch='', 
            file_name_append_tag=''
        ):
        for epoch in tqdm(range(epochs), desc="Evolving", unit="epoch"):
            new_pool = []

            self.select_evolution_strategy()
            if self.config["strategy"][0] == 0:
                self.select_in_depth_evolution_operation()
            else:
                self.config['in_depth_evolution_operation'] = None

            self.evolved_dataset = Dataset(
                filename_in_disk=Dataset.generate_filename(
                    epoch, 
                    category, 
                    file_name_manual_epoch, 
                    file_name_append_tag,
                    self.config['strategy'][1],
                    self.config['in_depth_evolution_operation'][1] if self.config['in_depth_evolution_operation'] else ''
                )
            )

            for instruction in tqdm(self.pool, desc="Instruction", unit="instruction"):
                try:
                    evolved_instruction, response = self.evolve_instruction(instruction)
                    if self.has_instruction_evolved(instruction, evolved_instruction, response):
                        logger.info("Instruction Evolved")
                        logger.debug('Evolved instruction: %s, Response: %s', evolved_instruction, response)

                        self.evolved_dataset.add_data(
                            evolved_instruction,
                            response,
                            category,
                            self.config['strategy'][1],
                            self.config['in_depth_evolution_operation'][1] if self.config['in_depth_evolution_operation'] else '',
                            epoch
                        )

                        new_pool.append(evolved_instruction)
                    else:
                        logger.info("Instruction Not Evolved")
                        logger.debug('Instruction Not Evolved: %s', instruction)

                        new_pool.append(instruction)

                        dump_enevolved_instructions(epoch, category, instruction, evolved_instruction, response)

                    clear_terminal()
                except Exception as e:
                    logger.error(e)

            self.evolved_dataset.save()
            self.pool = new_pool