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



evol_config = configparser.ConfigParser()
evol_config.read(here('evol_instgruct/config/config.ini'))

class InstructionEvolver:
    def __init__(self, initial_instructions, config=None, ):
        self.pool = initial_instructions
        self.evolved_dataset = {
            'instruction': [],
            'response': [],
            'category': [],
            'evolution_strategy': [],
            'in-depth-evolving_operation': [],
            'epoch': []

        }
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

        print(f"Prompt: {self.config['prompt']}")
        return self

    def generate_example(self):
        logger.info("Generating example")
        instruction = generate_from_generator_model(self.config["prompt"]) \
            .replace("#Rewritten Prompt#:", "") \
            .replace("#Created Prompt#:", "")\
            .strip().strip('\n')
        logger.info('Instruction has been generated')      
        print('Instruction: ', instruction)

        logger.info("Generating response")
        response = generate_from_generator_model(f"<human>: {instruction}\n<bot>: ")
        logger.info('Response has been generated')
        print('Response: ', response)

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
            print(equality_check_prompt)
            model_output = generate_from_evaluator_model(equality_check_prompt).lower()
            print(model_output)

            if "not equal" in model_output:
                logger.info("Evolved instruction has information gain. Deduced in %i attempts", counter+2)
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

            return any(phrase in instruction for phrase in disallowed_phrases)

        if \
        has_information_gain(original_instruction, evolved_instruction) and \
        not is_response_difficult(response) and \
        not response_contains_only_punctuation_and_stop_words(response) and \
        not instruction_contains_disallowed_phrases(evolved_instruction):
            return True
        else:
            return False

    def save_dataset(
            self, 
            epoch, 
            category, 
            file_name_manual_epoch="", 
            file_name_append_tag=""):
        filename = os.path.join(
            evol_config['data']['Location'],
            "evolved",
            category,
            f"""{epoch if not file_name_manual_epoch else file_name_manual_epoch}_{self.config['strategy'][1]}{f"_{self.config['in_depth_evolution_operation'][1]}" if self.config['in_depth_evolution_operation'] else ''}{f"_{file_name_append_tag}" if file_name_append_tag else ""}.json""")

        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        with open(here(filename), "w") as f:
            json.dump(self.evolved_dataset, f)


    def check_and_save_dataset(
            self, 
            time0, 
            epoch, 
            category, 
            file_name_manual_epoch="", 
            file_name_append_tag=""):
        if len(self.evolved_dataset['instruction']) % 5 >= 0 or time() - time0 >= 300:
            logger.info("Saving...")

            self.save_dataset(epoch, category, file_name_manual_epoch, file_name_append_tag)
            time0 = time()
            return True, time0
        else:
            print(time() - time0)
            return False, time0

    def clear_evolved_instructions(self):
        self.evolved_dataset = {
            'instruction': [],
            'response': [],
            'category': [],
            'evolution_strategy': [],
            'in-depth-evolving_operation': [],
            'epoch': []

        }

    def evolve(
            self,
            time0,
            epochs,
            category, 
            file_name_manual_epoch="", 
            file_name_append_tag=""):
        for epoch in tqdm(range(epochs), desc="Evolving", unit="epoch"):
            new_pool = []

            self.select_evolution_strategy()
            if self.config["strategy"][0] == 0:
                self.select_in_depth_evolution_operation()

            for instruction in tqdm(self.pool, desc="Instruction", unit="instruction"):
                try:
                    evolved_instruction, response = self.evolve_instruction(instruction)
                    if self.has_instruction_evolved(instruction, evolved_instruction, response):
                        logger.info("Instruction Evolved")
                        print(f"Instruction Evolved: {evolved_instruction}\n\nResponse: {response}")

                        self.evolved_dataset['instruction'].append(evolved_instruction)
                        self.evolved_dataset['response'].append(response)
                        self.evolved_dataset['category'].append(category)
                        self.evolved_dataset['evolution_strategy'].append(self.config["strategy"][1])
                        if self.config["in_depth_evolution_operation"]:
                            self.evolved_dataset['in-depth-evolving_operation'].append(self.config["in_depth_evolution_operation"][1])
                        else:
                            self.evolved_dataset['in-depth-evolving_operation'].append("")
                        self.evolved_dataset['epoch'].append(epoch)

                        new_pool.append(evolved_instruction)

                        dataset_was_saved, time0 = self.check_and_save_dataset(time0, epoch, category, file_name_manual_epoch, file_name_append_tag)
                        if dataset_was_saved:
                            logger.info("Saved")
                    else:
                        logger.info("Instruction Not Evolved")
                        print(f"Instruction Not Evolved: {evolved_instruction}")

                        new_pool.append(instruction)

                        dump_enevolved_instructions(epoch, category, instruction, evolved_instruction, response)

                    clear_terminal()
                except Exception as e:
                    logger.error(e)

            self.save_dataset(epoch, category, file_name_manual_epoch, file_name_append_tag)
            self.pool = new_pool
            self.clear_evolved_instructions()