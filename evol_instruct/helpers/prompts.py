import json
from pyprojroot import here


with open(here('evol_instruct/prompts.json'), 'r') as jsonfile:
    prompts = json.load(jsonfile)

def get_in_depth_evolving_base_prompt():
    return prompts['in_depth_evolving']['base']

def get_in_depth_evolving_prompt_with_add_constraint_operation():
    return get_in_depth_evolving_base_prompt() \
        .format(
            operation=prompts['in_depth_evolving']['operations']['add-constraints'], 
            instruction="{instruction}"
        )

def get_in_depth_evolving_prompt_with_deepening_operation():
    return get_in_depth_evolving_base_prompt() \
        .format(
            operation=prompts['in_depth_evolving']['operations']['deepening'], 
            instruction="{instruction}"
        )

def get_in_depth_evolving_prompt_with_concretizing_operation():
    return get_in_depth_evolving_base_prompt() \
        .format(
            operation=prompts['in_depth_evolving']['operations']['concretizing'], 
            instruction="{instruction}"
        )

def get_in_depth_evolving_prompt_with_increase_reasoning_steps_operation():
    return get_in_depth_evolving_base_prompt() \
        .format(
            operation=prompts['in_depth_evolving']['operations']['increase-reasoning-steps'], 
            instruction="{instruction}"
        )

def get_in_breadth_evolving_base_prompt():
    return prompts['in_breadth_evolving']

def get_equality_check_prompt(original_instruction, evolved_instruction):
    return prompts['equality_check_prompt'] \
        .format(
            original_instruction=original_instruction, 
            evolved_instruction=evolved_instruction
        )