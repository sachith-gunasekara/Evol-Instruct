import configparser
from pyprojroot import here
from time import time

from evol_instruct.init.logger import logger


config = configparser.ConfigParser()
config.read(here('evol_instruct/config/config.ini'))

def dump_enevolved_instructions(epoch, category, instruction, evolved_instruction, response):
    with open(here(os.path.join(config['data']['Location'], "unevolved_instructions.txt")), "a") as f:
        f.write("------------------------------------------------------------------------------\n")
        f.write(f"{epoch}, {category}\n")
        f.write("Instruction Not Evolved\n")
        f.write("------------------------------------------------------------------------------\n")

        f.write(f"{instruction}\n")
        f.write("========================================\n")
        f.write(f"{evolved_instruction}\n")
        f.write("========================================\n")
        f.write(f"{response}\n")
        f.write("\n\n\n")


def evolve_category(
        epochs,
        category,  
        start, 
        end, 
        data,
        file_name_manual_epoch="", 
        starting_data=None
    ):
    from evol_instruct.instruction_evolver import InstructionEvolver

    file_name_append_tag = f"{start}-{end}"

    if starting_data:
        evolve_data = starting_data
    else:
        category_data = data.filter(lambda x: x['category'] == category)
        evolve_data = category_data['instruction'][start:end]

    category_evolver = InstructionEvolver(evolve_data)

    print(len(category_evolver.pool), category_evolver.pool[:2])

    time0 = time()
    category_evolver.evolve(
        time0,
        epochs,
        category,
        file_name_manual_epoch,
        file_name_append_tag
    )
