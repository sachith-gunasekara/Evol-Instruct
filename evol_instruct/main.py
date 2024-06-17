from pyprojroot import here
import subprocess

from evol_instruct.init.logger import logger
from evol_instruct.init.datasets import datasets
from evol_instruct.init.model import evaluator_model_ggml_path, evaluator_model_gguf_path
from evol_instruct.helpers.bash import run_bash_script


# Prepare the generator model
logger.info('Dispatching prepare_generator_model.sh to run in the background')

prepare_generator_model_script = here('evol_instruct/scripts/prepare_generator_model.sh')
subprocess.run(['chmod', '+x', prepare_generator_model_script], check=True)
pgm_process = run_bash_script(prepare_generator_model_script, cwd=here('evol_instruct/workers'))

# Prepare the evaluator model
logger.info('Dispatching prepare_evaluator_model.sh to run in the background')

prepare_evaluator_model_script = here('evol_instruct/scripts/prepare_evaluator_model.sh')
subprocess.run(['chmod', '+x', prepare_evaluator_model_script], check=True)
pem_process = run_bash_script(prepare_evaluator_model_script, args=['-i', evaluator_model_ggml_path, '-o', evaluator_model_gguf_path], cwd=here('evol_instruct/workers'))

o, e = pgm_process.communicate()
print(o)
print(e)
print('Generator model is ready!')

o, e = pem_process.communicate()
print(o)
print(e)
print('Evaluator model is ready!')