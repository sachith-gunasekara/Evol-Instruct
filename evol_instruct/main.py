import modal
from modal import Image
from pyprojroot import here

# Initialize modal app
app = modal.App()

# Setup volume for storing model weights
volume = modal.Volume.from_name("evol-instruct", create_if_missing=True)
MODEL_DIR = "/vol"

GPU = 'T4'

image = (
    Image.from_registry("thr3a/cuda12.1-torch")
    .poetry_install_from_file(poetry_pyproject_toml="pyproject.toml")
)


@app.function(
    gpu=GPU,
    timeout=86400, # Allow one day timout period
    image=image,
    mounts=[
        modal.Mount.from_local_python_packages('evol_instruct'),
    ],
    volumes={
        MODEL_DIR: volume
    },
    _allow_background_volume_commits=True
)
def run_on_modal():
    from pyprojroot import here
    import subprocess

    subprocess.run(['ls', '-la'])

    exit()

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
    print('Output++++++++++++++++++++++++++++++++' + o)
    print('Error++++++++++++++++++++++++++++++++' + e)
    print('Generator model is ready!')

    o, e = pem_process.communicate()
    print('Output++++++++++++++++++++++++++++++++' + o)
    print('Error++++++++++++++++++++++++++++++++' + e)
    print('Evaluator model is ready!')


@app.local_entrypoint()
def main():
    run_on_modal.remote()