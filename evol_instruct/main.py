import modal
from modal import Image

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
        modal.Mount.from_local_file('setup.py', '/root/setup.py'),
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
    import os

    os.environ['HF_HUB_CACHE'] = '/vol/.cache'

    from evol_instruct.init.logger import logger
    from evol_instruct.data.prepare import prepare_datasets
    from evol_instruct.init.model import evaluator_model_ggml_path, evaluator_model_gguf_path
    from evol_instruct.helpers.bash import run_bash_script
    from evol_instruct.helpers.generate import generate_from_evaluator_model, generate_from_generator_model

    # Prepare the generator model
    logger.info('Dispatching prepare_generator_model.sh to run in the background')

    subprocess.run(['ls', '-la'], cwd='/root/evol_instruct/scripts')

    prepare_generator_model_script = here('evol_instruct/scripts/prepare_generator_model.sh')
    subprocess.run(['chmod', '+x', prepare_generator_model_script], check=True)
    pgm_process = run_bash_script(prepare_generator_model_script, cwd=here('evol_instruct/workers'))

    # Prepare the evaluator model
    logger.info('Dispatching prepare_evaluator_model.sh to run in the background')

    prepare_evaluator_model_script = here('evol_instruct/scripts/prepare_evaluator_model.sh')
    subprocess.run(['chmod', '+x', prepare_evaluator_model_script], check=True)
    pem_process = run_bash_script(prepare_evaluator_model_script, args=['-i', evaluator_model_ggml_path, '-o', evaluator_model_gguf_path], cwd=here('evol_instruct/workers'))


    data = prepare_datasets()

    o, e = pgm_process.communicate()
    print(generate_from_generator_model('''<human>: What is the capital of France? Explain this to me with a history of how this came to be
<bot>: The capital of France'''))

    # o, e = pgm_process.communicate()

    # o, e = pem_process.communicate()


@app.local_entrypoint()
def main(
    run_on_remote: bool = True
):
    if run_on_remote:
        run_on_modal.remote()
    else:
        run_on_modal.local()