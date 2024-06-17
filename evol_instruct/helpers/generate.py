import subprocess
import multiprocessing
from pyprojroot import here

from evol_instruct.init.model import generator_model_path, evaluator_model_gguf_path
from evol_instruct.init.logger import logger

def generate_from_generator_model(prompt, temp=0.8, timeout=600):
    prompt = prompt.replace("\"", "\\\"")
    cmd = f"{here('evol_instruct/workers/ggllm.cpp/falcon_main')} -t 2 -ngl 20 -b 512 --temp {temp} -m {generator_model_path} -p \"{prompt}\""

    def run_cmd(cmd, result):
        try:
            output = subprocess.check_output(cmd, shell=True)
            result['output'] = output
        except subprocess.CalledProcessError as e:
            result['error'] = str(e)

    # Create a multiprocessing Manager to share data between processes
    manager = multiprocessing.Manager()
    result = manager.dict()

    # Create a separate process to run the command
    process = multiprocessing.Process(target=run_cmd, args=(cmd, result))
    process.start()
    process.join(timeout=timeout)  # Wait for the process to finish or timeout

    if process.is_alive():
        # If the process is still running after the timeout, terminate it
        process.terminate()
        process.join()
        return ""

    if 'error' in result:
        return ""

    out = result.get('output', b'') \
        .decode("utf-8").replace("<|endoftext|>", "") \
        .strip().strip('\n') \
        .split("<bot>:", 1)[1].strip().strip('\n').replace("\\\"", "\"")

    return out

def generate_from_evaluator_model(prompt):
    prompt = prompt.replace("\"", "\\\"")
    prompt = f"<s>[INST]{prompt}[/INST]"
    cmd = f"{here('evol_instruct/workers/llama.cpp/main')} -c 2048 -t 2 -ngl 20 -m {evaluator_model_gguf_path} -p \"{prompt}\""

    try:
        output = subprocess.check_output(cmd, shell=True)
    except subprocess.CalledProcessError:
        return ""

    out = output.decode("utf-8").strip().strip('\n').split("[/INST]", 1)[1].replace("\\\"", "\"")

    return out