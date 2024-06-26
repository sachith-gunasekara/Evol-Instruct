import subprocess
import multiprocessing
from pyprojroot import here

from evol_instruct.init.model import generator_model_path, evaluator_model_gguf_path
from evol_instruct.helpers.bash import run_cmd_and_get_output


def adjust_escape_characters(prompt):
    prompt = prompt.replace("\"", "\\\"")
    return prompt

def generate_from_generator_model(prompt, temp=0.8, timeout=600):
    prompt = adjust_escape_characters(prompt)
    cmd = f"{here('evol_instruct/workers/ggllm.cpp/falcon_main')} -t 8 -ngl 60 -b 512 --temp {temp} -m {generator_model_path} -p \"{prompt}\""

    # Create a multiprocessing Manager to share data between processes
    manager = multiprocessing.Manager()
    result = manager.dict()

    # Create a separate process to run the command
    process = multiprocessing.Process(target=run_cmd_and_get_output, args=(cmd, result))
    process.start()
    process.join(timeout=timeout)  # Wait for the process to finish or timeout

    if process.is_alive():
        # If the process is still running after the timeout, terminate it
        process.terminate()
        process.join()
        out = ""
        return out

    out = result.get('output', b'') \
        .decode("utf-8").replace("<|endoftext|>", "") \
        .split("<bot>:", 1)[1] \
        .strip().strip('\n').replace("\\\"", "\"")

    return out

def generate_from_evaluator_model(prompt):
    prompt = adjust_escape_characters(prompt)
    prompt = f"[INST]{prompt}[/INST]"
    cmd = f"{here('evol_instruct/workers/llama.cpp/llama-cli')} -c 2048 -n 30 -t 8 -ngl 41 -m {evaluator_model_gguf_path} -p \"{prompt}\""

    result = run_cmd_and_get_output(cmd, {})

    out = result.get('output', b'') \
        .decode("utf-8") \
        .strip().strip('\n') \
        .split("[/INST]", 1)[1].replace("\\\"", "\"")

    return out