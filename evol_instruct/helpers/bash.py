import subprocess

from evol_instruct.init.logger import logger

def run_bash_script(bash_script: str, args: list = [], cwd: str = None) -> subprocess.Popen:
    """
    Runs a bash script with the given arguments.

    Args:
        bash_script (str): The path to the bash script to be executed.
        args (list): A list of arguments to be passed to the bash script.

    Returns:
        subprocess.Popen: The Popen object representing the running process.
    """
    process = subprocess.Popen(
        [bash_script] + args, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        cwd=cwd)

    return process