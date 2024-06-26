import subprocess
import platform

def run_bash_script_in_background(bash_script: str, args: list = None, cwd: str = None) -> subprocess.Popen:
    """
    Runs a bash script with the given arguments.

    Args:
        bash_script (str): The path to the bash script to be executed.
        args (list): A list of arguments to be passed to the bash script.

    Returns:
        subprocess.Popen: The Popen object representing the running process.
    """
    process = subprocess.Popen(
        [bash_script] + args if args else [bash_script], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        cwd=cwd)

    return process

def run_cmd_and_get_output(cmd: str, result: dict):
    """
    Runs the given command using the subprocess module and captures the output.

    Args:
        cmd (str): The command to be executed.
        result (dict): A dictionary to store the output or the error message.

    Returns:
        None

    Raises:
        subprocess.CalledProcessError: If the command execution fails.

    This function executes the given command using the subprocess module and captures the output. If the command executes successfully, the output is stored in the 'output' key of the 'result' dictionary. If the command execution fails, the error message is stored in the 'error' key of the 'result' dictionary.
    """
    try:
        output = subprocess.check_output(cmd)
        result['output'] = output
    except subprocess.CalledProcessError as e:
        result['error'] = str(e)
    
    return result

def clear_terminal():
    subprocess.Popen("cls" if platform.system() == "Windows" else "clear", shell=True)
