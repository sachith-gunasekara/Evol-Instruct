import os

def get_path(path: str) -> str:
    """
    Get the absolute path by joining the base path with the given path.

    Parameters:
        path (str): The relative path to be joined with the base path.

    Returns:
        str: The absolute path obtained by joining the base path with the given path.
    """

    base_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(base_path, '..', path)