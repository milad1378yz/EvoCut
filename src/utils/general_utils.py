import yaml
import os
import requests
import pickle
from typing import Dict, List
import json
import copy
import pyomo.environ as pyo
from pyomo.environ import TransformationFactory
import cloudpickle
import logging


def load_pickle(file_path: str) -> object:
    """
    Load a Python object from a pickle file.
    Args:
        file_path (str): The path to the pickle file.
    Returns:
        object: The Python object loaded from the pickle file.
    """
    with open(file_path, "rb") as f:
        obj = pickle.load(f)
    return obj


def save_pickle(data: object, file_path: str) -> None:
    """
    Save a Python object to a pickle file.
    Args:
        data (object): The Python object to save.
        file_path (str): The path to the pickle file.
    """
    with open(file_path, "wb") as f:
        cloudpickle.dump(data, f)


def load_config(config_path: str) -> Dict:
    """
    Load a YAML configuration file.
    Args:
            config_path (str): The path to the YAML configuration file.
    Returns:
            dict: The configuration data loaded from the YAML file.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_json(file_path: str) -> Dict:
    """
    Load a JSON file.
    Args:
        file_path (str): The path to the JSON file.
    Returns:
        dict: The JSON data loaded from the file.
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def get_proxies() -> Dict:
    """
    Retrieves HTTP and HTTPS proxy settings from environment variables.
    Attempts a request to "http://www.google.com" using these proxies.
    Returns:
        dict: HTTP and HTTPS proxy settings.
    """
    http_proxy = os.environ.get("http_proxy", "No HTTP proxy set")
    https_proxy = os.environ.get("https_proxy", "No HTTPS proxy set")
    try:
        response = requests.get(
            "http://www.google.com", proxies={"http": http_proxy, "https": https_proxy}
        )
        if response.status_code == 200:
            logging.info("HTTP proxy is working correctly.")
        else:
            logging.warning(f"HTTP proxy returned status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP proxy connection failed: {e}")

    return {"http": http_proxy, "https": https_proxy}


def extract_json(text: str) -> Dict:
    """
    Extracts the JSON object from the given text.

    Args:
        text (str): The input text containing a JSON object.
    Returns:
        dict: The extracted JSON object.
    """
    # Remove evrything between <think> and </think>
    start = text.find("<think>")
    end = text.find("</think>")

    if start != -1 and end != -1:
        text = text[:start] + text[end + 8 :]

    start = text.find("```json")
    if start == -1:
        pass
    else:
        text = text[start + 7 :]

    end = text.rfind("```")
    if end == -1:
        pass
    else:
        text = text[:end]

    # remove all \n only at first of the string (maybe so many \n)
    text = text.lstrip("\n")

    try:
        json_data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to extract JSON data: {e}")
    return json_data


def extract_code(text: str) -> str:
    """
    Extracts the text between the first and last triple backticks (```).
    If no triple backticks are found, returns the entire input text.
    Removes 'python' after the first ``` if present.
    Args:
        text (str): The input text containing code.
    Returns:
        str: The extracted code.
    """
    start = text.find("```python")
    if start == -1:
        start = text.find("```")
        if start == -1:
            return text
        text = text[start + 3 :]
    else:
        text = text[start + 9 :]
    if start == -1:
        return text

    if text.startswith("\n"):
        text = text[1:]

    end = text.rfind("```")
    if end == -1:
        return text

    text = text[:end].rstrip()

    return extract_code(text)


def adopt_code_to_fucntion(sample_code: str, added_cut: str) -> str:
    """
    Adapts a given code snippet by inserting an additional code segment with proper indentation.
    Args:
        sample_code (str): The original code snippet where the additional code will be inserted.
        added_cut (str): The additional code segment to be inserted into the original code.
    Returns:
        str: The modified code snippet with the additional code segment properly indented and inserted.
    """
    # Determine correct indentation from code structure
    lines = sample_code.split("\n")
    tab_size = 4
    in_docstring = False

    # Find closing of docstring
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('"""'):
            if in_docstring:
                # Look for first code line after docstring
                for j in range(i + 1, len(lines)):
                    code_line = lines[j]
                    if code_line.strip():
                        tab_size = len(code_line) - len(code_line.lstrip())
                        break
                break
            else:
                in_docstring = True

    # If all lines in the added cut has tabs or multi spaces, remove from all lines
    if all(line.startswith(" ") or line.startswith("\t") for line in added_cut.split("\n")):
        # Determine the minimum indentation level in the added cut
        min_indent = min(
            (len(line) - len(line.lstrip()) for line in added_cut.split("\n") if line.strip()),
            default=0,
        )
        # Remove the minimum indentation level from all lines
        added_cut = "\n".join(line[min_indent:] for line in added_cut.split("\n"))

    # Apply consistent indentation to generated code
    added_cut_corrected = "\n".join(" " * tab_size + line for line in added_cut.split("\n"))

    sample_code = sample_code.format(added_constraint=added_cut_corrected)
    return sample_code


def remove_constraint(text: str) -> str:
    """
    Remove the line that has "{added_constraint}" from the given text.
    Args:
        text (str): The input text containing the code.
    Returns:
        str: The modified text with the line containing "{added_constraint}" removed.
    """
    # Copy simple code and remove the line that has"{added_constraint}"
    text = copy.deepcopy(text)
    text = text.split("\n")
    text = [line for line in text if "{added_constraint}" not in line]
    text = "\n".join(text)
    return text


def make_lp_relax_model(model: pyo.AbstractModel) -> pyo.AbstractModel:
    """
    Create a linear programming relaxation of a given model.
    Args:
        model (pyo.AbstractModel): The original Pyomo model.
    Returns:
        pyo.AbstractModel: A new Pyomo model with integer variables relaxed to continuous.
    """
    xfrm = TransformationFactory("core.relax_integer_vars")
    model_relax = copy.deepcopy(model)
    xfrm.apply_to(model_relax)
    return model_relax


def find_main_constraints(model: pyo.ConcreteModel) -> List[str]:
    """
    Find the unique main constraints in the model.
    Args:
        model (pyo.ConcreteModel): The Pyomo model.
    Returns:
        List[str]: A list of unique main constraint names in the model.
    """
    main_constraints = set()
    for constr in model.component_objects(pyo.Constraint, active=True):
        main_constraints.add(constr.name)
    return list(main_constraints)
