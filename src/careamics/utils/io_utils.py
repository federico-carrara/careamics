from datetime import datetime
import json
import glob
from pathlib import Path
import pickle
import os
from typing import Any, Literal, Union

import git


def get_new_model_version(model_dir: Union[Path, str]) -> int:
    """Create a unique version ID for a new model run."""
    versions = []
    for version_dir in os.listdir(model_dir):
        try:
            versions.append(int(version_dir))
        except:
            print((
                f"Invalid subdirectory:{model_dir}/{version_dir}. " 
                "Only integer versions are allowed."
            ))
            exit()
    if len(versions) == 0:
        return "0"
    return f"{max(versions) + 1}"


def get_workdir(
    root_dir: str,
    model_name: str,
) -> tuple[Path, Path]:
    """Get the workdir for the current model.

    It has the following structure: "root_dir/YYMM/model_name/version".
    """
    os.makedirs(root_dir, exist_ok=True)
    
    rel_path = datetime.now().strftime("%y%m")
    cur_workdir = os.path.join(root_dir, rel_path)
    Path(cur_workdir).mkdir(exist_ok=True)

    rel_path = os.path.join(rel_path, model_name)
    cur_workdir = os.path.join(root_dir, rel_path)
    Path(cur_workdir).mkdir(exist_ok=True)

    rel_path = os.path.join(rel_path, get_new_model_version(cur_workdir))
    cur_workdir = os.path.join(root_dir, rel_path)
    try:
        Path(cur_workdir).mkdir(exist_ok=False)
    except FileExistsError:
        print(f"Workdir {cur_workdir} already exists.")
    return cur_workdir, rel_path


def get_git_status() -> dict[Any]:
    """Get current git status."""
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    repo = git.Repo(curr_dir, search_parent_directories=True)
    git_config = {}
    git_config["changedFiles"] = [item.a_path for item in repo.index.diff(None)]
    git_config["branch"] = repo.active_branch.name
    git_config["untracked_files"] = repo.untracked_files
    git_config["latest_commit"] = repo.head.object.hexsha
    return git_config


def get_model_checkpoint(
    ckpt_dir: str, mode: Literal['best', 'last'] = 'best'
) -> str:
    """Get the model checkpoint path.
    
    Parameters
    ----------
    ckpt_dir : str
        Checkpoint directory.
    mode : Literal['best', 'last'], optional
        Mode to get the checkpoint, by default 'best'.
    
    Returns
    -------
    str
        Checkpoint path.
    """
    output = []
    for fpath in glob.glob(ckpt_dir + "/*.ckpt"):
        fname = os.path.basename(fpath)
        if mode == 'best':
            if fname.startswith('best'):
                output.append(fpath)
        elif mode == 'last':
            if fname.startswith('last'):
                output.append(fpath)
    assert len(output) == 1, '\n'.join(output)
    return output[0]


def load_file(file_path: str):
    """Load a file with the appropriate method based on the file extension.
    
    Parameters
    ----------
    file_path : str
        File path.
    """
    # Get the file extension
    _, ext = os.path.splitext(file_path)

    # Check the extension and load the file accordingly
    if ext == '.pkl':
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    elif ext == '.json':
        with open(file_path) as f:
            return json.load(f)
    else:
        raise ValueError(
            f"Unsupported file extension: {ext}. Only .pkl and .json are supported."
        )


def load_config(
    config_fpath: str, 
    config_type: Literal['algorithm', 'training', 'data']
) -> dict:
    """Load a configuration file.
    
    Parameters
    ----------
    config_fpath : str
        Configuration file path.
    config_type : Literal['algorithm', 'training', 'data']
        Configuration type.
    """
    for fname in glob.glob(os.path.join(config_fpath, '*config.*')):
        fname = os.path.basename(fname)
        if fname.startswith(config_type):
            return load_file(os.path.join(config_fpath, fname))
    raise ValueError(f"Config file not found in {config_fpath}.")