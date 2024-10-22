from datetime import datetime
from pathlib import Path
import os
from typing import Any, Union

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
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    repo = git.Repo(curr_dir, search_parent_directories=True)
    git_config = {}
    git_config["changedFiles"] = [item.a_path for item in repo.index.diff(None)]
    git_config["branch"] = repo.active_branch.name
    git_config["untracked_files"] = repo.untracked_files
    git_config["latest_commit"] = repo.head.object.hexsha
    return git_config