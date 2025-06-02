import os
import tempfile
import pandas as pd
from huggingface_hub import HfApi, create_repo, upload_file
from huggingface_hub.utils import RepositoryNotFoundError


def ensure_hf_repo(repo_name: str, token: str, private: bool = False) -> str:
    """
    Ensure that a Hugging Face *dataset* repo with `repo_name` exists.
    If it does not exist, create it (as a dataset).
    Returns the full repo ID string, e.g. "username/flowmatic_dataset".
    """
    api = HfApi()
    user = api.whoami(token=token)["name"]
    full_repo_id = f"{user}/{repo_name}"

    # Check if the dataset repo exists; if not, create it as a dataset
    try:
        api.repo_info(repo_id=full_repo_id, token=token, repo_type="dataset")
    except RepositoryNotFoundError:
        create_repo(
            repo_id=full_repo_id,
            token=token,
            private=private,
            repo_type="dataset",
        )

    return full_repo_id


def push_df_to_hf(
    df: pd.DataFrame,
    repo_name: str,
    token: str,
    path_in_repo: str = "cleaned_data.csv",
    commit_message: str = "Add cleaned data",
    branch: str = "main",
) -> None:
    """
    Convert `df` to a temporary CSV file, then push it to the given HF repo
    under `path_in_repo`. If the repo does not exist, it will be created
    automatically (as a dataset).
    """
    full_repo_id = ensure_hf_repo(repo_name, token)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name
        df.to_csv(tmp_path, index=True)

    upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo=path_in_repo,
        repo_id=full_repo_id,
        token=token,
        commit_message=commit_message,
        repo_type="dataset",
        create_pr=False,
        revision=branch,
    )

    try:
        os.remove(tmp_path)
    except OSError:
        pass
