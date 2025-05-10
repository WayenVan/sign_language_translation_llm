from git import Repo
from pathlib import Path
import subprocess
import datetime
import os

import logging

logger = logging.getLogger(__name__)


def save_git_info(repo_path, info_path):
    """
    Save git information to files
    """
    os.makedirs(info_path, exist_ok=True)
    info = get_git_info(repo_path)
    remote_info = get_remote_info(repo_path)
    commit_file = Path(info_path).joinpath("commit.txt")
    branch_file = Path(info_path).joinpath("branch.txt")
    dirty_file = Path(info_path).joinpath("git-dirty.patch")

    if remote_info["remotes"] is not None:
        remote_file = Path(info_path).joinpath("remotes.txt")
        with open(remote_file, "w") as f:
            for remote in remote_info["remotes"]:
                f.write(f"{remote['name']}: {', '.join(remote['urls'])}\n")

    with open(commit_file, "w") as f:
        f.write(info["commit"])
    with open(branch_file, "w") as f:
        f.write(info["branch"])
    if info["dirty"]:
        save_dirty_patch(repo_path, patch_path=dirty_file)
    logger.info(f"Git info saved to: {info_path}")


def get_remote_info(repo_path="."):
    repo = Repo(repo_path)

    remotes_info = []
    for remote in repo.remotes:
        urls = list(remote.urls)
        remotes_info.append({"name": remote.name, "urls": urls})

    # 当前分支跟踪的远程分支（如果有）
    if not repo.head.is_detached:
        active_branch = repo.active_branch
        tracking_branch = active_branch.tracking_branch()
        tracking = tracking_branch.name if tracking_branch else None
    else:
        tracking = None

    return {"remotes": remotes_info, "tracking": tracking}


def get_git_info(repo_path):
    repo = Repo(repo_path)
    branch = repo.active_branch.name if not repo.head.is_detached else "DETACHED"
    commit = repo.head.commit.hexsha[:7]
    dirty = repo.is_dirty()
    untracked = repo.untracked_files

    return {
        "branch": branch,
        "commit": commit,
        "dirty": dirty,
        "untracked": bool(untracked),
        "detached": repo.head.is_detached,
    }


def save_dirty_patch(repo_path, patch_path="git-dirty.patch"):
    repo = Repo(repo_path)

    # 生成唯一 patch 文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    patch_file = Path(patch_path).with_name(
        f"{Path(patch_path).stem}-{timestamp}.patch"
    )

    # 保存 staged + unstaged 改动
    if repo.is_dirty(untracked_files=True):
        with open(patch_file, "w") as f:
            subprocess.run(["git", "diff"], cwd=repo_path, stdout=f)
            subprocess.run(["git", "diff", "--cached"], cwd=repo_path, stdout=f)

        logger.info(f"Dirty patch saved to: {patch_file}")
    else:
        logger.info("No dirty changes found. Working directory is clean.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    save_dirty_patch(".", patch_path="outputs/git-dirty.patch")
