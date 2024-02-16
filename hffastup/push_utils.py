# Copyright 2020 The HuggingFace Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python3
# Multi-processing upload to HF Hub:
# Example usage:
# import push_utils
# import datasets
# datasets.config.MAX_SHARD_SIZE = "2GB"
# datasets.logging.set_verbosity_info()
# dataset = datasets.load_dataset("fsd50k.py")
# push_utils.upload_to_hf_hub(dataset, "CPJKU/fsd50k")
# push_utils.push_dataset_card(dataset, "CPJKU/fsd50k")
#

import os.path
import tempfile
from typing import Union
from huggingface_hub import HfApi, CommitOperationAdd
import os
from multiprocessing import Process, Queue
import time


import datasets
from datasets.utils import logging
from huggingface_hub import (
    CommitOperationAdd,
    DatasetCard,
    DatasetCardData,
    HfApi,
)
from pathlib import Path
from datasets.utils.metadata import MetadataConfigs
from datasets.info import DatasetInfo, DatasetInfosDict
from datasets.splits import SplitDict, SplitInfo
from datasets import DatasetDict
from datasets.utils.py_utils import convert_file_size_to_int
from regex import X

datasets.config.MAX_SHARD_SIZE = "2GB"
datasets.logging.set_verbosity_info()
logger = datasets.logging.get_logger()

if not os.getenv("HF_HUB_ENABLE_HF_TRANSFER"):
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    logger.warning(
        "Warning: HF_HUB_ENABLE_HF_TRANSFER env variable is not set, defaulting to 1 to push the dataset faster"
    )


def set_verbosity(verbosity: int) -> None:
    """Set the level for the Hugging Face Datasets library's root logger.
    Args:
        verbosity:
            Logging level, e.g., `datasets.logging.DEBUG` and `datasets.logging.INFO`.
    """
    datasets.logging.setLevel(verbosity)


def get_shard_prefix(split):
    return "shard_" + split


api = HfApi()


def upload_batch(repo_id, in_repo, paths):
    pid = os.getpid()
    if not len(in_repo):
        return
    attempt = 0
    while attempt < 5:
        try:
            operations = [
                CommitOperationAdd(
                    path_or_fileobj=path_or_fileobj,
                    path_in_repo=path_in_repo,
                )
                for path_or_fileobj, path_in_repo in zip(paths, in_repo)
            ]
            if api.file_exists(repo_id, in_repo[0], repo_type="dataset"):
                logger.info(
                    f"proc {pid}: Upload attempt={attempt} file exists {paths[0]}. Reuploading anyway"
                )
            commit_info = api.create_commit(
                repo_id=repo_id,
                repo_type="dataset",
                operations=operations,
                commit_message=f"upload {in_repo}",
            )
            logger.info(
                "\nproc %s: Done in attempt %s! %s to %s\n"
                % (
                    os.getpid(),
                    attempt,
                    str(paths),
                    repo_id,
                )
            )
            break
        except Exception as e:
            logger.warning(
                f"proc {pid}: Upload attempt={attempt} for {paths} failed.", exc_info=1
            )
            attempt += 1
            time.sleep(5)
            continue
    if attempt >= 5:
        logger.critical(
            f"proc {pid}: Upload failed after {attempt} attempts for {paths}",
            exc_info=1,
        )
        with open("failed_uploads.txt", "a") as f:
            for path in paths:
                f.write(path + "," + repo_id + "\n")


def upload_proc(batch_size, queue, delete_local=True, repo_save_path="data/"):
    """Read from the queue; this spawns as a separate Process"""

    logger.info("proc %s: started!" % (os.getpid()))
    current_batch_paths = []
    current_batch_repo_paths = []
    while True:
        msg, args = queue.get()  # Read from the queue
        if (msg == "DONE" and len(current_batch_paths)) or len(
            current_batch_paths
        ) >= batch_size:
            upload_batch(repo_id, current_batch_repo_paths, current_batch_paths)
            if delete_local:
                for p in current_batch_paths:
                    logger.info("proc %s: Deleting %s" % (os.getpid(), p))
                    os.remove(p)
            current_batch_paths = []
            current_batch_repo_paths = []
        if msg == "DONE":
            break
        shard_name, repo_id, save_path = args
        logger.info(
            "proc %s: batching %s to %s %s"
            % (os.getpid(), save_path, repo_id, repo_save_path + shard_name)
        )
        current_batch_paths.append(save_path)
        current_batch_repo_paths.append(repo_save_path + shard_name)


def start_upload_procs(
    qq, num_proc, batch_size, delete_local=True, repo_save_path="data/"
):
    """Start the reader processes and return all in a list to the caller"""
    all_reader_procs = list()
    for _ in range(0, num_proc):
        reader_p = Process(
            target=upload_proc,
            args=(batch_size, qq, delete_local, repo_save_path),
        )
        reader_p.daemon = True
        reader_p.start()  # Launch reader_p() as another proc

        all_reader_procs.append(reader_p)

    return all_reader_procs


def upload_to_hf_hub(
    dataset_dict: DatasetDict,
    repo_id: str,
    num_proc: int = 2,
    tmp_save_dir=tempfile.gettempdir(),
    max_shard_size: Union[str, int] = "2GB",
    stop_at_shard: int = None,
    files_per_commit=10,
):
    """Pushes the dataset to the hub as a Parquet dataset.
    The dataset is processed to shards and uploaded in parallel. It useful for large datasets.
    It's recommended to use `hf_transfer`: `pip install hf_transfer` to speed up the upload.

    Args:
        dataset_dict (DatasetDict): The dataset object
        repo_id (str): The dataset repo on HF Hub to upload to
        num_proc (int, optional): The number of process to perform shards batching and upload. Defaults to 2.
        tmp_save_dir (_type_, optional): The temp folder to store the processed dataset shards locally before uploading.
            It should hold `2 * num_proc * files_per_commit` files temporarly. Defaults to tempfile.gettempdir().
        max_shard_size (Union[str, int], optional): Max shard file size . Defaults to "2GB".
        stop_at_shard (int, optional): For debugging uploading will stop after reaching this shard number. Defaults to None.
        files_per_commit (int, optional): The numbers of files to be uploaded in one commit. Defaults to 10.

    Example:

        ```python
        >>> datasets.logging.set_verbosity_info()
        >>> dataset = datasets.load_dataset("dataset_id.py")
        >>> push_utils.upload_to_hf_hub(dataset, "<organization>/<dataset_id>")
        >>> push_utils.push_dataset_card(dataset, "<organization>/<dataset_id>")
        ```
    """
    qq = Queue(maxsize=num_proc * files_per_commit)
    all_upload_procs = start_upload_procs(qq, num_proc, batch_size=files_per_commit)
    for split in dataset_dict.keys():
        logger.info(f"Uploading {split} split")
        dataset = dataset_dict[split]
        shard_name = get_shard_prefix(split) + "_{}.parquet"
        max_shard_size = convert_file_size_to_int(max_shard_size)
        num_shards = max(int(dataset.data.nbytes / max_shard_size) + 1, 1)
        logger.info(f"    Split {split} will be split into {num_shards} shards")
        logger.info(
            f"    the {num_shards} shards will be upoaded in {num_proc} processes with {files_per_commit} files per commit"
        )

        for i in range(num_shards):
            if stop_at_shard is not None:
                if i >= stop_at_shard:
                    break
            shard = dataset.shard(num_shards=num_shards, index=i, contiguous=True)
            out_path = os.path.join(tmp_save_dir, shard_name.format(i))
            if api.file_exists(
                repo_id, "data/" + shard_name.format(i), repo_type="dataset"
            ):
                logger.info(
                    f"Shard {shard_name.format(i)} already exists in repo {repo_id} , skipping"
                )
                if os.path.isfile(out_path):
                    logger.info(
                        f"local shard {shard_name.format(i)} exists locally, deleting it"
                    )
                    os.remove(out_path)
                continue

            if os.path.isfile(out_path):
                logger.info(
                    f"Shard {shard_name.format(i)} already exists locally, skipping"
                )
            else:
                tmp_file = os.path.join(tmp_save_dir, f"{os.getpid()}_tmp.parquet")
                shard.to_parquet(
                    tmp_file
                )  # save to temp file to avoid, partialy written files
                os.rename(tmp_file, out_path)
            qq.put(("upload", (shard_name.format(i), repo_id, out_path)))

    for _ in range(0, len(all_upload_procs)):
        qq.put(("DONE", None))

    for idx, a_reader_proc in enumerate(all_upload_procs):
        logger.info("    Waiting for reader_p.join() index %s" % idx)
        a_reader_proc.join()  # Wait for a_reader_proc() to finish


def push_dataset_card(dataset: DatasetDict, repo_id: str):
    """Pushes a dataset card to the repo
    If the repo already has a dataset card, its meta data will be overwritten.
    If the repo does not have a dataset card, one will be created with the metadata and the dataset info.

    Args:
        dataset (DatasetDict): The dataset object
        repo_id (str): Hugginfaced dataset repo id
    """
    api = HfApi()
    s1 = next(iter(dataset.values()))
    config_name = s1.config_name or "default"
    revision = None
    data_dir = config_name if config_name != "default" else "data"

    info_to_dump: DatasetInfo = s1.info.copy()
    info_to_dump.config_name = config_name
    info_to_dump.splits = SplitDict()
    total_uploaded_size = 0
    total_dataset_nbytes = 0

    for split in dataset.keys():
        dataset_nbytes = dataset[split]._estimate_nbytes()
        info_to_dump.splits[split] = SplitInfo(
            str(split), num_bytes=dataset_nbytes, num_examples=len(dataset[split])
        )
        total_uploaded_size += 0
        total_dataset_nbytes += dataset_nbytes
        info_to_dump.download_checksums = None
        info_to_dump.download_size = total_uploaded_size
        info_to_dump.dataset_size = total_dataset_nbytes
        info_to_dump.size_in_bytes = total_uploaded_size + total_dataset_nbytes

    metadata_config_to_dump = {
        "data_files": [
            {"split": split, "path": f"{data_dir}/{get_shard_prefix(split)}_*"}
            for split in dataset.keys()
        ],
    }

    try:
        dataset_card_path = api.hf_hub_download(
            repo_id, "README.md", repo_type="dataset", revision=revision
        )
        dataset_card = DatasetCard.load(Path(dataset_card_path))
        dataset_card_data = dataset_card.data
    except:
        logger.info(f"No dataset card found in {repo_id}, making an empty one")
        dataset_card = None
        dataset_card_data = DatasetCardData()

    DatasetInfosDict({config_name: info_to_dump}).to_dataset_card_data(
        dataset_card_data
    )
    MetadataConfigs({config_name: metadata_config_to_dump}).to_dataset_card_data(
        dataset_card_data
    )
    data_card_content = (
        f"---\n{dataset_card_data}\n---\n# {repo_id}\n{info_to_dump.description}\n "
    )
    if info_to_dump.homepage and len(info_to_dump.homepage):
        data_card_content = (
            data_card_content + f"\n## Homepage\n {info_to_dump.homepage}\n"
        )
    if info_to_dump.citation and len(info_to_dump.citation):
        data_card_content = (
            data_card_content + f"\n## Citation\n ```\n{info_to_dump.citation}\n```\n"
        )
    if info_to_dump.license and len(info_to_dump.license):
        data_card_content = (
            data_card_content + f"\n## License\n {info_to_dump.license}\n"
        )  ## ## "
    dataset_card = (
        DatasetCard(data_card_content) if dataset_card is None else dataset_card
    )
    additions = []
    additions.append(
        CommitOperationAdd(
            path_in_repo="README.md", path_or_fileobj=str(dataset_card).encode()
        )
    )
    commit_message = "Generate dataset card"

    api.create_commit(
        repo_id,
        operations=additions,
        commit_message=commit_message,
        repo_type="dataset",
        revision=revision,
    )


def upload_dataset_folder_to_hf_hub(
    local_path: str,
    repo_id: str,
    num_proc: int = 2,
    private: bool = False,
    exist_ok: bool = True,
    files_per_commit=10,
):
    """Pushes folder of a dataset to the hub.
    The files of the dataset are uploaded in parallel. It useful for large datasets.
    It's recommended to use `hf_transfer`: `pip install hf_transfer` to speed up the upload.

    Args:
        local_path (path,str): The path to the dataset
        repo_id (str): The dataset repo on HF Hub to upload to
        num_proc (int, optional): The number of process to perform shards batching and upload. Defaults to 2.
        tmp_save_dir (_type_, optional): The temp folder to store the processed dataset shards locally before uploading.
            It should hold `2 * num_proc * files_per_commit` files temporarly. Defaults to tempfile.gettempdir().
        files_per_commit (int, optional): The numbers of files to be uploaded in one commit. Defaults to 10.

    Example:

        ```python
        >>> datasets.logging.set_verbosity_info()
        >>> dataset = datasets.load_dataset("dataset_id.py")
        >>> push_utils.upload_to_hf_hub(dataset, "<organization>/<dataset_id>")
        >>> push_utils.push_dataset_card(dataset, "<organization>/<dataset_id>")
        ```
    """
    api = HfApi()
    api.create_repo(
        repo_id,
        private=private,
        exist_ok=exist_ok,
        repo_type="dataset",
    )
    qq = Queue(maxsize=num_proc * files_per_commit)
    all_upload_procs = start_upload_procs(
        qq, num_proc, batch_size=files_per_commit, delete_local=False, repo_save_path=""
    )

    from pathlib import Path

    local_path = Path(local_path)
    files_to_upload = list(x for x in local_path.rglob("*") if x.is_file())
    logger.info(
        f"Uploading  {len(files_to_upload)} files, using {num_proc} processes, {files_per_commit} files per commit"
    )
    for p in files_to_upload:
        shard_name = str(p.relative_to(local_path))
        if api.file_exists(repo_id, shard_name, repo_type="dataset"):
            logger.info(
                f"file {shard_name} already exists in repo {repo_id} , skipping"
            )
            continue
        qq.put(("upload", (str(shard_name), repo_id, str(p))))

    for _ in range(0, len(all_upload_procs)):
        qq.put(("DONE", None))

    for idx, a_reader_proc in enumerate(all_upload_procs):
        logger.info("    Waiting for reader_p.join() index %s" % idx)
        a_reader_proc.join()  # Wait for a_reader_proc() to finish
