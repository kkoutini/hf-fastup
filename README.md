# HF-fastup

Pushes a HF dataset to the HF hub as a Parquet dataset, allowing streaming.
The dataset is processed to shards and uploaded in parallel. It useful for large datasets, for example, with embedded data.

## Usage

Make sure hf_transfer is installed and `HF_HUB_ENABLE_HF_TRANSFER` is set to `1`.

```python
import hffastup
import datasets
datasets.logging.set_verbosity_info()

# load any HF dataset
dataset = datasets.load_dataset("my_large_dataset.py")

hffastup.upload_to_hf_hub(dataset, "Org/repo") # upload to HF Hub
hffastup.push_dataset_card(dataset, "Org/repo") # Makes a dataset card and pushes it to HF Hub

```
