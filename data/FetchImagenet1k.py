import pathlib
script_location = pathlib.Path(__file__).parent.resolve()

from datasets import load_dataset

# use !huggingface-cli login
# If the dataset is gated/private, make sure you have run huggingface-cli login
dataset = load_dataset("imagenet-1k")

