from PositivesGeneration import generate_positives
from datasets import load_dataset, load_from_disk, interleave_datasets
from ImagenetMapping import mapping

import pathlib
script_location = pathlib.Path(__file__).parent.resolve()
pos_path = str(script_location) + "\\positive\\train"

GENERATE_POSITIVES = False

#### load imagenet1k

# If the dataset is gated/private, make sure you have run huggingface-cli login
print("loading imagenet...")
# imagenet_train = load_from_disk("sub-imagenet")
imagenet_train = load_dataset("imagenet-1k", cache_dir="D:\.cache\huggingface\datasets", split="train[:4%]")

# select images with aspect ratio not greater than 4/3
print("filtering images...")
imagenet_train_filtered = imagenet_train.filter(lambda image: image["image"].size[0] / image["image"].size[1] <= 4/3 and image["image"].size[1] / image["image"].size[0] <= 4/3 and image["image"].size[0] >= 224 and image["image"].size[1] >= 224)
imagenet_train_filtered = imagenet_train_filtered.select(list(range(10000))) # dry run; try to remove this lol

# create list with labels
print("creating list with labels...")
imagenet_id_labels = imagenet_train_filtered["label"]
imagenet_labels = [mapping[id] for id in imagenet_id_labels]

#### generate positives
if GENERATE_POSITIVES:
    print("generating " + str(len(imagenet_id_labels)) + " positives...")
    generate_positives(imagenet_labels)
else:
    print("using only postives from dir")

#### create positives dataset 
print("creating positives dataset...")
postive_images_dataset = load_dataset("imagefolder", data_dir=pos_path, split="train")
postive_dataset = postive_images_dataset.add_column(name="label", column=[1]*len(postive_images_dataset))

#### create negatives dataset 
print("creating negatives dataset...")
imagenet_train_filtered = imagenet_train_filtered.remove_columns(["label"])
negative_dataset = imagenet_train_filtered.add_column(name="label", column=[0]*len(imagenet_train_filtered))

#### merge datasets
print("merging datasets...")
dataset_no_split = interleave_datasets([postive_dataset, negative_dataset])

#### split dataset
print("train test vali split...")
train_vali = dataset_no_split.train_test_split(test_size=0.05, seed=42)
train_test_vali = train_vali["train"].train_test_split(test_size=0.05263, seed=42) # to get a 90, 5, 5 split 0.05 / (1-0.05) = 0.05263
train_test_vali["vali"] = train_vali["test"]

#### save dataset
print("saving dataset...")
train_test_vali.save_to_disk("diffusion_and_real")

