import pathlib
script_location = pathlib.Path(__file__).parent.resolve()
pos_path = str(script_location) + "\\positive"

#### load imagenet1k

# If the dataset is gated/private, make sure you have run huggingface-cli login
print("loading imagenet...")
from datasets import load_dataset
imagenet_train = load_dataset("imagenet-1k", cache_dir="D:\.cache\huggingface\datasets", split="train")
imagenet_train = imagenet_train.select(list(range(20))) # dry run remove this

# select images with aspect ratio not greater than 4/3
print("filtering images...")
imagenet_train_filtered = imagenet_train.filter(lambda image: image["image"].size[0] / image["image"].size[1] <= 4/3 and image["image"].size[1] / image["image"].size[0] <= 4/3 and image["image"].size[0] >= 224 and image["image"].size[1] >= 224)

# create list with labels
print("creating list with labels...")
from ImagenetMapping import mapping
imagenet_id_labels = imagenet_train_filtered["label"]
imagenet_labels = [mapping[id] for id in imagenet_id_labels]

#### generate positives
print("generating positives...")
from PositivesGeneration import generate_positives
print(str(imagenet_labels))
generate_positives(imagenet_labels)

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
from datasets import interleave_datasets
dataset_no_split = interleave_datasets([postive_dataset, negative_dataset])

#### split dataset
print("train test vali split...")
train_vali = dataset_no_split.train_test_split(test_size=0.15, seed=42)
train_test_vali = train_vali["train"].train_test_split(test_size=0.17647058824, seed=42) # to get a 70, 15, 15 split 0.15 / (1-0.15) = 0.17647058823
train_test_vali["vali"] = train_vali["test"]

#### save dataset
print("saving dataset...")
train_test_vali.save_to_disk("diffusion_and_real")

