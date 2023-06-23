import datasets
import torch
from transformers import BeitForImageClassification, BeitFeatureExtractor, BeitImageProcessor
import numpy as np

base_model_name = "microsoft/beit-base-patch16-224-pt22k"

processor = BeitImageProcessor.from_pretrained(base_model_name)

def crop_image(img):
    width, height = img.size
    target_size = min(width, height)
    left = max(0, (width - target_size ) // 2)
    right = left + target_size
    top = max(0, (height - target_size ) // 2)
    bottom = top + target_size
    return img.crop((left, top, right, bottom)).convert('RGB')

def process_example(example):
    img = crop_image(example['image'])
    inputs = processor(img, return_tensors='pt')
    inputs['labels'] = example['label']
    return inputs

def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = processor([crop_image(x) for x in example_batch['image']], return_tensors='pt')

    # Don't forget to include the labels!
    inputs['labels'] = example_batch['label']
    return inputs

ds = datasets.load_from_disk("../../data/diffusion_and_real/")

prepared_ds = ds.with_transform(transform)


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

metric = datasets.load_metric("f1")
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

labels = ["negative","positive"]

model = BeitForImageClassification.from_pretrained(
    base_model_name,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
).to("cuda")

from transformers import TrainingArguments

training_args = TrainingArguments(
  output_dir="./BEiT-diff-detect",
  per_device_train_batch_size=16,
  evaluation_strategy="steps",
  num_train_epochs=1,
  fp16=True,
  save_steps=5000,
  eval_steps=1000,
  logging_steps=1125,
  learning_rate=2e-4,
  save_total_limit=2,
  remove_unused_columns=False,
  push_to_hub=False,
  report_to='tensorboard',
  load_best_model_at_end=True,
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["vali"],
    tokenizer=processor,
)

if __name__ == "__main__":
    # start the training
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    # eval
    metrics = trainer.evaluate(prepared_ds['test'])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)