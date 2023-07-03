# diffusion-detection

This model was trained to distinguish real world images (negative) from machine generated ones (postive).

## Model usage

```python
from transformers import BeitImageProcessor, BeitForImageClassification
from PIL import Image

processor = BeitImageProcessor.from_pretrained('TimKond/diffusion-detection')
model = BeitForImageClassification.from_pretrained('TimKond/diffusion-detection')

image = Image.open("2980_saltshaker.jpg")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
```

## Training and evaluation data

[BEiT-base-patch16-224-pt22k](https://huggingface.co/microsoft/beit-base-patch16-224-pt22k) was loaded as a base model for further fine tuning:

As negatives a subsample of 10.000 images from [imagenet-1k](https://huggingface.co/datasets/imagenet-1k) was used. Complementary 10.000 positive images were generated using [Realistic_Vision_V1.4](https://huggingface.co/SG161222/Realistic_Vision_V1.4). 

The labels from imagenet-1k were used as prompts for image generation. [GitHub reference](https://github.com/TimKond/diffusion-detection/blob/main/data/DatasetGeneration.py)

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0002
- train_batch_size: 32
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3
- mixed_precision_training: Native AMP

### Framework versions

- Transformers 4.29.2
- Pytorch 1.11.0+cu113
- Datasets 2.12.0
- Tokenizers 0.13.3