from diffusers import StableDiffusionPipeline
import torch
import pathlib
script_location = pathlib.Path(__file__).parent.resolve()

model_id = "SG161222/Realistic_Vision_V1.4"
model = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker = None) 
pipe = model.to("cuda")

prompt_config = lambda x: str(x) + ", 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"

test_label_list = ["bird","fly","goose","table","Ipad","25 y.o. girl"]

def divide_list(list, n):
    for i in range(0, len(list), n):
        yield list[i:i + n]

def generate_positives(label_list):
    prompt_list = [prompt_config(label) for label in label_list]

    for i, label_sublist in enumerate(divide_list(prompt_list, 1)): # split list into chunks of 1-5 for simultaneous processing

        if i % 50 == 0: 
            print("generating image: " + str(i) + " / "+ str(len(prompt_list)))

        results = pipe(label_sublist, num_inference_steps=50, guidance_scale=7.5, height=512, width=512) # https://huggingface.co/blog/stable_diffusion # BEiT input is 224*224; but using other values then 512*512 here results in performance loss

        for j, image in enumerate(results.images):

            # convert to jpg; major change; imagenet is also jpg
            rgb_im = image.convert('RGB')
            file_name = str(i+j) + "_" + label_list[i+j]
            file_name = file_name.replace("train", "trn") # datasets reads a file with train in file name in the train directory twice :/
            rgb_im.save( str(script_location) + "\\positive\\train\\" + file_name + ".jpg", quality=75) # default pil quality is 75; quailty needs to match imagenet quality, this could f the training  

if __name__ == "__main__":
    generate_positives(test_label_list)