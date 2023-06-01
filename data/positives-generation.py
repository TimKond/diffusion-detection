from diffusers import StableDiffusionPipeline
import torch
import pathlib
script_location = pathlib.Path(__file__).parent.resolve()

model_id = "SG161222/Realistic_Vision_V1.4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker = None) # consider float 32
pipe = pipe.to("cuda")

prompt_config = lambda x: str(x) + ", 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"

label_list = ["bird","fly","goose","table","Ipad","25 y.o. girl"]

prompt_list = [prompt_config(label) for label in label_list]

def divide_list(list, n):
    for i in range(0, len(list), n):
        yield list[i:i + n]


for i, label_sublist in enumerate(divide_list(label_list, 1)): # split list into chunks of 1-5 for simultaneous processing

    results = pipe(label_sublist, num_inference_steps=50, guidance_scale=7.5, height=512, width=512) # https://huggingface.co/blog/stable_diffusion # BEiT input is 224*224; but using other values here results in poor performance

    for j, image in enumerate(results.images):

        image.save( str(script_location) + "\\raw\\positive\\" + str(i+j) + "_" + label_list[i+j] + ".png" )  