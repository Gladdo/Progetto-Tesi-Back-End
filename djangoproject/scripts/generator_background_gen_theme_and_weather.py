from transformers import pipeline
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import numpy as np
import torch
from diffusers.utils import load_image

def gen_theme_and_weather(background_image, background_depth, theme_description, weather_description, str , save_path):
    control_image = background_depth

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
    )

    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "dreamlike-art/dreamlike-photoreal-2.0", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
    ).to("cuda")

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.enable_xformers_memory_efficient_attention()

    pipe.enable_model_cpu_offload()

    image = pipe(

            prompt=theme_description + ", " + weather_description + ", photorealistic, best quality, high quality",
        
            num_inference_steps=24,
            strength=str,
            guidance_scale=12,

            image=background_image,
            control_image=control_image,
        
        
        ).images[0]
        
    image.save(save_path + '/background_edited.png')