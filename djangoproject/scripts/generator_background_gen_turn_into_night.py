from PIL import Image
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.utils import load_image

def gen_turn_into_night(background_image, save_path):

    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", safety_checker=None, torch_dtype=torch.float16).to("cuda")

    prompt = "made into a night environment" 
    num_inference_steps = 20
    image_guidance_scale = 1.5
    guidance_scale = 10

    edited_image = pipeline(
    prompt=prompt,
    image=background_image,
    num_inference_steps=num_inference_steps,
    image_guidance_scale=image_guidance_scale,
    guidance_scale=guidance_scale,
    
    ).images[0]

    edited_image.save(save_path + "/background_edited.png")