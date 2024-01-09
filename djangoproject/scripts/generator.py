
from django.conf import settings

from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers import StableDiffusionInpaintPipeline
from diffusers import StableDiffusionImg2ImgPipeline

import multiprocessing

mp = multiprocessing.get_context("spawn")

import torch
import gc

import uuid
import os
import shutil

from PIL import Image

"""---------------------------------------------------------------------------------------------------------------------
    STEP1
---------------------------------------------------------------------------------------------------------------------"""

def step1(MODEL_ID, prompt, negative_prompt, inference_steps, strength, cfg, image, control_image, mask_image, save_path):
    # ---------------------------------------------------------
    # Pipeline setup ------------------------------------------

    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16)

    pipe_ram = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        MODEL_ID, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None
    )

    pipe = pipe_ram.to("cuda")

    del pipe_ram
    
    gc.collect()

    pipe.enable_xformers_memory_efficient_attention()

    # ---------------------------------------------------------
    # Image generation ----------------------------------------

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=inference_steps,
        strength=strength,
        guidance_scale=cfg,

        image=image,
        control_image=control_image,
        mask_image=mask_image,
    ).images[0]

    result.save(save_path + "/out1.jpg" )

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

"""---------------------------------------------------------------------------------------------------------------------
    STEP2&3
---------------------------------------------------------------------------------------------------------------------"""

def step23(
previous_image, MODEL_ID2, prompt2, negative_prompt2, inference_steps2, strength2, cfg2, mask_image2, 
prompt3, negative_prompt3, inference_steps3, strength3, cfg3, mask_image3,
user_data_path
):
    # ---------------------------------------------------------
    # Pipeline setup ------------------------------------------

    pipe_ram = StableDiffusionInpaintPipeline.from_pretrained(MODEL_ID2, torch_dtype=torch.float16, safety_checker=None )
    pipe = pipe_ram.to("cuda")

    del pipe_ram
    gc.collect()

    # remove this command when outside of docker container
    pipe.enable_xformers_memory_efficient_attention()

    # ---------------------------------------------------------
    # Image generation ----------------------------------------

    result = pipe(
        prompt=prompt2,
        negative_prompt=negative_prompt2,
        num_inference_steps=inference_steps2,
        strength=strength2,
        guidance_scale=cfg2,

        image=previous_image,
        mask_image=mask_image2,
    ).images[0]

    # ---------------------------------------------------------
    # Image generation ----------------------------------------

    result = pipe(
        prompt=prompt3,
        negative_prompt=negative_prompt3,
        num_inference_steps=inference_steps3,
        strength=strength3,
        guidance_scale=cfg3,

        image=result,
        mask_image=mask_image3,
    ).images[0]

    result.save(user_data_path + "/out23.jpg" )

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

"""---------------------------------------------------------------------------------------------------------------------
    STEP4
---------------------------------------------------------------------------------------------------------------------"""

def step4(previous_image, MODEL_ID4, prompt4, negative_prompt4, inference_steps4, strength4, cfg4, user_data_path):
    # ---------------------------------------------------------
    # Pipeline setup ------------------------------------------

    pipe_ram = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL_ID4,  torch_dtype=torch.float16, safety_checker=None)
    pipe = pipe_ram.to("cuda")

    del pipe_ram
    gc.collect()

    # remove this command when outside of docker container
    pipe.enable_xformers_memory_efficient_attention()

    # ---------------------------------------------------------
    # Image generation ----------------------------------------

    result = pipe(
        prompt=prompt4,
        negative_prompt=negative_prompt4,
        num_inference_steps=inference_steps4,
        strength=strength4,
        guidance_scale=cfg4,

        image=previous_image,
    ).images[0]

    result.save(user_data_path + "/out4.jpg" )

    del pipe
    gc.collect()
    torch.cuda.empty_cache()


"""---------------------------------------------------------------------------------------------------------------------
    STEP5
---------------------------------------------------------------------------------------------------------------------"""

def step5(previous_image, subject_face_mask, model_path, model_trigger, user_data_path ):
    from libs.diffusers.scripts.convert_lora_safetensor_to_diffusers import convert

    base_model_path="runwayml/stable-diffusion-inpainting"
    checkpoint_path= model_path
    dump_path = user_data_path + "/model"

    pipe_ram = convert(base_model_path, checkpoint_path, "lora_unet", "lora_te", 1)
    pipe = pipe_ram.to('cuda')

    del pipe_ram
    gc.collect()

    pipe.save_pretrained(dump_path, True)

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # Image generation with merged model ----------------------

    pipe_ram = StableDiffusionInpaintPipeline.from_pretrained( user_data_path + "/model", torch_dtype=torch.float16, safety_checker=None )
    pipe = pipe_ram.to("cuda")

    del pipe_ram
    gc.collect()

    # remove this command when outside of docker container
    pipe.enable_xformers_memory_efficient_attention()

    # --------------------------------------------------------
    # INPAINTING ---------------------------------------------

    subject_trigger = model_trigger

    result = pipe(
        prompt= subject_trigger,
        negative_prompt="Bad anatomy, blurry, bad quality, deformations",
        height=1200,
        width=1200,
        num_inference_steps=25,
        strength=0.78,
        guidance_scale=11,

        cross_attention_kwargs={"scale": 1},
        image=previous_image,
        mask_image=subject_face_mask,
    ).images[0]

    result.save(user_data_path + "/out5.jpg" )

    del pipe
    gc.collect()
    torch.cuda.empty_cache()



"""---------------------------------------------------------------------------------------------------------------------
    GENERATION
---------------------------------------------------------------------------------------------------------------------"""

def generate(
    poi,
    poi_image,
    action,
    action_image,
    age,
    gender,
    other_details,
    using_lora,
    lora_model,
):

    unique_folder_name =  str(uuid.uuid4())[:8]
    user_data_path = "./tmp_data/userdata/" + unique_folder_name 
    os.makedirs(user_data_path)
    
    """---------------------------------------------------------------------------------------------------------------------
    SETUP FILES
    ---------------------------------------------------------------------------------------------------------------------"""

    background_image = Image.open(poi_image.image)
    pose_image = Image.open(action_image.pose_image)
    mask_image = Image.open(action_image.mask_image)
    mask_image_refined = Image.open(action_image.mask_image_refined)   
    subject_face_mask = Image.open(action_image.subject_face_mask)

    """---------------------------------------------------------------------------------------------------------------------
    PROMPT ELEMENTS
    ---------------------------------------------------------------------------------------------------------------------"""

    subject = age + " " + gender + ", " 
    action = action.description + " "    
    subject_details =  other_details + ", "

    camera_position = "eye level camera, "

    match action_image.shot_type:
        case "CLS":
            camera_position += "subject close shot, "
        case "MES":
            camera_position += "subject medium shot, "
        case "FUS":
            camera_position += "subject full shot, "

    environment = poi_image.description + ", detailed background, "

    quality_modifiers = "high quality, very detailed, " \
                        " best quality, perfect light, perfect shadows"

    """---------------------------------------------------------------------------------------------------------------------
    PROMPT
    ---------------------------------------------------------------------------------------------------------------------"""

    prompt = \
        "Image of a " + subject \
        + action \
        + " wearing " + subject_details \
        + "in " + environment \
        + camera_position \
        + quality_modifiers

    print(" ------------------------------ ")
    print(" USING PROMPT: " + prompt)
    print(" ------------------------------ ")

    negative_prompt = \
        "worst quality, low quality, bad anatomy, deformed, disfigured, mutation hands, mutation fingers, extra fingers, missing fingers"

    """---------------------------------------------------------------------------------------------------------------------
    PIPELINES CONFIGURATIONS
    ---------------------------------------------------------------------------------------------------------------------"""

    # ---------------------------------------------------------
    # Pipe 1 Configuration:------------------------------------

    prompt1 = "Image of a " + subject \
        + action \
        + subject_details \
        + quality_modifiers

    negative_prompt1 = "" 

    inference_steps1 = 25
    strength1 = 0.98
    cfg1 = 16

    image1 = background_image
    control_image1 = pose_image
    mask_image1 = mask_image

    MODEL_ID1 = "runwayml/stable-diffusion-inpainting"

    # ---------------------------------------------------------
    # Pipe 2 Configuration:------------------------------------

    prompt2 = prompt
    negative_prompt2 = negative_prompt

    inference_steps2 = 50
    strength2 = 0.5
    cfg2 = 16

    mask_image2 = mask_image_refined

    MODEL_ID2 = "dreamlike-art/dreamlike-photoreal-2.0"

    # ---------------------------------------------------------
    # Pipe 3 Configuration:------------------------------------

    prompt3 = prompt
    negative_prompt3 = negative_prompt

    inference_steps3 = 50
    strength3 = 0.5
    cfg3 = 16

    mask_image3 = mask_image_refined

    # MODEL_ID3 = MODEL_ID2

    # ---------------------------------------------------------
    # Pipe 4 Configuration:------------------------------------

    prompt4 = prompt
    negative_prompt4 = negative_prompt

    inference_steps4 = 50
    strength4 = 0.4
    cfg4 = 18

    MODEL_ID4 = "dreamlike-art/dreamlike-photoreal-2.0"

    """---------------------------------------------------------------------------------------------------------------------
    PIPELINE | Inpaint della posa sull'immagine di background
    ---------------------------------------------------------------------------------------------------------------------"""
    """ 
    In questo primo step si usa un modello addestrato specificatamente a fare l'inpainting; non tutti i modelli sono infatti
    addestrati a questo scopo: modelli non addestrati per fare questa particolare tecnica finiscono tendenzialmente a impattare
    maggiormente il background sul quale viene fatto l'inpaint del soggetto desiderato.
    """

    p = mp.Process(target=step1, args=( MODEL_ID1, prompt1, negative_prompt1, inference_steps1, strength1, cfg1, image1, control_image1, mask_image1, user_data_path))
    p.start()
    p.join()
    p.close()

    del p
    
    gc.collect()

    """---------------------------------------------------------------------------------------------------------------------
    PIPELINE | Inpaint per dettagliare 
    ---------------------------------------------------------------------------------------------------------------------"""
    """
    Nel primo step, avendo utilizzato un modello specifico per l'inpainting, si è dovuto scendere a compromessi con la qualità
    dell'immagine generata: se da una parte non viene toccato lo sfondo, generalmente questi modelli possono non essere
    addestrati alla generazione del concetto che vogliamo rappresentare e possono quindi generare un'immagine di pessima qualità.
    Per ovviare a questo problema il secondo step esegue un'inpaiting che va a ridefinire l'area toccata dal precedente step 
    (delinata da una mask); questo step deve tener conto dei seguenti aspetti e bilanciare di conseguenza il suo impatto:
        - Avere sufficiente libertà da poter ridefinire in modo sostanziale il soggetto precedentemente inserito
        - Avere sufficiente restrizione da non impattare troppo il background
    Il risultato di questo step dev'essere una parziale omogenizzazione della qualità dell'immagine (quella sottostante la
    maschera)
    """
    previous_image = Image.open(user_data_path + "/out1.jpg")

    p = mp.Process(target=step23, args=(
        previous_image, MODEL_ID2, prompt2, negative_prompt2, inference_steps2, strength2, cfg2, mask_image2, 
        prompt3, negative_prompt3, inference_steps3, strength3, cfg3, mask_image3,
        user_data_path
    ))

    p.start()
    p.join()
    p.close()

    del p
    
    gc.collect()
    
    """---------------------------------------------------------------------------------------------------------------------
    PIPELINE | Inpaint per dettagliare
    ---------------------------------------------------------------------------------------------------------------------"""
    """
    Esegue un passaggio analogo al precedente; infatti un singolo passaggio di definizione può non essere sufficiente per
    ottenere la qualità desiderata.
    Come detto siamo vincolati dal non voler completamente scombussolare l'immagine di background e per questo motivo lo
    step di redefinizione è ristretto a una strength bassa (si vuole mantenere l'immagine più coerente possibile a quella
    originale).
    L'idea è che più step con poca strength riescono ad aggiungere più dettagli e rimanere più coerenti all'immagine originale
    rispetto ad un singolo step con una strength maggiore.
    """

    

    """---------------------------------------------------------------------------------------------------------------------
    PIPELINE | Omogenizzazione dell'immagine attraverso Img2Img
    ---------------------------------------------------------------------------------------------------------------------"""
    """
    I precedenti step hanno aiutato a dettagliare il soggetto inpaintato nel primo step andando ad applicare migliorie in
    un'area dell'immagine delimitata da una maschera (quella che appunto contiene il soggetto dell'inapint); dopo due
    step successivi di inpaint di questo tipo può dunque iniziare a comparire una netta evidenza tra quella che è il bordo
    della maschera precedentemente ritoccata e l'esterno della maschera in cui è ancora rappresentato lo sfondo originale.
    Questo step serve proprio per ovviare a tale problema: prende in input l'intera immagine e ripropone il prompt originale:
    applica un lieve strato di noise all'immagine (poca strength) e provvede a risolverlo; in questo modo si ottiene una
    immagine complessivamente "ritoccata" allo stesso modo.
    """
    previous_image = Image.open(user_data_path + "/out23.jpg")

    p = mp.Process(target=step4, args=(previous_image, MODEL_ID4, prompt4, negative_prompt4, inference_steps4, strength4, cfg4, user_data_path))

    p.start()
    p.join()
    p.close()

    del p
    
    gc.collect()

    """---------------------------------------------------------------------------------------------------------------------
    PIPELINE | SUBJECT FACE INPAINT
    ---------------------------------------------------------------------------------------------------------------------"""
    """
    Inserimento del volto del soggetto all'interno dell'immagine; si utilizza nuovamente il modello specifico per l'inpainting
    con la speranza che non vada ad impattare troppo la parte d'immagine non inerente al volto.
    Si effettua un procedimento di inpainting per fare uso dell'immagine generata precedentemente: il modello utilizzato nei
    precedenti step ha già generato un volto casuale; questo step può fare uso di tale informazione: utilizzando un'inpaint
    NON completamente "sovrascrivente" si può fare uso del precedente posizionamento della testa generata, dell'angolo utilizzato,
    delle dimensioni utilizzate e via dicendo e semplicemente si va a trasformare i tratti nel volto del soggetto desiderato.
    La chiave per questo step è trovare il giusto valore per lo strength dell'inpaint:
        - Valori troppo alti rischiano di sovrascrivere completamente il contenuto della maschera per il volto
        - Valori troppo bassi rischiano di non sovrascrivere il precedente volto con i tratti del nuovo volto
    """

    result = Image.open(user_data_path + "/out4.jpg")

    if(using_lora):
        p = mp.Process(target=step5, args=( result, subject_face_mask, lora_model.model_path.path, lora_model.model_trigger, user_data_path,))

        p.start()
        p.join()
        p.close()

        del p
        
        gc.collect()

        result = Image.open(user_data_path + "/out5.jpg")

    unique_file_name =  unique_folder_name + ".jpg"
    result.save(settings.MEDIA_ROOT + "/outputs/" + unique_file_name)
    print("generated image: " + settings.MEDIA_ROOT + "/outputs/" + unique_file_name)    
    
    shutil.rmtree(user_data_path)
    return unique_file_name