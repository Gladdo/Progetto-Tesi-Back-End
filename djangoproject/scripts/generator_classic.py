
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers import StableDiffusionInpaintPipeline
from diffusers import StableDiffusionImg2ImgPipeline

import torch
import gc

from PIL import Image

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
    maggiormente il background sul quale viene fatto l'inpaint del soggetto desiderato e dunque non sono adatti.
    Infatti dato che in questo step vogliamo generare da zero un determinato soggetto, la strenght della generazione è alta e
    si da molta libertà al generatore di cambiare l'immagine; se un modello non è addestrato per l'inpainting coglierà questa
    occasione per modificare l'intera area contenuta nella maschera (background incluso)
    """

    # ---------------------------------------------------------
    # Pipeline setup ------------------------------------------

    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16)

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        MODEL_ID1, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None
    )

    pipe.to("cuda")

    pipe.enable_xformers_memory_efficient_attention()

    # ---------------------------------------------------------
    # Image generation ----------------------------------------

    result = pipe(
        prompt=prompt1,
        negative_prompt=negative_prompt1,
        num_inference_steps=inference_steps1,
        strength=strength1,
        guidance_scale=cfg1,

        image=image1,
        control_image=control_image1,
        mask_image=mask_image1,
    ).images[0]

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    """---------------------------------------------------------------------------------------------------------------------
    PIPELINE | Inpaint per dettagliare 
    ---------------------------------------------------------------------------------------------------------------------"""
    """
    Nel primo step, avendo utilizzato un modello specifico per l'inpainting, si è dovuto scendere a compromessi con la qualità
    dell'immagine generata: se da una parte non viene toccato lo sfondo, generalmente questi modelli possono non essere
    addestrati alla generazione del concetto che vogliamo rappresentare e possono quindi generare un'immagine di pessima qualità.

    Adesso dunque dobbiamo migliorare la qualità di quanto generato nel precedente step; per fare ciò si utilizza un'altro step di
    inpainting, con la stessa mask precedente, ma utilizzando un modello addestrato appositamente per la generazione di immagini 
    fotorealistiche.

    Tale modello può non essere tuttavia addestrato a fare l'inpainting; per tale motivo dobbiamo vincolare la strength a bassi
    valori, altrimenti cè il rischio di impattare in modo sostanziale lo sfondo.
    Questo step deve tener conto dei seguenti aspetti e bilanciare di conseguenza il suo impatto:
        - Avere sufficiente libertà da poter ridefinire in modo sostanziale il soggetto precedentemente inserito
        - Avere sufficiente restrizione da non impattare troppo il background
    
    Il risultato di questo step dev'essere una parziale omogenizzazione della qualità dell'immagine (quella sottostante la
    maschera)
    """
    # ---------------------------------------------------------
    # Pipeline setup ------------------------------------------

    pipe = StableDiffusionInpaintPipeline.from_pretrained(MODEL_ID2, torch_dtype=torch.float16, safety_checker=None )
    pipe.to("cuda")

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

        image=result,
        mask_image=mask_image2,
    ).images[0]

    """---------------------------------------------------------------------------------------------------------------------
    PIPELINE | Inpaint per dettagliare
    ---------------------------------------------------------------------------------------------------------------------"""
    """
    Esegue un passaggio analogo al precedente; infatti, data la bassa strenght utilizzata, un singolo passaggio di definizione può 
    non essere sufficiente per ottenere la qualità desiderata.
    
    Un'opzione poteva essere aumentare la strenght del passo precedente; tuttavia si vuole mantenere l'immagine più coerente possibile 
    a quella originale.

    L'idea è dunque di utilizzare più step ciascuno con una strength molto bassa: questi riescono ad aggiungere dettagli senza 
    scombussolare l'immagine originale (a differenza dell'utilizzo di un singolo step con una strength maggiore).
    """

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

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    """---------------------------------------------------------------------------------------------------------------------
    PIPELINE | Omogenizzazione dell'immagine attraverso Img2Img
    ---------------------------------------------------------------------------------------------------------------------"""
    """
    I precedenti step hanno aiutato a dettagliare il soggetto generato del primo step andando ad applicare migliorie in
    un'area dell'immagine delimitata da una maschera; nonostante l'utilizzo di una strength bassa, dopo due step successivi di 
    inpaint può iniziare a comparire una netta separazione tra quella che è l'area dell'immagine sotto la precedente maschera e l'esterno 
    della stessa ( in cui è ancora conservato senza ritocchi lo sfondo originale).
    Questo step serve proprio per ovviare a tale problema: si da in input l'intera immagine e il prompt originale ad uno step di
    generazione Image to Image; impostando nuovamente uno strenght basso si va ad omogenizzare la qualità dell'immagine senza
    impattarne il contenuto.
    """

    # ---------------------------------------------------------
    # Pipeline setup ------------------------------------------

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL_ID4,  torch_dtype=torch.float16, safety_checker=None)
    pipe.to("cuda")

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

        image=result,
    ).images[0]

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    """---------------------------------------------------------------------------------------------------------------------
    PIPELINE | SUBJECT FACE INPAINT
    ---------------------------------------------------------------------------------------------------------------------"""
    """
    A questo punto i precedenti step hanno già generato il soggetto nell'ambiente desiderato, con la posa desiderata e una
    qualiutà soddisfacente; bisogna procedere ad inserire il volto dell'utente.
    Per fare ciò si fa nuovamente uso dell'inpaiting utilizzando ancora un modello specificatamente addestrato a tale scopo.
    La maschera di questo inpaiting è necessariamente attorno al volto del soggetto, con una grandezza sufficente a ricoprire almeno
    il doppio dell'estensione della testa (così da lasciare margine al generatore di lavorare eventuali accessori per la testa o
    semplicemente i capelli).
    Per far si di rendere coerente il nuovo volto con la precedente immagine, si vuole fare in modo che questo step non vada a 
    sostiture completamente il contenuto della maschera, bensì tenga di conto la testa iniziale e faccia uso del suo posizionamento,
    dell'angolo utilizzato e delle sue dimensioni; in questo modo l'inpainting va solamente a trasformare i tratti nel volto 
    del soggetto, ne cambia la fisionomia, ma presupponibilmente lascia invariata la posizione della testa.
    La chiave per questo step è dunque di trovare il giusto valore per lo strength dell'inpaint:
        - Valori troppo alti rischiano di sovrascrivere completamente il contenuto della maschera per il volto
        - Valori troppo bassi rischiano di non sovrascrivere il precedente volto con i tratti del nuovo volto

    Un'altro importante aspetto è lasciare al generatore sufficienti pixel su cui inserire il volto: il generatore può essere
    addestrato su volti appresi in alta risoluzione; se adesso il volto dev'essere generato in un insieme di pixel inferiore,
    il generatore può non essere capace di sintetizzare le giuste features in tale numero contenuto di pixels.
    Questo aspetto può essere in aprte arginato andando a produrre LoRA addestrato su diverse risoluzioni del volto; oppure, come
    proposto qui, è necessario aumentare la risoluzione dell'immagine prodotta in output.
    """

    from libs.diffusers.scripts.convert_lora_safetensor_to_diffusers import convert
    import uuid
    import os
    import shutil

    if(using_lora):
        base_model_path="runwayml/stable-diffusion-inpainting"
        checkpoint_path= lora_model.model_path.path
        unique_folder_name =  str(uuid.uuid4())[:8]
        tmp_model_path = "./tmp_data/lora_merged_models/" + unique_folder_name 
        os.makedirs(tmp_model_path)
        dump_path = tmp_model_path + "/model"

        pipe = convert(base_model_path, checkpoint_path, "lora_unet", "lora_te", 1).to('cuda')
        #pipe.to("cuda")
        pipe.save_pretrained(dump_path, True)
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

        # ---------------------------------------------------------
        # Image generation with merged model ----------------------
    
        pipe = StableDiffusionInpaintPipeline.from_pretrained( tmp_model_path + "/model", torch_dtype=torch.float16, safety_checker=None )
        pipe.to("cuda")

        # remove this command when outside of docker container
        pipe.enable_xformers_memory_efficient_attention()

        # --------------------------------------------------------
        # INPAINTING ---------------------------------------------

        subject_trigger = lora_model.model_trigger

        result = pipe(
            prompt= subject_trigger,
            negative_prompt="Bad anatomy, blurry, bad quality, deformations",
            height=1200,
            width=1200,
            num_inference_steps=25,
            strength=0.78,
            guidance_scale=11,

            cross_attention_kwargs={"scale": 1},
            image=result,
            mask_image=subject_face_mask,
        ).images[0]

        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        shutil.rmtree(tmp_model_path)
    
    return result