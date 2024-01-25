import sys
sys.path.append('../scripts')
from scripts.generator_background_gen_depth_map import gen_depth_map
from scripts.generator_background_gen_theme_and_weather import gen_theme_and_weather
from scripts.generator_background_gen_turn_into_night import gen_turn_into_night

import gc

import multiprocessing
mp = multiprocessing.get_context("spawn")

from PIL import Image

def transform_background(original_background, background_editing_config , save_path):
    
    theme_description = background_editing_config['theme_description']
    weather_description = background_editing_config['weather_description']
    is_weather_set = background_editing_config['is_weather_set']
    is_theme_set = background_editing_config['is_theme_set']
    is_night_time = background_editing_config['is_night_time']

    original_background.save(save_path + "/background_edited.png")

    # outputs to: save_path + "/background_depth.png"
    p = mp.Process(target=gen_depth_map, args=(original_background, save_path))
    p.start()
    p.join()
    p.close()
    del p
    gc.collect()

    background_depth = Image.open(save_path + "/background_depth.png")
    background_edited = Image.open(save_path + "/background_edited.png")

    if(is_weather_set or is_theme_set):
        str = 0.42

        if(is_theme_set):
            str = 0.65

        # outputs to: save_path + "/background_edited.png"
        p = mp.Process(target=gen_theme_and_weather, args=(background_edited, background_depth, theme_description, weather_description, str, save_path))
        p.start()
        p.join()
        p.close()
        del p
        gc.collect()

    background_edited = Image.open(save_path + "/background_edited.png")
    
    if(is_night_time):

        # outputs to: save_path + "/background_edited.png"
        p = mp.Process(target=gen_turn_into_night, args=(background_edited, save_path))
        p.start()
        p.join()
        p.close()
        del p
        gc.collect()
    
    background_edited = Image.open(save_path + "/background_edited.png")
    return background_edited



