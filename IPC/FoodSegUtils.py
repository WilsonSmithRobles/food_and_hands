import numpy as np
from defs import foodseg_categories

def colorize_FoodSeg_Mask(img, seg_result):
    seg_color = np.zeros(img.shape, dtype=np.uint8)
    for category in foodseg_categories:
        seg_color[(seg_result == category['id']).all(-1)] = category['color']
    
    return seg_color

def analyze_FoodSeg_mask(FoodSeg_Mask):
    foodtags = np.unique(FoodSeg_Mask)
    ingredients_log = "\n"
    for index, tag in enumerate(foodtags):
        if index == 0:
            continue
        ingredients_log += f'{foodseg_categories[tag]["tag"]} --- '

    return ingredients_log