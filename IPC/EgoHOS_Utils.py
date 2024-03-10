
import numpy as np
from .defs import hands_categories

def colorize_egoHOS_mask(img, seg_result):
    seg_color = np.zeros(img.shape, dtype=np.uint8)
    for category in hands_categories:
        seg_color[(seg_result == category['id']).all(-1)] = category['color']
    
    return seg_color

def analyze_egoHOS_mask(EgoHOS_Mask, right_hand : bool, left_hand : bool):
    egoHOS_tags = np.unique(EgoHOS_Mask)
    egoHOS_log = ""
    if left_hand:
        egoHOS_log += "\n"
        if (np.any(egoHOS_tags == 1)):
            egoHOS_log += f'Left hand is FOUND'
        else:
            egoHOS_log += f'Left hand is NOT FOUND'
            
        egoHOS_log += "\n"
        if (np.any(egoHOS_tags == 3)):
            egoHOS_log += f'Left hand object is FOUND'
        else:
            egoHOS_log += f'Left hand object is NOT FOUND'
    
    if right_hand:
        egoHOS_log += "\n"
        if (np.any(egoHOS_tags == 2)):
            egoHOS_log += f'Right hand is FOUND'
        else:
            egoHOS_log += f'Right hand is NOT FOUND'

        egoHOS_log += "\n"
        if (np.any(egoHOS_tags == 4)):
            egoHOS_log += f'Right hand object is FOUND'
        else:
            egoHOS_log += f'Right hand object is NOT FOUND'

    return egoHOS_log, egoHOS_tags