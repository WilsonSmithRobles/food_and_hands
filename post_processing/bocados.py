import cv2
import numpy as np
import os, sys
import argparse
from loguru import logger

def extract_fps_from_log(log_file):
    with open(log_file, 'r') as file:
        for line in file:
            if "Fps" in line:
                fps_str = line.split("Fps del vÃ­deo: ")[1].split(".")[0]
                logger.info(f"{fps_str}")
                fps = float(fps_str)
                return fps
    return None

def remove_values_not_equal(image, number):
    mask = (image == number)
    image[mask == False] = 0
    return image

def convert_non_zero_to_255(image):
    mask = (image != 0)
    image[mask] = 255
    return image

def compute_mask_width(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_left = float('inf')
    max_right = float('-inf')
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        max_left = min(max_left, x)
        max_right = max(max_right, x + w)
    
    max_width = max_right - max_left
    return max_width

def convert_values_above_5_percent_to_0(image):
    height, _ = image.shape
    threshold_height = int(0.95 * height)
    mask = np.arange(height) < threshold_height
    image_copy = image.copy()
    image_copy[mask, :] = 0
    return image_copy



def contar_bocados(masks_dir : str, log_file : str, hand_tag : int, output_log : str):
    '''Función para contar los bocados de una persona en un vídeo analizado con food_n_hands
    
    Parámetros:
        masks_dir (str): Directorio donde están las máscaras de EgoHOS
        log_file (str): Fichero de log del análisis.
        hand_tag (int): Dependiendo de la mano que queramos usar para el análisis este tag es 1 o 2.
        output_log (str): Fichero de salida de este post procesado.
        
    Devuelve:
        El número de bocados según el post procesado de esta función.
    '''
    if not os.path.exists(masks_dir):
        logger.error("Masks dir does not exist")
        return 1
    if not os.path.exists(log_file):
        logger.error("File does not exist.")
        return 1
    if not log_file.endswith(".log"):
        logger.error("File is not a log file.")
        return 1
    if hand_tag != 1 and hand_tag != 2:
        logger.error("Invalid hand tag. Must be set to 1 or 2")
        return 1
    fps = extract_fps_from_log(log_file)
    logger.info(f"FPS: {fps}")
    if fps == None:
        logger.error("Invalid fps count at the log file.")
        return 1
    msecs_frame = 2 * 1000 / fps    # · 2 porque las máscaras se guardan skipeando un frame (doblando el tiempo)


    mask_files = os.listdir(masks_dir)
    mask_pngs = [filename for filename in mask_files if filename.endswith(".png")]
    mask_pngs.sort(key=lambda x: int(x.split(".")[0]))
    

    result_logger = logger.bind(application="food_and_hands_post_process")
    result_logger.add(sink=output_log, rotation="10 MB")
    

    bocado_frames_timeout = 1000 / msecs_frame  # 1s de timeout para contar otro bocado si se dan las condiciones
    logger.info(f"bocado frames: {bocado_frames_timeout}")
    ultimo_bocado = 0
    bocado_count = 0
    for mask_filename in mask_pngs:
        frame_number = int(mask_filename.split(".")[0])
        EgoHOS_Mask = cv2.imread(os.path.join(masks_dir, mask_filename), cv2.IMREAD_GRAYSCALE)
        
        hand_mask = remove_values_not_equal(EgoHOS_Mask, hand_tag)
        hand_mask = convert_non_zero_to_255(hand_mask)

        hand_width = compute_mask_width(hand_mask)
        bottom_hand_mask = convert_values_above_5_percent_to_0(hand_mask)
        bottom_hand_width = compute_mask_width(bottom_hand_mask)

        if bottom_hand_width > 0.95 * hand_width:   # Bocado detectado
            if frame_number - ultimo_bocado > bocado_frames_timeout:    # Cuantificar solo si hace rato que se cuantifica uno.
                bocado_count += 1
                result_logger.info(f"Bocado detectado en frame {frame_number}. Segundo {frame_number * msecs_frame / 1000}")
            ultimo_bocado = frame_number

    result_logger.info(f"Cuenta de cucharadas: {bocado_count}")
    
    return 0



def main():
    parser = argparse.ArgumentParser(description="Cliente de food_n_hands. Extrae un fichero log de acuerdo a la mano del usuario respecto a los flags.")
    parser.add_argument("-i", "--input", required = True, help = "Path/a/directorio que se envía a analizar")
    parser.add_argument("-o", "--output", required = True, help = "Path/a/archivo_log que se va a escribir")
    args = parser.parse_args()
    
    dir_path = args.input
    masks_dir = os.path.join(dir_path, "EgoHOS_Masks")
    log_file = os.path.join(dir_path, "log.log")
    logger.info(f"Masks dir: {masks_dir} && log file: {log_file}")

    return contar_bocados(masks_dir, log_file, 2, args.output)


if __name__ == "__main__":
    sys.exit(main())