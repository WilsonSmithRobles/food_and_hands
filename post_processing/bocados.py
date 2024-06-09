import torch
torch.device('cpu')
from fastai.vision.all import load_learner, Path, Image
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

def normalize_image(image : np.array):
    numpy_image = np.array(image)
    mask = ((numpy_image != 1)
            & (numpy_image != 2)
            & (numpy_image != 3)
            & (numpy_image != 4)
            & (numpy_image != 5))
    numpy_image[mask] = 0
    numpy_image[numpy_image == 1] = 255
    numpy_image[numpy_image == 2] = 255
    numpy_image[numpy_image == 3] = 128
    numpy_image[numpy_image == 4] = 128
    numpy_image[numpy_image == 5] = 128
    pil_image = Image.fromarray(numpy_image.astype(np.uint8))
    return numpy_image, pil_image

def img_tfms_inf(image_path : Path):
    image = Image.open(image_path)
    numpy_image, pil_image = normalize_image(image)
    return numpy_image, pil_image.resize((256, 256), Image.NEAREST)



def contar_bocados(masks_dir : str, log_file : str, output_log : str, model):
    '''Función para contar los bocados de una persona en un vídeo analizado con food_n_hands
    
    Parámetros:
        masks_dir (str): Directorio donde están las máscaras de EgoHOS
        log_file (str): Fichero de log del análisis.
        output_log (str): Fichero de salida de este post procesado.
        model (fastai model): Modelo de fastai cargado para inferencia mediante learn_load.
        
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
    # ultimo_bocado = 0
    bocado_count = 0
    for mask_filename in mask_pngs:
        frame_number = int(mask_filename.split(".")[0])
        _, pil_image2analyze = img_tfms_inf(Path(os.path.join(masks_dir, mask_filename)))
        predictions = model.predict(pil_image2analyze)

        if (predictions[0] == "bite"):
            result_logger.info(f"Bocado detectado en frame {frame_number}. Segundo {frame_number * msecs_frame / 1000}")
            bocado_count += 1

        # if bottom_hand_width > 0.95 * hand_width:   # Bocado detectado
        #     if frame_number - ultimo_bocado > bocado_frames_timeout:    # Cuantificar solo si hace rato que se cuantifica uno.
        #         bocado_count += 1
        #         result_logger.info(f"Bocado detectado en frame {frame_number}. Segundo {frame_number * msecs_frame / 1000}")
        #     ultimo_bocado = frame_number

    result_logger.info(f"Cuenta de cucharadas: {bocado_count}")
    
    return 0



def main():
    parser = argparse.ArgumentParser(description="Utilidad para el cliente de food_n_hands. Extrae un fichero log de acuerdo a la mano del usuario respecto a los flags.")
    parser.add_argument("-i", "--input", required = True, help = "Path/a/directorio que se envía a analizar")
    parser.add_argument("-m", "--model_path", required = True, help = "Path/a/modelo fastai que se usa para analizar")
    parser.add_argument("-o", "--output", required = True, help = "Path/a/archivo_log que se va a escribir")
    args = parser.parse_args()
    
    dir_path = args.input
    output = args.output
    model_path = args.model_path
    model = load_learner(Path(model_path), cpu=True)
    masks_dir = os.path.join(dir_path, "EgoHOS_Masks")
    log_file = os.path.join(dir_path, "log.log")
    logger.info(f"Masks dir: {masks_dir} && log file: {log_file}")

    return contar_bocados(masks_dir, log_file, output, model=model)


if __name__ == "__main__":
    sys.exit(main())