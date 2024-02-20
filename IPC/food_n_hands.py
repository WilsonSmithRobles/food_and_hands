import cv2
import argparse
import numpy as np
from loguru import logger

from FoodSegUtils import analyze_FoodSeg_mask, colorize_FoodSeg_Mask
from EgoHOS_Utils import analyze_egoHOS_mask, colorize_egoHOS_mask
from client_protocols import FoodNhands_Client, ThreadWithReturnValue



def main():
    parser = argparse.ArgumentParser(description="Cliente de food_n_hands. Extrae un fichero log de acuerdo a la mano del usuario respecto a los flags.")
    parser.add_argument("-i", "--input", required = True, help = "Path/a/vídeo que se envía a analizar")
    parser.add_argument("- ip1", "--ip_address1", default = "127.0.0.1", help = "IP en la que abrir el servidor de FoodSeg.")
    parser.add_argument("- p1", "--port1", default = "33334", help = "Puerto en el que abrir el servidor de FoodSeg.")
    parser.add_argument("- ip2", "--ip_address2", default = "127.0.0.1", help = "IP en la que abrir el servidor de EgoHOS.")
    parser.add_argument("- p2", "--port2", default = "33333", help = "Puerto en el que abrir el servidor de EgoHOS.")
    parser.add_argument("--right", action="store_true", help = "Flag para señalar que el usuario es diestro.")
    parser.add_argument("--left", action="store_true", help = "Flag para señalar que el usuario es zurdo.")
    args = parser.parse_args()

    FoodSegHOST, FoodSegPORT = args.ip_address1, int(args.port1)
    EgoHOS_HOST, EgoHOS_PORT = args.ip_address2, int(args.port2)
    video_path = args.input
    left, right = False, False
    if args.right:
        right = True
    if args.left:
        left = True

    if not (right or left):
        logger.error("Por favor, establezca si el usuario es al menos diestro o zurdo.")
        return

    result_logger = logger.bind(log_file="log.log")
    result_logger.add("log.log", rotation="10 MB")

    cap = cv2.VideoCapture(video_path)
    encoding = "BGR"

    if not cap.isOpened():
        result_logger.Error("Unable to open video file.")
        return

    frame_number = 1
    ret, frame = cap.read()
    if not ret:
        return
    
    # Get the frame size
    frame_height, frame_width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter('D:\\wrobles\\resultados\\output_video.avi', fourcc, fps, (frame_width, frame_height))
    result_logger.info(f"\nVideo path: {video_path}. Fps del vídeo: {fps}. Shape: ({frame_width}, {frame_height})")

    while True:
        frame_number += 1
        ret, frame = cap.read()
        if not ret:
            break

        if not frame_number % 2 == 0:
            continue

        # try:
        #     FoodSeg_Mask = FoodNhands_Client(FoodSegHOST, FoodSegPORT, frame, encoding)
        #     FoodSeg_Mask_color = colorize_FoodSeg_Mask(frame, FoodSeg_Mask)

        # except Exception as e:
        #     result_logger.Error("FoodSeg: " + str(e))
        #     return
            
        # try:
        #     EgoHOS_Mask = FoodNhands_Client(EgoHOS_HOST, EgoHOS_PORT, frame, encoding)
        #     EgoHOS_Mask_color = colorize_egoHOS_mask(frame, EgoHOS_Mask)
        # except Exception as e:
        #     result_logger.Error("EgoHOS: " + str(e))
        #     return

        FoodSeg_Thread = ThreadWithReturnValue(target=FoodNhands_Client, args=(FoodSegHOST, FoodSegPORT, frame, encoding,))
        EgoHOS_Thread = ThreadWithReturnValue(target=FoodNhands_Client, args=(EgoHOS_HOST, EgoHOS_PORT, frame, encoding,))

        EgoHOS_Thread.start()
        FoodSeg_Thread.start()

        FoodSeg_Mask = FoodSeg_Thread.join()
        EgoHOS_Mask = EgoHOS_Thread.join()

        if FoodSeg_Mask is None or EgoHOS_Mask is None:
            logger.error("Error at masks retrieval.")
            return

        try:
            FoodSeg_Mask_color = colorize_FoodSeg_Mask(frame, FoodSeg_Mask)
            EgoHOS_Mask_color = colorize_egoHOS_mask(frame, EgoHOS_Mask)

        except Exception as e:
            logger.error(f"Error at colirizing masks: {str(e)}")
            return
        
        frame_log = f"\nIn frame {frame_number} we find:"
        ingredients_log = analyze_FoodSeg_mask(FoodSeg_Mask)
        egoHOS_log = analyze_egoHOS_mask(EgoHOS_Mask, right, left)

        frame_log += ingredients_log + egoHOS_log
        result_logger.info(frame_log)

        masked_image_comb = frame.copy()
        masked_image_comb[FoodSeg_Mask_color > 0] = FoodSeg_Mask_color[FoodSeg_Mask_color > 0]
        masked_image_comb[EgoHOS_Mask_color > 0] = EgoHOS_Mask_color[EgoHOS_Mask_color > 0]

        try:
            out.write(masked_image_comb)
        
        except Exception as e:
            result_logger.error(f'{str(e)}')
            return



if __name__ == "__main__":
    main()