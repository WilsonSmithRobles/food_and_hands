import socket
import client_protocols
import cv2
import argparse
import numpy as np
from loguru import logger
from defs import foodseg_categories, hands_categories

def colorize_egoHOS_mask(img, seg_result):
    seg_color = np.zeros(img.shape, dtype=np.uint8)
    for category in hands_categories:
        seg_color[(seg_result == category['id']).all(-1)] = category['color']
    
    return seg_color

def analyze_egoHOS_mask(EgoHOS_Mask, right_hand : bool, left_hand : bool):
    egoHOS_log = "\n"
    if left_hand:
        if (np.any(EgoHOS_Mask == 1)):
            egoHOS_log += f'Left hand is FOUND'
        else:
            egoHOS_log += f'Left hand is NOT FOUND'
        if (np.any(EgoHOS_Mask == 3)):
            egoHOS_log += f'Left hand object is FOUND'
        else:
            egoHOS_log += f'Left hand object is NOT FOUND'
    
    egoHOS_log += "\n"
    if right_hand:
        if (np.any(EgoHOS_Mask == 2)):
            egoHOS_log += f'Right hand is FOUND'
        else:
            egoHOS_log += f'Right hand is NOT FOUND'
        if (np.any(EgoHOS_Mask == 4)):
            egoHOS_log += f'Right hand object is FOUND'
        else:
            egoHOS_log += f'Right hand object is NOT FOUND'
    

def colorize_FoodSeg_Mask(img, seg_result):
    seg_color = np.zeros(img.shape, dtype=np.uint8)
    for category in foodseg_categories:
        seg_color[(seg_result == category['id']).all(-1)] = category['color']
    
    return seg_color

def analyze_FoodSeg_mask(FoodSeg_Mask):
    foodtags = np.unique(FoodSeg_Mask)
    ingredients_log = f""
    for index, tag in enumerate(foodtags):
        if index == 0:
            continue
        ingredients_log += f'{foodseg_categories[tag]["tag"]} --- '

    return ingredients_log


def FoodNhands_Client(host : str, port : int, image, encoding):
    HOST, PORT = host, port

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))
    print(f"Connected to {HOST}:{PORT}")

    rows, cols, _ = image.shape
    client_protocols.send_image_metadata(client_socket, image, rows, cols, encoding = encoding)
    client_socket.sendall(image.tobytes())

    rows, cols, encoding, data_length = client_protocols.receive_image_metadata(client_socket)
    image_received = client_protocols.receive_image_data(client_socket, data_length, rows, cols, encoding)

    client_socket.close()
    
    return image_received


def main():
    parser = argparse.ArgumentParser(description="Cliente de food_n_hands. Extrae un fichero log de acuerdo a la mano del usuario respecto a los flags.")
    parser.add_argument("-i", "--input", required = True, help = "Path/a/vídeo que se envía a analizar")
    parser.add_argument("- ip1", "--ip_address1", default = "127.0.0.1", help = "IP en la que abrir el servidor de FoodSeg.")
    parser.add_argument("- p1", "--port1", default = "33334", help = "Puerto en el que abrir el servidor de FoodSeg.")
    parser.add_argument("- ip2", "--ip_address2", default = "127.0.0.1", help = "IP en la que abrir el servidor de EgoHOS.")
    parser.add_argument("- p2", "--port2", default = "33333", help = "Puerto en el que abrir el servidor de EgoHOS.")
    parser.add_argument("--right", help = "Flag para señalar que el usuario es diestro.")
    parser.add_argument("--left", help = "Flag para señalar que el usuario es zurdo.")
    args = parser.parse_args()

    FoodSegHOST, FoodSegPORT = args.ip_address1, int(args.port1)
    EgoHOS_HOST, EgoHOS_PORT = args.ip_address2, int(args.port2)
    video_path = args.input
    left, right = False, False
    if args.right:
        right = True
    if args.left:
        left = True

    if right or left:
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

        try:
            FoodSeg_Mask = FoodNhands_Client(FoodSegHOST, FoodSegPORT, frame, encoding)
            FoodSeg_Mask_color = colorize_FoodSeg_Mask(frame, FoodSeg_Mask)

        except Exception as e:
            result_logger.Error("FoodSeg: " + str(e))
            return
            
        try:
            EgoHOS_Mask = FoodNhands_Client(EgoHOS_HOST, EgoHOS_PORT, frame, encoding)
            EgoHOS_Mask_color = colorize_egoHOS_mask(frame, EgoHOS_Mask)
        except Exception as e:
            result_logger.Error("EgoHOS: " + str(e))
            return
        
        frame_log = f"\nIn frame {frame_number} we find:\n"
        ingredients_log = analyze_FoodSeg_mask(FoodSeg_Mask)
        egoHOS_log = analyze_egoHOS_mask(EgoHOS_Mask)

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