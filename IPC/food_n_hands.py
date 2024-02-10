import socket
import client_protocols
import cv2
import argparse
import numpy as np
from defs import category_ids

def colorize_egoHOS_mask(img, seg_result, alpha = 0.4):
    seg_color = np.zeros((img.shape))
    seg_color[(seg_result == 0).all(-1)] = (0,    0,   0)     # background
    seg_color[(seg_result == 1).all(-1)] = (255,  0,   0)     # left_hand
    seg_color[(seg_result == 2).all(-1)] = (0,    0,   255)   # right_hand
    seg_color[(seg_result == 3).all(-1)] = (255,  0,   255)   # left_object1
    seg_color[(seg_result == 4).all(-1)] = (0,    255, 255)   # right_object1
    seg_color[(seg_result == 5).all(-1)] = (0,    255, 0)     # two_object1
    seg_color[(seg_result == 6).all(-1)] = (255,    255, 0)     # two_object1
    return seg_color

def colorize_FoodSeg_Mask(img, seg_result, alpha = 0.4):
    seg_color = np.zeros(img.shape, dtype=np.uint8)
    for category in category_ids:
        seg_color[(seg_result == category['id']).all(-1)] = category['color']
    
    return seg_color


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
    parser = argparse.ArgumentParser(description="Cliente de food_n_hands")
    parser.add_argument("-i", "--path_to_image", required = True, help = "Path/a/imagen que se envía a analizar")
    parser.add_argument("- ip1", "--ip_address1", default = "127.0.0.1", help = "IP en la que abrir el servidor de FoodSeg.")
    parser.add_argument("- p1", "--port1", default = "33334", help = "Puerto en el que abrir el servidor de FoodSeg.")
    parser.add_argument("- ip2", "--ip_address2", default = "127.0.0.1", help = "IP en la que abrir el servidor de EgoHOS.")
    parser.add_argument("- p2", "--port2", default = "33333", help = "Puerto en el que abrir el servidor de EgoHOS.")
    args = parser.parse_args()
    

    FoodSegHOST, FoodSegPORT = args.ip_address1, int(args.port1)
    EgoHOS_HOST, EgoHOS_PORT = args.ip_address2, int(args.port2)
    image_path = args.path_to_image

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    encoding = "BGR"

    try:
        FoodSeg_Mask = FoodNhands_Client(FoodSegHOST, FoodSegPORT, image, encoding)
        FoodSeg_Mask_color = colorize_FoodSeg_Mask(image, FoodSeg_Mask)

    except Exception as e:
        print("Error FoodSeg: " + str(e))
        return
        
    try:
        EgoHOS_Mask = FoodNhands_Client(EgoHOS_HOST, EgoHOS_PORT, image, encoding)
        EgoHOS_Mask_color = colorize_egoHOS_mask(image, EgoHOS_Mask)
    except Exception as e:
        print("Error EgoHOS: " + str(e))
        return
    

    masked_image_comb = image.copy()
    masked_image_comb[FoodSeg_Mask_color > 0] = FoodSeg_Mask_color[FoodSeg_Mask_color > 0]
    masked_image_comb[EgoHOS_Mask_color > 0] = EgoHOS_Mask_color[EgoHOS_Mask_color > 0]
    cv2.namedWindow('Masked image combo', cv2.WINDOW_NORMAL)
    cv2.imshow('Masked image combo',masked_image_comb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()