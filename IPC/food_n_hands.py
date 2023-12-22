import socket
import client_protocols
import cv2
import argparse
import numpy as np

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

    mask_image = image_received.copy()

    mask_image[mask_image != 0] = 255
    
    return mask_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cliente de food_n_hands")
    parser.add_argument("-i", "--path_to_image", required = True, description = "Path/a/imagen que se envía a analizar")
    parser.add_argument("- ip1", "--ip_address1", default = "127.0.0.1", description = "IP en la que abrir el servidor de FoodSeg.")
    parser.add_argument("- p1", "--port1", default = "33333", description = "Puerto en el que abrir el servidor de FoodSeg.")
    parser.add_argument("- ip2", "--ip_address2", default = "127.0.0.1", description = "IP en la que abrir el servidor de EgoHOS.")
    parser.add_argument("- p2", "--port2", default = "33334", description = "Puerto en el que abrir el servidor de EgoHOS.")
    args = parser.parse_args()


    FoodSegHOST, FoodSegPORT = args.ip_address1, int(args.port1)
    EgoHOS_HOST, EgoHOS_PORT = args.ip_address2, int(args.port2)
    image_path = args.path_to_image

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    encoding = "BGR"

    FoodSeg_Mask = FoodNhands_Client(image, FoodSegHOST, FoodSegPORT, encoding)

    EgoHOS_Mask = FoodNhands_Client(image, EgoHOS_HOST, EgoHOS_PORT, encoding)
    
    # Create a red channel with value 255 where mask1 has value 255
    red_channel = np.zeros_like(image)
    red_channel[FoodSeg_Mask == 255] = [0, 0, 255]

    # Create a blue channel with the average value of blue and original value where mask2 has value 255
    blue_channel = np.zeros_like(image)
    blue_channel[EgoHOS_Mask == 255] = [255, 0, 0]

    # Take the average of corresponding pixel values from image1 and image2 where mask is True
    result_image = np.zeros_like(image, dtype=np.uint8)
    result_image[FoodSeg_Mask] = (image[FoodSeg_Mask] + red_channel[FoodSeg_Mask]) // 2
    result_image[EgoHOS_Mask] = (image[EgoHOS_Mask] + blue_channel[EgoHOS_Mask]) // 2

    cv2.imshow('Image',result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()