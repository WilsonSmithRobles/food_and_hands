import cv2
import numpy as np
import os
import argparse
import sys
import socket
import struct

# Usaremos TCP aunque vayamos a usar localhost dado que a pesar de ello, UDP a veces descarta paquetes.
def receive_image_metadata(conn):
    # Receive metadata
    metadata_size = struct.calcsize("!II10sQ")
    metadata_raw = conn.recv(metadata_size)

    # Unpack metadata
    rows, cols, encoding, data_length = struct.unpack("!II10sQ", metadata_raw)
    encoding = encoding.decode('utf-8')

    return rows, cols, encoding, data_length

def receive_image_data(conn, data_length, rows, cols):
    # Receive image data
    image_data = b''
    while len(image_data) < data_length:
        chunk = conn.recv(data_length - len(image_data))
        if not chunk:
            break
        image_data += chunk
    
    # Convert the binary data to a NumPy array
    image_array = np.frombuffer(image_data, dtype=np.uint8)

    # Reshape the array based on image dimensions
    image = image_array.reshape((rows, cols, 3))  # Assuming 3 channels (BGR)

    return image


def main():
    HOST, PORT = "localhost", 50005
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)

    print(f"Server listening on {HOST}:{PORT}")

    while True:
        conn, addr = server_socket.accept()
        print(f"Connection from {addr}")

        rows, cols, encoding, data_length = receive_image_metadata(conn)

        image_received = receive_image_data(conn, data_length, rows, cols)

        # Close the connection
        conn.close()

        cv2.imshow('Image',image_received)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == "__main__":
    main()