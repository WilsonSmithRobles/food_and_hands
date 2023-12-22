import struct
import socket
import numpy as np
import sys

def receive_image_metadata(conn):
    # Receive metadata
    metadata_size = struct.calcsize("!II10sQ")
    metadata_raw = conn.recv(metadata_size)

    # Unpack metadata
    rows, cols, encoding, data_length = struct.unpack("!II10sQ", metadata_raw)
    encoding = encoding.decode('utf-8')

    return rows, cols, encoding, data_length


def receive_image_data(conn, data_length, rows, cols, encoding):
    # Receive image data
    image_data = b''
    while len(image_data) < data_length:
        chunk = conn.recv(data_length - len(image_data))
        if not chunk:
            break
        image_data += chunk
    
    # Convert the binary data to a NumPy array
    image_array = np.frombuffer(image_data, dtype=np.uint8)

    if encoding.startswith("BGR"):      # Por la razón que sea al decodificar el utf 8 tenemos un string de longitud 10. Si usamos == daría false
        num_channels = 3
    elif encoding.startswith("Grayscale"):
        num_channels = 1
    else:
        sys.exit("Undefined format")

    # # Reshape the array based on image dimensions
    image = image_array.reshape((rows, cols, num_channels))  # Assuming 3 channels (BGR)

    return image


def send_image_metadata(client_socket : socket, image, rows, cols, encoding):

    # Get image metadata
    data_length = len(image.tobytes())

    # Pack metadata into a binary format
    metadata = struct.pack("!II10sQ", rows, cols, encoding.encode('utf-8'), data_length)

    # Send metadata
    client_socket.sendall(metadata)