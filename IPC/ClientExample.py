import socket
import cv2
import struct

def send_image_metadata(client_socket, image):
    # Get image metadata
    rows, cols, _ = image.shape
    encoding = "BGR"  # You can modify this based on your needs
    data_length = len(image.tobytes())

    # Pack metadata into a binary format
    metadata = struct.pack("!II10sQ", rows, cols, encoding.encode('utf-8'), data_length)

    # Send metadata
    client_socket.sendall(metadata)


def client_connect():
    HOST, PORT = "localHOST", 50005

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))

    print(f"Connceted to {HOST}:{PORT}")

    # Read the image
    image = cv2.imread("example.jpg")  # Replace with your image file

    # Send metadata
    send_image_metadata(client_socket, image)

    # Send image data
    client_socket.sendall(image.tobytes())

    client_socket.close()

    # cv2.imshow('Image',image_received)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    client_connect()
