import os
from dotenv import load_dotenv
import cv2
import numpy as np

load_dotenv()

HEADER_SIZE = 64 #64 bytes, should be good enough for our data size
FORMAT = "utf-8" #For decoding strings

INITIAL_PREDICTION_MSG = "!INIT_PRED_MSG" #Message sent by car to request an initial prediction
CLOSE_CONNECTION_MSG = "!CLOSE" #Message sent by car to notify of connection closing

#===== FUTURE IMPLEMENTATION =====
#OBSTALCE_MSG = "!OBSTACLE" #When something is detected in the direction of intended travel,
#                            a different message will be sent

PORT = int(os.environ.get("PORT")) 
SERVER_IP = os.environ.get("SERVER_IP") #IP of host computer or server
SERVER_ADDRESS = (SERVER_IP, PORT)

def send_bytes(conn, message):
    msg_len = str(len(message)).encode(FORMAT)
    msg_len += b" " * (HEADER_SIZE - len(msg_len)) #Padding to ensure constant length for header

    conn.send(msg_len)
    conn.sendall(message)

def send_string(conn, msg):
    message = msg.encode(FORMAT)
    send_bytes(conn, message)

def send_image(conn, image, format_ext=".jpg"):
    success, img_encoded = cv2.imencode(format_ext, image)

    if success:
        byte_data = img_encoded.tobytes() #np.uint8 -> bytes
        send_bytes(conn, byte_data)
    else:
        print("Image could not be encoded")


def recieve_bytes(conn):
    msg_len = conn.recv(HEADER_SIZE).decode(FORMAT)
    while not msg_len:
        msg_len = conn.recv(HEADER_SIZE).decode(FORMAT)
        
    msg_len = int(msg_len)

    temp_msg_len = msg_len

    chunks = []
    while True:
        chunk = conn.recv(temp_msg_len) # Loop in case that full message is not recieved first time
        if not chunk:
            break

        chunks.append(chunk)
        temp_msg_len -= len(chunk)

    msg = b"".join(chunks) 

    return msg

def recieve_string(conn):
    msg = recieve_bytes(conn)
    return msg.decode(FORMAT)

def recieve_image(conn):
    img_bytes = recieve_bytes(conn)
    bytes_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(bytes_arr, cv2.IMREAD_ANYCOLOR)
    return img