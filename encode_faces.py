import face_recognition
import numpy as np
import os, sys
from cv2 import cv2
from PIL import Image

KNOWN_FACES_DIR =   'C:/Users/Me/Desktop/Known_Faces/'
# UNKNOWN_FACE = 'C:/Users/Me/Desktop/Unknown_Face/ggf.png'
TOLERANCE = 0.5
MODEL = 'cnn' 

#We have a dir for every known person in the KNOWN_FACES_DIR. People are separed by folders and their names are the folder names, in caps.
name = input("Enter your name (All Caps): ") 

print('Loading known faces...')
known_faces = []
known_names = []


# for name in os.listdir(KNOWN_FACES_DIR):
for filename in os.listdir(KNOWN_FACES_DIR+name):
    print(filename)
    image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')

    # Get 128-dimension face encoding
    try:
        encoding = face_recognition.face_encodings(image)[0]
    except:
        print("No face found in image")  #Also add code to delete image

    # Append encodings and name
    known_faces.append(encoding)
    # known_names.append(name)

# print("NAME IS "+name)

print("Encoding faces")

np.savetxt("C:/Users/Me/Desktop/Encodings/UDIT/encode.txt", known_faces)
# np.savetxt('C:/Users/Me/Desktop/Encodings/names.txt', known_names)
print("Successfully stored")
    
