import face_recognition
import os, sys
from cv2 import cv2
from PIL import Image

KNOWN_FACES_DIR =   'C:/Users/Me/Desktop/Known_Faces/'
UNKNOWN_FACE = 'C:/Users/Me/Desktop/Unknown_Face/ggf.png'
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
    known_names.append(name)


print('Processing unknown face...')

image = face_recognition.load_image_file(UNKNOWN_FACE)
locations = face_recognition.face_locations(image, model=MODEL)
encodings = face_recognition.face_encodings(image, locations)

# First we need to convert it from RGB to BGR as we are going to work with cv2
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Assumption- May have multiple people in the image
print(f', found {len(encodings)} face(s)')
for face_encoding, face_location in zip(encodings, locations):

    results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)

    match = None
    if True in results:  # If a recognised face is found
        match = known_names[results.index(True)]
        print('Verification complete')
        print('Welcome '+ match)

    else:
        print("No match found. Unauthorized")

cv2.waitKey(0)
cv2.destroyWindow(filename)