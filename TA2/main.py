import cv2  
import numpy as np
# Import the load_model function to load the downloaded model
from keras.models import load_model  

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

camera = cv2.VideoCapture(0)

while True:
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    cv2.imshow("Webcam Image", image)
 
    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Print original image
    print(image)
    cv2.waitKey(0)

    # Normalize the image array
    image = (image / 127.5) -1
    
    # Print normalized image
    print(image)
    cv2.waitKey(0)

    keyboard_input = cv2.waitKey(1)
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
