from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("C:\\Users\\201-23\\Downloads\\opencv-install\\python-image-classify\\keras_model.h5", compile=False)
class_names = open("C:\\Users\\201-23\\Downloads\\opencv-install\\python-image-classify\\labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the model's input shape.
    image_input = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image_input = (image_input / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image_input)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()  # 클래스 이름에서 공백 제거
    confidence_score = prediction[0][index]

  
    left, top = 50, 50   # 적절한 위치로 조정 가능
 
    label = f"{class_name} ({np.round(confidence_score * 1, 2)}%)"
    cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Print prediction and confidence score
    print("Class:", class_name, end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
