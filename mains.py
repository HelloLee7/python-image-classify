from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

# 출력 형식을 지수 형식으로 하지 않음
np.set_printoptions(suppress=True)

# 모델 읽어들이기
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

camera = cv2.VideoCapture(0)

while True:
    # 프레임 읽기
    ret, image = camera.read()

    # 예측을 위한 이미지 축소
    frame = image[:, :, :] 
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)


    # 예측 가능한 형태로 변형
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # 효과적인 예측을 위한 정규화
    image = (image / 127.5) - 1

    # 예측하기
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # 확률 출력하기 
    # print("Class:", class_name[2:], end="")
    # print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
    cv2.putText(frame, f"{class_name[2:-1]} : {str(np.round(confidence_score * 100))[:-2]}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2) 
    
    
    # 이미지 보기
    cv2.imshow("Webcam Image", frame)

    # 키 입력 기다리기
    keyboard_input = cv2.waitKey(1)

    # 27은 ESC를 의미함
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()