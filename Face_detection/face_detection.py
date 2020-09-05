#импортирование библиотек
import cv2
from config import CASCADE


def face_detection():
    faceCascade = cv2.CascadeClassifier(CASCADE)

    cap = cv2.VideoCapture(0)

    while (True):
        success, img = cap.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Определение лица
        faces = faceCascade.detectMultiScale(
            gray,     
            scaleFactor=1.1,
            minNeighbors=5,     
            minSize=(10, 10)
        )

        # Прорисовка рамки
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

        cv2.imshow('video',img)

        k = cv2.waitKey(30) & 0xff
        if k == 27: # press 'ESC' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    face_detection()