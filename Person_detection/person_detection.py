#py person_detection.py --prototxt Deploy.prototxt.txt --model Deploy.caffemodel

#Импортируем библиотеки
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2

from config import args, CLASSES, COLORS, VIDEO

def start_person_detection():
	"""
	Определение объектов в видео потоке с помощь openCV, использован Dataset и информация с 
	https://www.pyimagesearch.com/2017/09/18/real-time-object-detection-with-deep-learning-and-opencv/?utm_source=mybridge&utm_medium=blog&utm_campaign=read_more

	"""
	# Загрузка моделей
	print("Загрузка модели...")
	net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

	# Загрузка видеопотока
	print("Старт видеопотока...")
	vs = VideoStream(VIDEO).start() # В качестве аргумента можно использовать видеопоток от камеры
	time.sleep(2.0)
	fps = FPS().start() 

	# Запуск видеопотока и разбиение на фреймы
	while True:
		# Чтение фреймов и разбиение на файлы
		frame = vs.read()
		frame = imutils.resize(frame, width=1200)

		# Преобразование фрейма в двоичный объект
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
			0.007843, (300, 300), 127.5)

		# Передача двоичного объекта
		net.setInput(blob)
		detections = net.forward()

		# Обнаружение
		for i in np.arange(0, detections.shape[2]):
			# Вероятность для отображения
			confidence = detections[0, 0, i, 2]

			# Фильтрация данных
			if confidence > args["confidence"]:
				# Определение координат объекта, прорисовка рамки
				idx = int(detections[0, 0, i, 1])
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# Отображение на рамке текста
				label = "{}: {:.2f}%".format(CLASSES[idx],
					confidence * 100)
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					COLORS[idx], 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(frame, label, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

		# Вывод фреймов
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# Выход при ключе "q"
		if key == ord("q"):
			break

		# Обновление FPS
		fps.update()

	# Прекращение работы и вывод информации
	fps.stop()
	print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

	# Закрытие окна и прекращение рабоыт
	cv2.destroyAllWindows()
	vs.stop()

if(__name__ == "__main__"):
	start_person_detection()