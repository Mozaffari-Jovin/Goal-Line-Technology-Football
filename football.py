import cv2
import numpy as np
from ultralytics import YOLO

### goal line equation
lx1 = 220.
ly1 = 170.
lx2 = 400.
ly2 = 330.
m = (ly2-ly1)/(lx2-lx1)
b = -m * lx1 + ly1

### YOLO model
model = YOLO("models/yolov8x")

for i in range(1, 7):
	img = cv2.imread(f"media/football/{i}.jpg")

	results = model.predict(source=img) # read an image
	# print(results[0].boxes.xyxy, results[0].boxes.cls)
	# cv2.waitKey(0) # YOLO uses opencv to display

	for result in results:
		for (obj_xyxy, obj_cls) in zip(result.boxes.xyxy, result.boxes.cls): # two concurrent loops
			# print(int(obj_cls)) # tensor to integer
			obj_cls = int(obj_cls)
			if obj_cls == 32: # 32 is the ball's ID in YOLO
				x1 = obj_xyxy[0].item()
				y1 = obj_xyxy[1].item()
				x2 = obj_xyxy[2].item()
				y2 = obj_xyxy[3].item()
				# print(x1, y1, x2, y2)

				# ball center and radius
				ball_x = (x2+x1)/2
				ball_y = (y2+y1)/2
				ball_r = ((x2-x1)+(y2-y1))/4
				# print(ball_x, ball_y, ball_r)

				# ball position w.r.t. the goal line equation
				yhat = ball_x * m +b
				print (ball_y, yhat)

				if yhat < ball_y:
					goal = False
				else:
					distance = abs(m*ball_x-ball_y+b)/(m**2+1)**0.5
					print(distance, ball_r)
					if distance > ball_r:
						goal = True
						break


				cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
		if goal:
			break
	# cv2.imshow("Frames", img)
	# cv2.waitKey(0)
	if goal:
		break

if goal:
	print("GOAAALL!")

cv2.destroyAllWindows()