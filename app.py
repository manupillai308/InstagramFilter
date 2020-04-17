import cv2
import dlib
import numpy as np
from animation import Animate
import os

predictor_path = "./shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
cam = cv2.VideoCapture(0)

root_dir = "./patches"

patches = [os.path.join(root_dir, path) for path in os.listdir(root_dir)]


animate = Animate(patches)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
source_h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
source_w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
writer = cv2.VideoWriter('output.mp4', fourcc, 10.0, (source_w, source_h))
while(cam.isOpened()):
	ret, img = cam.read()
	if ret:
		dets = detector(img[...,[2,1,0]], 1)
		if len(dets) < 1:
			continue
		d = dets[0]
		shape = predictor(img[...,[2,1,0]], d)
		points = np.empty((4, 2), dtype=int)
		for i, ix in enumerate([18,25,5,11]):
			points[i][0], points[i][1] = shape.part(ix).x, shape.part(ix).y 
		shift = (points[:, 1].max() - points[:, 1].min())*3//2
		diag = np.linalg.norm(points[0] - points[3], 2)
		points[:, 1] = points[:, 1] - shift
		points[:2, 1] += int(diag*2//6)
		patch = animate.get(points, img.shape[::-1][1:])
		flag = np.logical_not(patch.astype("bool"))
		process_img = flag * img + patch
		cv2.imshow("Camera Window", process_img)
		writer.write(process_img)
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break
	else:
		break


cam.release()
writer.release()
cv2.destroyAllWindows()
