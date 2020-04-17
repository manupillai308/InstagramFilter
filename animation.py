import cv2
import numpy as np

class Animate:
	def __init__(self, patch_folder, frame_size = (640, 480), thresh = 100):
		self.patch_folder = patch_folder
		self.frame_size = frame_size
		self.patch_arr = []
		for patch in self.patch_folder:
			self.patch_arr.append(cv2.resize(cv2.imread(patch), self.frame_size))
		np.random.shuffle(self.patch_arr)
		self.cur_point = 0
		self.call = 0
		self.break_ = 1
		self.thresh = thresh	
		self.src = np.float32([[0,0], [640, 0], [0, 480], [640, 480]])
	
	def get(self, dst, img_shape = None):
		if img_shape is None:
			img_shape = self.frame_size
		if self.break_ <= 75:
			self.call+=1
			if self.call % self.break_ == 0:
				self.cur_point = (self.cur_point + 1) % len(self.patch_arr)
			if self.call % self.thresh == 0:
				self.break_ +=4
			
		M = cv2.getPerspectiveTransform(self.src, dst.astype("float32"))
		return cv2.warpPerspective(self.patch_arr[self.cur_point], M, img_shape)
	
	def reset(self):
		np.random.shuffle(self.patch_arr)
		self.cur_point = 0
		self.call = 0
		self.break_ = 1




if __name__ == "__main__":
	animate = Animate(patches)

	dst = np.float32([[281, 224-100],[366, 222-100], [289, 316-100],[371, 320-100]])  
	while True:
		cv2.imshow("window", animate.get(dst))
		if cv2.waitKey(1) & 0xFF == ord("q"):
				break
