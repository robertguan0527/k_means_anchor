import glob
import xml.etree.ElementTree as ET
import os
import numpy as np
import csv
import cv2
import time

from kmeans import kmeans, avg_iou

ANNOTATIONS_PATH = "wider_face_train_bbx_gt.txt"
CLUSTERS = 6

# def load_dataset(path):
# 	ret =[]
# 	with open(path, 'r') as f:
# 		for txt_line in f:
# 			if '.jpg' in txt_line or '.bpm' in txt_line or len(txt_line.split(' ')) < 2:
# 				continue
# 			else:
# 				box = [txt_line.split(' ')[2],txt_line.split(' ')[3]]
# 				ret.append(box)
# 	return np.array(ret)
def wr_csv(path,data =[], mode = 'r'):
	csvfile = open(path, mode)
	if mode =='w':
		writer = csv.writer(csvfile)
		writer.writerows(data)
		csvfile.close()
	else:
		ret= []
		reader = csv.reader(csvfile)
		for item in reader:
			if len(item):
				ret.append(item)
		csvfile.close()
		return ret





def load_dataset(path):

		# root is the path of data folder's location

	images_name_list = []
	ground_truth = []
	datasets =[]
	# print(self._gt_path)
	if not os.path.exists('wider.csv'):
		with open(path, 'r') as f:
			for i in f:
				images_name_list.append(i.rstrip())
				ground_truth.append(i.rstrip())

		images_name_list = list(filter(lambda x: x.endswith('.jpg') or x.endswith('.bmp'), images_name_list))
		for i in range(len(images_name_list)):
			datasets +=get_item(i,images_name_list,r"D:\DNN learning\kmeans-anchor-boxes-master\WIDER_train\images",
			ground_truth)
			print(f"Picture{i} is loading total number of picture are {len(images_name_list)},please wait....")

			wr_csv('wider.csv', data = datasets,mode = 'w')
	else:
		datasets = wr_csv('wider.csv')

	return np.array(datasets)

def search(image_name,gt_list):
	for i, line in enumerate(gt_list):
		if image_name == line:
			return i


def get_item(index,img_list, images_folder,gt):

	image_name = img_list[index]
	image = cv2.imread(os.path.join(images_folder, image_name))
	height, width, chanels = image.shape
	# 查找文件名
	loc = search(image_name,gt)
	# 解析人脸个数
	face_nums = int(gt[loc + 1])
	# 读取矩形框
	rects = []

	for i in range(loc + 2, loc + 2 + face_nums):
		line = gt[i]
		x, y, w, h = line.split(' ')[:4]
		x, y, w, h = list(map(lambda k: int(k), [x, y, w, h]))

		rects.append([w/width,h/height])

	# 图像

	return rects







if __name__ =="__main__":
	data = load_dataset(ANNOTATIONS_PATH)
	data = np.array(list(map(lambda x : [float(x[0]),float(x[1])],data)))

	out = kmeans(data, k=CLUSTERS)
	print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
	print("Boxes:\n {}".format(out))

	ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
	print("Ratios:\n {}".format(sorted(ratios)))