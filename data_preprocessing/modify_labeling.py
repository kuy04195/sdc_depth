import sys
import os
import skimage.io
import shutil
import numpy as np
from multiprocessing import Pool
import pickle, gzip

if len(sys.argv) < 3:
	print("need (test, train, val) and (num_classes)")
	exit()

image_dir = "../../data/gtFine/{}".format(sys.argv[1])
num_classes = int(sys.argv[2])

def target(city_dir, f):	
	ins_dir = os.path.join(city_dir, f)
	sem_dir = ins_dir.replace("instance", "semantic")

	instances = os.listdir(ins_dir)
	semantics = os.listdir(sem_dir)

	for tid in range(1, num_classes):
		mask_dir = os.path.join(sem_dir, "{}.png".format(tid))
		if not os.path.exists(mask_dir):
			continue

		mask = skimage.io.imread(mask_dir)
		mask = mask == 0

		image_list = [x for x in instances if x.startswith(str(tid)+"_")]
		for img in image_list:
			img_dir = os.path.join(ins_dir, img)
			img = skimage.io.imread(img_dir)

			img = img != 255
			img = np.logical_and(img, mask).astype(np.uint8)

			img[img==1] = tid
			img[img==0] = 255
			
			skimage.io.imsave(img_dir, img)

if __name__ == "__main__":
	pool = Pool(processes=6)

	cities = os.listdir(image_dir)
	for city in cities:
		city_dir = os.path.join(image_dir, city)
		files = os.listdir(city_dir)

		for f in files:
			if f.endswith("_instance") == False:
				continue
			#if f != "munster_000117_000019_instance":
			#	continue
			_ = pool.apply_async(target, args=(city_dir, f))

	pool.close()
	pool.join()