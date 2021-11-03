import os, sys
from multiprocessing import Pool
import skimage.io
import numpy as np
from labels import trainId2label

def createLabelImage(gt_semantic, num_classes):
    suffix = '_gtFine_color.png'
    img = skimage.io.imread(gt_semantic + suffix)

    mask_dir = gt_semantic + "_semantic"

    if os.path.exists(mask_dir):
        print("mask_dir exists")
    else:
        os.mkdir(mask_dir)

    for tid in range(1, num_classes):
        color = trainId2label[tid].color + (255,)
        sem = img == color
        sem = np.sum(sem, axis=2, keepdims=False)
        sem = sem == 4
        if np.sum(sem) != 0:
            sem = (1 - sem) * 255
            skimage.io.imsave(os.path.join(mask_dir, "{}.png".format(tid)), sem)

# call the main method
if __name__ == "__main__":
    #Maskdir = "../../data/gtFine/{}".format(sys.argv[1])
    suffix = '_gtFine_color.png'
    gt_semantics = []
    DIR = os.path.join("../../data/gtFine", sys.argv[1])
    cities = os.listdir(DIR)
    for city in cities:
        city_path = os.path.join(DIR, city)
        files = os.listdir(city_path)
        for f in files:
            if f.endswith(suffix):
                gt_semantics.append(os.path.join(city_path, f[:-len(suffix)]))

    counter = 0
    pool = Pool(processes=6)
    for gt_semantic in gt_semantics:
    #    target(jsonfile)
        _ = pool.apply_async(createLabelImage, args=(gt_semantic, int(sys.argv[2])))
        print("successfully draw masks" + str(counter + 1))
        counter += 1

    pool.close()
    pool.join()
#    print(counter)
