#!/usr/bin/python
#
# Reads labels as polygons in JSON format and converts them to label images,
# where each pixel has an ID that represents the ground truth label.
#
# Usage: json2labelImg.py [OPTIONS] <input json> <output image>
# Options:
#   -h   print a little help text
#   -t   use train IDs
#
# Can also be used by including as a module.
#
# Uses the mapping defined in 'labels.py'.
#
# See also createTrainIdLabelImgs.py to apply the mapping to all annotations in Cityscapes.
#

# python imports
import os, sys, getopt, re
from multiprocessing import Pool

# Image processing
# Check if PIL is actually Pillow as expected
#try:
#    from PIL import PILLOW_VERSION
#except:
#    print("Please install the module 'Pillow' for image processing, e.g.")
#    print("pip install pillow")
#    sys.exit(-1)

try:
    import PIL.Image     as Image
    import PIL.ImageDraw as ImageDraw
except:
    print("Failed to import the image processing packages.")
    sys.exit(-1)


# cityscapes imports
sys.path.append( os.path.normpath( os.path.join( os.path.dirname( __file__ ) , '..' , 'helpers' ) ) )
from annotation import Annotation
from labels     import name2label

# Print the information
def printHelp():
    print('{} [OPTIONS] inputJson outputImg'.format(os.path.basename(sys.argv[0])))
    print('')
    print('Reads labels as polygons in JSON format and converts them to label images,')
    print('where each pixel has an ID that represents the ground truth label.')
    print('')
    print('Options:')
    print(' -h                 Print this help')
    print(' -t                 Use the "trainIDs" instead of the regular mapping. See "labels.py" for details.')

# Print an error message and quit
def printError(message):
    print('ERROR: {}'.format(message))
    print('')
    print('USAGE:')
    printHelp()
    sys.exit(-1)

# The path that masks save to("train/val")
#Maskdir = "../../data/gtFine/{}".format(sys.argv[1])

# The json files path("train/val")
#jsondir = "../../data/Json_files/{}".format(sys.argv[1])
#jsonfiles = os.listdir(jsondir)

# Convert the given annotation to a label image

def createLabelImage(jsonfile, annotation, encoding, outline=None):
    # the size of the image
    size = (annotation.imgWidth, annotation.imgHeight)
    cnt = 0
    # the background
    if encoding == "ids":
        background = name2label['unlabeled'].id
    elif encoding == "trainIds":
        background = name2label['unlabeled'].trainId
    elif encoding == "color":
        background = name2label['unlabeled'].color
    else:
        print("Unknown encoding '{}'".format(encoding))
        return None

    # choose the classes that you want to make the masks
    mask_list = ["bicycle", "motorcycle", "train", "bus", "truck", "car", "person", "sky", "vegetation", "traffic sign", 
        "traffic light", "building", "sidewalk", "road", "ground"]

    # make a dir to save the masks
    suffix = "_gtFine_polygons.json"
    mask_dir = jsonfile[:-len(suffix)]+"_instance"
    if os.path.exists(mask_dir):
        print("mask_dir exists")
    else:
        os.mkdir(mask_dir)

    # loop over all objects
    for obj in annotation.objects:
        if obj.label in mask_list:

            # this is the image that we want to create
            if encoding == "color":
                labelImg = Image.new("RGBA", size, background)
            else:
                labelImg = Image.new("L", size, background)

            # a drawer to draw into the image
            drawer = ImageDraw.Draw(labelImg)

            label = obj.label
            polygon = obj.polygon

            # If the object is deleted, skip it
            if obj.deleted:
                continue

            # If the label is not known, but ends with a 'group' (e.g. cargroup)
            # try to remove the s and see if that works
            if (not label in name2label) and label.endswith('group'):
                label = label[:-len('group')]

            if not label in name2label:
                printError("Label '{}' not known.".format(label))

            # If the ID is negative that polygon should not be drawn
            if name2label[label].id < 0:
                continue

            if encoding == "ids":
                val = name2label[label].id
            elif encoding == "trainIds":
                val = name2label[label].trainId
            elif encoding == "color":
                val = name2label[label].color

            try:
                if outline:
                    drawer.polygon(polygon, fill=val, outline=val)
                else:
                    drawer.polygon(polygon, fill=val)
            except:
                print("Failed to draw polygon with label {}".format(label))
                raise
            # labelImg.show()

            # save photo
            # the parameter 'val' is the corresponding trainId of the mask but not the validation set
            labelImg.save(os.path.join(mask_dir, "{}_{}.png".format(val, cnt)))
            cnt = cnt + 1

# multiprocess
def target(jsonfile):
    annotation = Annotation()
    annotation.fromJsonFile(jsonfile)
    createLabelImage(jsonfile, annotation, "trainIds")

# call the main method
if __name__ == "__main__":
    # main(sys.argv[1:])
    

    #Maskdir = "../../data/gtFine/{}".format(sys.argv[1])
    suffix = '.json'
    jsonfiles = []
    DIR = os.path.join("../../data/gtFine", sys.argv[1])
    cities = os.listdir(DIR)
    for city in cities:
        city_path = os.path.join(DIR, city)
        files = os.listdir(city_path)
        for f in files:
            if f.endswith(suffix):
                jsonfiles.append(os.path.join(city_path, f))

    counter = 0
    pool = Pool(processes=6)
    for jsonfile in jsonfiles:
        #if jsonfile != "../../data/gtFine/val/frankfurt/frankfurt_000001_054219_gtFine_polygons.json":
        #    continue
        #print(jsonfile)
        _ = pool.apply_async(target, args=(jsonfile,))
        print("successfully draw masks" + str(counter + 1))
        counter += 1

    pool.close()
    pool.join()
#    print(counter)
