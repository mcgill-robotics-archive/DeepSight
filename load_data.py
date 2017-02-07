import numpy as np
from PIL import Image
from ast import literal_eval
import sys, glob, cv2
from collections import OrderedDict


def load(path_to_data):
    # takes a string pointing to a location that contains an img folder with raw images,
    # and a label folder that contains the labels and bounding boxes as text files for each frame
    files = glob.glob(path_to_data + '/img/*')[:1]
    images_w_labels = []

    for f in files:
        root = f[-9:-4]
        img = load_image(f, False)
        label = contains_buoy(load_label(path_to_data + '/label/' + root + '.txt'))
        images_w_labels.append([img, label])
    return images_w_labels


def contains_buoy(label):
    buoy = False
    bbox = [0, 0, 0, 0]  # x,y,width,height

    for l in label:
        if l[1] == 1:
            buoy = True
            break
    if buoy:
        bbox = label[0][0]
        label = [0, 1]
    else:
        label = [1, 0]

    return OrderedDict(buoy=np.array(label).reshape((1, 2)), bbox=np.array([bbox]))


def load_label(f):
    with open(f, 'rb') as file_obj:
        label = []
        for line in file_obj.readlines():
            label.append(literal_eval(line))

        return label


# @profile
def load_image(f, flat=False, net_type="VGG"):
    img = cv2.imread(f)

    # img = Image.open(f)
    # img.load()
    data = np.asarray(img, dtype="float32")

    if flat:
        img_size = data.shape[0] * data.shape[1] * data.shape[2]
        data = img.reshape((1, img_size))
    else:
        # Add the batch dimension to the image
        if net_type is "VGG":
            data = data.reshape([1] + list(data.shape))
        elif net_type is "Custom":
            data = data.reshape([1] + list(data.shape))
            data = np.swapaxes(data, 2, 3)
            data = np.swapaxes(data, 1, 2)
        # Set the dimension order to (BATCH, DEPTH, WIDTH, HEIGHT)

    return data


def main():
    print load(sys.argv[1])


# img = load_image(sys.argv[1], False)
# print img


if __name__ == '__main__':
    main()
