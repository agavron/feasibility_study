import json
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
import numpy as np


def reduce(img, factor):
    """ Reducing of image's block by averaging

    :param img: input block of image
    :param factor: scale of reducing
    :return: reduced block of image
    """
    result = np.zeros((img.shape[0] // factor, img.shape[1] // factor))
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i, j] = np.mean(img[i * factor: (i + 1) * factor,
                                   j * factor: (j + 1) * factor])
    return result


def rotate(img, angle):
    """ Rotation of image's block by angle

    :param img: input block of image
    :param angle: angle divisible by 90
    :return: rotated block
    """
    return ndimage.rotate(img, angle, reshape=False)


def flip(img, direction):
    """ Mirror flip/ no flip

    :param img: input image's block
    :param direction: 1 - stay the same, -1 - flip
    :return: flipped/not flipped block
    """
    return img[::direction, :]


def applyTransformation(img, direction, angle, contrast=1.0, brightness=0.0):
    """ Changing of image's block by parameters below except contrast and brightness

    :param img: input image's block
    :param direction: 1 - stay the same, -1 - flip
    :param angle: angle divisible by 90
    :param contrast: stay the same = 1.0
    :param brightness: stay the same = 0.0
    :return: transformed image's block
    """
    return contrast * rotate(flip(img, direction), angle) + brightness


def findContrastAndBrightness2(D, S):
    """ Find the most optimal contrast and brightness for source block S to reduce difference between
    source block S and destination block D by least-squares solution

    :param D: destination block - what we are going to compress
    :param S: source block - from list of all transformations
    :return: contrast and brightness
    """
    # Fit the contrast and the brightness
    A = np.concatenate((np.ones((S.size, 1)), np.reshape(S, (S.size, 1))), axis=1)
    b = np.reshape(D, (D.size,))
    x, _, _, _ = np.linalg.lstsq(A, b)
    # x = optimize.lsq_linear(A, b, [(-np.inf, -2.0), (np.inf, 2.0)]).x
    return x[1], x[0]


def plotImage(img):
    plt.figure()
    plt.imshow(img)
    plt.show()


class Image:
    def __init__(self, image, path):
        self.path = path
        self.image = mpimg.imread(self.path + image)

    def greyscale(self):
        if len(self.image.shape) != 3:
            self.image = self.image
        else:
            self.image = np.mean(self.image[:, :, :2], 2)

    def centerCrop(self, size):
        x = (self.image.shape[1] - size) // 2
        y = (self.image.shape[0] - size) // 2
        self.image = self.image[y: y + size, x: x + size]

    def reduce(self, factor):
        reduced_image = np.zeros((self.image.shape[0] // factor, self.image.shape[1] // factor))
        for i in range(reduced_image.shape[0]):
            for j in range(reduced_image.shape[1]):
                reduced_image[i, j] = np.mean(self.image[i * factor: (i + 1) * factor,
                                              j * factor: (j + 1) * factor])
        self.image = reduced_image

    def preprocessing(self, size, factor):
        self.greyscale()
        self.centerCrop(size)
        self.reduce(factor)


class TransformSpace:
    def __init__(self, image, source_size, destination_size, step):
        self.image = image
        self.source_size = source_size
        self.destination_size = destination_size
        self.step = step

    def generateAllTransformations(self):
        factor = self.source_size // self.destination_size
        transformed_blocks = []
        for k in range((self.image.shape[0] - self.source_size) // self.step + 1):
            for l in range((self.image.shape[1] - self.source_size) // self.step + 1):
                # Extract the source block and reduce it to the shape of destination
                S = reduce(self.image[k * self.step: k * self.step + self.source_size,
                           l * self.step: l * self.step + self.source_size], factor)
                # plotImage(S)
                # Generate all possible transformed blocks
                for direction, angle in candidates:
                    transformed_blocks.append((k, l, direction, angle, applyTransformation(S, direction, angle)))
        return transformed_blocks


class Packing:
    def __init__(self, image, source_size, destination_size, step, transformed_blocks):
        """

        :param image: input image
        :param source_size: size of source block S
        :param destination_size: size of block of input image which is going to be compressed
        :param step: for overlapping/non-overlapping
        :param transformed_blocks: list with all transformations of source blocks
        """
        self.image = image
        self.source_size = source_size
        self.destination_size = destination_size
        self.step = step
        self.transformed_blocks = transformed_blocks

    def compress(self):
        """ Compressing of input image

        :return: (result of compressing, namely all fitted transformations of source blocks;
        just all transformations of source blocks without fitted contrast and brightness)
        """
        transformations = []
        transformations_to_json = []
        for i in range(self.image.shape[0] // self.destination_size):
            transformations.append([])
            transformations_to_json.append([])
            for j in range(self.image.shape[1] // self.destination_size):
                #print(i, j)
                transformations[i].append(None)
                transformations_to_json[i].append(None)
                min_d = float('inf')
                # Extract the destination block
                D = self.image[i * self.destination_size: (i + 1) * self.destination_size,
                    j * self.destination_size: (j + 1) * self.destination_size]
                # Test all possible transformations and take the best one
                for k, l, direction, angle, S in self.transformed_blocks:
                    trans_block = S
                    contrast, brightness = findContrastAndBrightness2(D, S)
                    S = contrast * S + brightness
                    d = np.sum(np.square(D - S))
                    if d < min_d:
                        min_d = d
                        transformations[i][j] = (k, l, direction, angle, contrast, brightness)
                        # initial trans_block is interested because 1) this is raw source block without fitted contrast
                        # and brightness. It's going to be used for other images, not specifically for this case;
                        # 2) this block is the most useful because it has min error d
                        transformations_to_json[i][j] = (k, l, direction, angle, trans_block)
        return transformations, transformations_to_json


class Unpacking:
    def __init__(self, transformations, source_size, destination_size, step, num_iterations):
        """ This function is not used for final goal here (saving transformations of train to the file), but for
        checking of decompressing

        :param transformations:
        :param source_size:
        :param destination_size:
        :param step:
        :param num_iterations:
        """
        self.transformations = transformations
        self.source_size = source_size
        self.destination_size = destination_size
        self.step = step
        self.num_iterations = num_iterations

    def decompress(self):
        factor = self.source_size // self.destination_size
        height = len(self.transformations) * self.destination_size
        width = len(self.transformations[0]) * self.destination_size
        iterations = [np.random.randint(0, 256, (height, width))]
        cur_img = np.zeros((height, width))
        for i_iter in range(self.num_iterations):
            #print(i_iter)
            for i in range(len(self.transformations)):
                for j in range(len(self.transformations[i])):
                    k, l, flip, angle, contrast, brightness = self.transformations[i][j]
                    S = reduce(iterations[-1][k * self.step: k * self.step + self.source_size,
                               l * self.step: l * self.step + self.source_size], factor)
                    D = applyTransformation(S, flip, angle, contrast, brightness)
                    cur_img[i * self.destination_size: (i + 1) * self.destination_size,
                    j * self.destination_size: (j + 1) * self.destination_size] = D
            iterations.append(cur_img)
            cur_img = np.zeros((height, width))
        return iterations


def clean_transformations(transformations_to_json):
    """ Doesn't matter which exactly blocks will be used for decompressing. We are interested in all input
    transformations of source blocks S. But this list is too huge. This function saves only blocks with unique
    coordinates (k, l) for reducing of list's size

    :param transformations_to_json: all transformations of source blocks with (k, l) coordinates
    :return: cleaned list of transformations (approximately 4 times smaller in compare to input size)
    """
    final_transformations_to_json = []
    k_l_list = []
    for i in range(len(transformations_to_json)):
        for j in range(len(transformations_to_json[i])):
            k, l, flip, angle, im_array = transformations_to_json[i][j]
            if (k, l) not in k_l_list:
                temp_array = transformations_to_json[i][j][4].tolist()
                final_transformations_to_json.append((k, l, flip, angle, temp_array))
                k_l_list.append((k, l))
    return final_transformations_to_json


def create_transform(directory, candidates):
    """ Creation of transformations for train subset and saving them to json
    """

    dict_to_json = {}
    count = 0
    for file in os.listdir(directory):
        if file.endswith('.JPEG'):
            print(file)
            im = Image(file, directory)
            if im.image.shape[0] >= 256 and im.image.shape[1] >= 256:
                im.preprocessing(size=256, factor=4)
                #plotImage(im.image)

                transform = TransformSpace(image=im.image, source_size=8, destination_size=4, step=8)
                transformed_blocks = transform.generateAllTransformations()

                packing = Packing(image=im.image, source_size=8, destination_size=4, step=8,
                                  transformed_blocks=transformed_blocks)
                (transformations, transformations_to_json) = packing.compress()

                final_transformations_to_json = clean_transformations(transformations_to_json)

                dict_to_json[file] = final_transformations_to_json

                unpacking = Unpacking(transformations=transformations, source_size=8, destination_size=4, step=8,
                                      num_iterations=8)
                source = unpacking.decompress()[-1]
                #plotImage(source)

                count += 1
                print(count)

    with open(directory + 'train_transformations.json', 'w') as fp:
        json.dump(dict_to_json, fp, indent=4)

    with open(directory + 'train_transformations.json', 'r') as test_jf:
        json_log = json.load(test_jf)
        print('Success')


if __name__ == '__main__':
    """ Save all useful transformations (based on fractal compressing) of source blocks of each image from train to the 
    file. For further using of these blocks for compressing and decompressing of other images
    """

    # Mirror flip
    directions = [1, -1]
    # Angles divisible by 90 for simplicity
    angles = [0, 90, 180, 270]
    # All permutations of directions and angles for creation of pairs
    candidates = [[direction, angle] for direction in directions for angle in angles]

    directory = '/Users/alexeygavron/Documents/environments/data/src/src/data/images_dataset/train/'
    create_transform(directory, candidates)