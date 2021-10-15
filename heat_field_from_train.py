import json
from random import shuffle
import sklearn.preprocessing as pr
import numpy as np
import matplotlib.pyplot as plt


def blocks_squeeze(blocks_array):
    """ Returning of list of sub lists to common list

    :param blocks_array:
    :return: list of blocks
    """
    blocks_array_squeeze = []
    for block_list in blocks_array:
        for block in block_list:
            blocks_array_squeeze.append(block)

    return blocks_array_squeeze


def compare_blocks(blocks_array_shuffle):
    """ Comparison of blocks by euclidean distance within each sub list

    :param blocks_array_shuffle: list of sub lists of blocks
    :return: list of sub lists of blocks but with less overall size
    """
    heat_field_full = []
    overall_progress = 0
    distance = []
    count = 0
    for blocks_list in blocks_array_shuffle:
        heat_field = []
        heat_field.append(blocks_list[0])
        for block in blocks_list:
            # print(count)
            for matrix in heat_field:
                norm_matrix = pr.normalize(matrix, norm='l1')
                norm_block = pr.normalize(block, norm='l1')
                euclidean_dist = np.linalg.norm(norm_matrix - norm_block)
                # euclidean_dist = np.linalg.norm(np.asarray(matrix) - np.asarray(block))
                distance.append(euclidean_dist)
                if euclidean_dist > 0.02:
                    heat_field.append(block)
                    break
                elif 0 < euclidean_dist <= 0.02:
                    count += 1
                    # print(count)
                    break
        heat_field_full.append(heat_field)
        overall_progress += 1
        # print('Overall progress: ' + str(overall_progress) + '/' + str(len(blocks_array_shuffle)))

    return heat_field_full


def mixing(blocks_array, num_elem):
    """ Shuffle of blocks in list and split them into sub lists with size = num_elem (200, except the last sub list)

    :param blocks_array: list of blocks
    :param num_elem: length of sub list
    :return: shuffled list of sub lists
    """
    shuffle(blocks_array)
    blocks_array_update = []
    num_sets = len(blocks_array) // num_elem
    reminder = len(blocks_array) % 3

    for i in range(num_sets):
        blocks_array_update.append(blocks_array[:num_elem])
        del blocks_array[:num_elem]

    if reminder != 0:
        blocks_array_update.append(blocks_array)

    return blocks_array_update


if __name__ == '__main__':
    """ The length of train_transformations list is huge for further comparison of each block of each file from val
    and test. The main goal here is removing of those blocks which are close to each other in terms of l-norm. Only
    unique blocks are remain. Result is saved to json file
    """
    directions = [1, -1]
    angles = [0, 90, 180, 270]
    candidates = [[direction, angle] for direction in directions for angle in angles]
    directory = '/Users/alexeygavron/Documents/environments/data/src/src/data/images_dataset/'

    with open(directory + 'train_transformations.json', 'r') as test_jf:
        json_log = json.load(test_jf)
        print('Success')

    number_of_blocks = 0
    for image in json_log:
        number_of_blocks += len(json_log[image])

    print(number_of_blocks)

    blocks_array_init = []
    for image in json_log:
        for item in json_log[image]:
            blocks_array_init.append(item[4])

    num_iterations = 400
    # doesn't work for num_epochs > 1 due to changing of blocks_array size. Idea was to generate several json files
    # independently. One more epoch - one more independent json file
    num_epochs = 1

    for epoch in range(num_epochs):
        reducing_length = []
        blocks_array = blocks_array_init
        for iter in range(num_iterations):
            initial_len = len(blocks_array)
            blocks_array = mixing(blocks_array, num_elem=200)
            blocks_array = compare_blocks(blocks_array)
            blocks_array = blocks_squeeze(blocks_array)
            final_len = len(blocks_array)
            print('Epoch: ' + str(epoch) + ' Iteration ' + str(iter) + ': ' + str(initial_len - final_len))
            reducing_length.append(initial_len - final_len)

        dict_to_json = {}

        for item in range(final_len):
            dict_to_json[item] = blocks_array[item]

        with open(directory + 'core_' + str(1) + '.json', 'w') as fp:
            json.dump(dict_to_json, fp, indent=4)

        with open(directory + 'core_' + str(1) + '.json', 'r') as test_jf:
            json_log = json.load(test_jf)
            print('Success')

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(reducing_length)
        fig.savefig(directory + 'iterations_core_' + str(1) + '.png')  # save the figure to file
        plt.close(fig)