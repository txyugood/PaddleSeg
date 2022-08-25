import os

import h5py
import cv2
import numpy as np
from PIL import Image


def get_color_map_list(num_classes):
    """
    Returns the color map for visualizing the segmentation mask,
    which can support arbitrary number of classes.
    Args:
        num_classes (int): Number of classes.
    Returns:
        (list). The color map.
    """

    num_classes += 1
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = color_map[3:]
    return color_map


def main():
    origin_dir = "/Users/alex/Downloads/project_TransUNet/data/Synapse"

    target_dir = "/Users/alex/Downloads/project_TransUNet/data/Synapse_image"

    sample_list = open(
        os.path.join(origin_dir, 'lists/lists_Synapse', 'train.txt')).readlines(
        )
    color_map = get_color_map_list(256)
    train_dir = os.path.join(target_dir, 'train')
    test_dir = os.path.join(target_dir, 'test')
    os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'labels'), exist_ok=True)

    # train_lines = []
    # for sample in sample_list:
    #     sample = sample.strip('\n')
    #     data_path = os.path.join(origin_dir,'train_npz', sample +'.npz')
    #     data = np.load(data_path)
    #     image, label = data['image'], data['label']
    #     image = image * 255.0
    #     lbl_pil = Image.fromarray(label.astype(np.uint8), mode='P')
    #     lbl_pil.putpalette(color_map)
    #
    #     cv2.imwrite(os.path.join(train_dir,'images', sample +'.png'), image)
    #     lbl_pil.save(os.path.join(train_dir,'labels', sample +'.png'))
    #
    #     train_lines.append(os.path.join('train/images', sample +'.png') + " " +os.path.join('train/labels', sample +'.png') + "\n")
    # with open(os.path.join(target_dir, 'train.txt'), 'w+') as f:
    #     f.writelines(train_lines)

    test_lines = []
    sample_list = open(
        os.path.join(origin_dir, 'lists/lists_Synapse',
                     'test_vol.txt')).readlines()
    for sample in sample_list:
        sample = sample.strip('\n')
        filepath = os.path.join(origin_dir, 'test_vol_h5',
                                "{}.npy.h5".format(sample))
        data = h5py.File(filepath)
        images, labels = data['image'][:], data['label'][:]
        for i in range(images.shape[0]):
            im = images[i] * 255.0
            label = labels[i]
            lbl_pil = Image.fromarray(label.astype(np.uint8), mode='P')
            lbl_pil.putpalette(color_map)
            filename = sample + f'_{i:>04d}.png'
            cv2.imwrite(os.path.join(test_dir, 'images', filename), im)
            lbl_pil.save(os.path.join(test_dir, 'labels', filename))
            test_lines.append(
                os.path.join('test/images', filename) + " " + os.path.join(
                    'test/labels', filename) + "\n")
        with open(os.path.join(target_dir, 'test.txt'), 'w+') as f:
            f.writelines(test_lines)


if __name__ == '__main__':
    main()
