# Assumes you've unzipped the MNIST data files from 
# http://yann.lecun.com/exdb/mnist/ in ../MNIST_data

import os
import sys
import time
import random

def get_mnist_training_images(count, dump_images_to=None):
    """
    Returns a random subset of the MNIST test images.

    Return data type:
        {ID -> (label, image data array)}

    ID ranges from 60001 to 70000
    """
    assert 0 <= count <= 60000, "Max 60,000 images"
    if dump_images_to is not None:
        assert os.path.exists(dump_images_to)
    labels = _read_training_labels()
    all_ids = list(range(60000))
    ids_to_get = _pick_ids(all_ids, count)
    images = _get_training_images_with_ids(ids_to_get)
    image_labels = [labels[id_to_get] for id_to_get in ids_to_get]
    assert len(images) == len(image_labels)
    assert len(ids_to_get) == len(image_labels)
    results = {}
    for i in range(len(images)):
        results[ids_to_get[i]] = (image_labels[i], images[i])
    if dump_images_to:
        _dump_images(results, dump_images_to)
    return results

def get_mnist_testing_images(count, dump_images_to=None):
    """
    Returns a random subset of the MNIST test images.

    Return data type:
        {ID -> (label, image data array)}

    ID ranges from 60001 to 70000
    """
    assert 0 <= count <= 10000, "Max 10,000 images"
    if dump_images_to is not None:
        assert os.path.exists(dump_images_to)
    labels = _read_testing_labels()
    all_ids = list(range(10000))
    ids_to_get = _pick_ids(all_ids, count)
    images = _get_testing_images_with_ids(ids_to_get)
    image_labels = [labels[id_to_get] for id_to_get in ids_to_get]
    ids_to_get = [id_to_get+60001 for id_to_get in ids_to_get]
    assert len(images) == len(image_labels)
    assert len(ids_to_get) == len(image_labels)
    results = {}
    for i in range(len(images)):
        results[ids_to_get[i]] = (image_labels[i], images[i])
    if dump_images_to:
        _dump_images(results, dump_images_to)
    return results

def _pick_ids(all_ids, count):
    """Chooses <count> ids from the list <all_ids> w/o replacement."""
    chosen_ids = []
    while len(chosen_ids) < count:
        random_idx = random.choice(list(range(len(all_ids))))
        tmp_id = all_ids.pop(random_idx)
        chosen_ids.append(tmp_id)
    return sorted(chosen_ids)

def _dump_images(data, path):
    for image_id, image_data in data.items():
        _dump_image(image_id, image_data, path)

def _dump_image(image_id, image_data, path):
    """Dumps a 28x28 grayscale as a BMP"""
    # Header bytes
    magic_letters = ["0x42", "0x4D"] 
    total_file_size = ["0x66", "0x09"] + ["0x00"]*2 # 2406
    reserved = ["0x00"]*4
    pixel_offset = ["0x36"] + ["0x00"]*3 
    bitmap_info_header = ["0x28"] + ["0x00"]*3
    pixel_width = ["0x1C"] + ["0x00"]*3 
    pixel_height = ["0x1C"] + ["0x00"]*3 
    color_plane = ["0x01", "0x00"]
    bits_per_pixel = ["0x18", "0x00"]
    disable_compression = ["0x00"]*4
    size_of_raw_data =  ["0x10", "0x03"]+ ["0x00"]*2 # 784
    horiz_resolution =  ["0x13", "0x0B"] + ["0x00"]*2
    vert_resolution =  ["0x13", "0x0B"] + ["0x00"]*2
    color_number = ["0x00"]*4
    important_colors = ["0x00"]*4
   
    # Prep image data to be built from bottom up
    ubyte_rows = []
    tmp_row = []
    for ubyte in image_data[1]:
        tmp_row.append(ubyte)
        if len(tmp_row) == 28:
            ubyte_rows.append(tmp_row)
            tmp_row = []
    assert len(ubyte_rows) == 28
    for ubyte_row in ubyte_rows:
        assert len(ubyte_row) == 28
    ubyte_rows.reverse()
    pixel_data = []
   
    # Generate grayscale pixel data from intensities
    reflect_table = {}
    for i,j in zip(range(256), sorted(range(256), key=lambda x: -x)):
        reflect_table[i] = j
    for ubyte_row in ubyte_rows:
        for ubyte in ubyte_row:
            pixel_data.append(hex(reflect_table[ubyte])) # B
            pixel_data.append(hex(reflect_table[ubyte])) # G
            pixel_data.append(hex(reflect_table[ubyte])) # Rk

    # Dump BMP
    img_data = (magic_letters + total_file_size + reserved + pixel_offset +
                bitmap_info_header + pixel_width + pixel_height + color_plane +
                bits_per_pixel + disable_compression + size_of_raw_data +
                horiz_resolution + vert_resolution + color_number +
                important_colors + pixel_data)
    img_path = os.path.join(path, "{}-{}.bmp".format(image_id, image_data[0]))
    with open(img_path, 'wb') as output:
        output.write(bytearray(int(i, 16) for i in img_data))

def _get_training_images_with_ids(ids_to_get):
    path = "../MNIST_data/train-images-idx3-ubyte"
    return _get_images(path, ids_to_get, [234,96], 60000)
   
def _get_testing_images_with_ids(ids_to_get):
    path = "../MNIST_data/t10k-images-idx3-ubyte"
    return _get_images(path, ids_to_get, [39,16], 10000)

def _get_images(path, ids_to_get, magic_num, count):
    with open(path, "rb") as fin:
        # Magic Number
        assert ord(fin.read(1)) == 0
        assert ord(fin.read(1)) == 0
        assert ord(fin.read(1)) == 8
        assert ord(fin.read(1)) == 3
        # Count
        assert ord(fin.read(1)) == 0
        assert ord(fin.read(1)) == 0
        assert ord(fin.read(1)) == magic_num[0]
        assert ord(fin.read(1)) == magic_num[1]
        # Rows
        assert ord(fin.read(1)) == 0
        assert ord(fin.read(1)) == 0
        assert ord(fin.read(1)) == 0
        assert ord(fin.read(1)) == 28
        # Columns
        assert ord(fin.read(1)) == 0
        assert ord(fin.read(1)) == 0
        assert ord(fin.read(1)) == 0
        assert ord(fin.read(1)) == 28
        results = []
        curr_id = 0
        for _ in range(count):
            img = []
            for i in range(28):
                for j in range(28):
                    img.append(ord(fin.read(1)))
            if curr_id in ids_to_get:
                results.append(img)
            curr_id += 1
        assert len(results) == len(ids_to_get)
        return results

def _read_training_labels():
    path = "../MNIST_data/train-labels-idx1-ubyte"
    return _read_labels(path, [234,96], 60000)

def _read_testing_labels():
    path = "../MNIST_data/t10k-labels-idx1-ubyte"
    return _read_labels(path, [39,16], 10000)

def _read_labels(path, magic_num, count):
    with open(path, "rb") as fin:
        # Magic Number
        assert ord(fin.read(1)) == 0
        assert ord(fin.read(1)) == 0
        assert ord(fin.read(1)) == 8
        assert ord(fin.read(1)) == 1
        # Count
        assert ord(fin.read(1)) == 0
        assert ord(fin.read(1)) == 0
        assert ord(fin.read(1)) == magic_num[0]
        assert ord(fin.read(1)) == magic_num[1]
        labels = []
        while True:
            byte = fin.read(1)
            if not byte:
                break
            labels.append(ord(byte))
        assert len(labels) == count
        return labels

if __name__ == "__main__":
    tick = time.time()
    print('Dumping 20 training images to ../media/')
    get_mnist_training_images(20, '../media/')
    print('\tTook {0:.2f} sec'.format(time.time() - tick))
    tock = time.time()
    print('Dumping 10 testing images to ../media/')
    get_mnist_testing_images(10, '../media/')
    print('\tTook {0:.2f} sec'.format(time.time() - tock))
    print('Total time: {0:.2f} sec'.format(time.time() - tick))


