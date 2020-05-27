import matplotlib.pyplot as plt
import struct
import os
import cv2
import sys
import numpy as np
from PIL import Image
import random

random.seed(1)

PATH_TO_TEXTURES = "textures"
THRESHOLD = 10

DELTAS = (
    (-1, 0),
    (-1, 1),
    (-1, -1),
    (1, 0),
    (1, 1),
    (1, -1),
    (0, 1),
    (0, -1)
)
SIZE = 28

def binary_threshold(x):
    if x < THRESHOLD:
        return 0
    return 255

def read_idx(filename):
    with open(filename, 'rb') as f:
        _, _, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

def in_bounds(coords):
    return coords[0] >= 0 and coords[0] < SIZE and coords[1] >= 0 and coords[1] < SIZE

def is_boundary_pixel(coords, bitmap):
    x = coords[0]
    y = coords[1]
    for deltaX, deltaY in DELTAS:
        neighbors = (x + deltaX, y + deltaY)
        if in_bounds(neighbors):
            if bitmap[neighbors[1]][neighbors[0]] == 0:
                return True
        else:
            return True
    return False

def inside_boundary(coord, boundary_set):
    #hotspot
    x0 = coord[0]
    y0 = coord[1]
    values = [False, False, False, False]
    for iv, delta in enumerate(((1, 0), (0, 1), (-1, 0), (0, -1))):
        dx = delta[0]
        dy = delta[1]
        x = x0
        y = y0
        while True:
            if in_bounds((x, y)):
                if boundary_set[y][x] == 1:
                    values[iv] = True
                    break
                x += dx
                y += dy
            else:
                break
    return values[0] and values[1] and values[2] and values[3]

def fill_bitmap(arr):
    #given a numpy array, return a filled version
    height = arr.shape[0]
    width = arr.shape[1]
    binary_set = np.array(arr)
    for y in range(height):
        for x in range(width):
            binary_set[y][x] = binary_threshold(binary_set[y][x])

    #run DFS to find disjoint sets
    visited = np.zeros((height, width))
    def explore(coords, _set):
        visited[coords[1]][coords[0]] = 1
        _set.add(coords)
        color = binary_set[coords[1]][coords[0]]
        for deltaX, deltaY in DELTAS:
            neighbors = (coords[0] + deltaX, coords[1] + deltaY)
            if in_bounds(neighbors) \
                and color == binary_set[neighbors[1]][neighbors[0]] \
                and visited[neighbors[1]][neighbors[0]] == 0:
                explore(neighbors, _set)

    disjoint_sets = []
    for y in range(height):
        flag = False
        for x in range(width):
            if visited[y][x] == 0:
                _set = set()
                explore((x, y), _set)
                disjoint_sets.append(_set)
                test_item = next(iter(_set))
                if len(_set) > 392 and binary_set[test_item[1]][test_item[0]] == 255:
                    flag = True
                    break
        if flag:
            break
    
    def key(_set):
        coord = next(iter(_set))
        if binary_set[coord[1]][coord[0]] == 255:
            return len(_set)
        return 0

    max_set = max(disjoint_sets, key=key)
    plt_arr = np.zeros((height, width))
    result = np.zeros((height, width))

    #find outline of max set
    boundary_set = np.zeros((height, width))
    for coord in max_set:
        plt_arr[coord[1]][coord[0]] = 200
        result[coord[1]][coord[0]] = 255   
        if is_boundary_pixel(coord, binary_set): #hotspot
            boundary_set[coord[1]][coord[0]] = 1
            plt_arr[coord[1]][coord[0]] = 255

    #for each node not in max set, if inside boundary, then flip it on
    #to check if in boundary, extend rays in 4 directions. If all 4 hit a boundary, then
    #   it is inside.
    for _set in disjoint_sets:
        if _set != max_set:
            for coord in _set:
                plt_arr[coord[1]][coord[0]] = 100
                if inside_boundary(coord, boundary_set):
                    plt_arr[coord[1]][coord[0]] = 200
                    result[coord[1]][coord[0]] = 255

    #show_bitmap(plt_arr) #debug array
    #show_bitmap_sbs(arr, result)
    return result

def load_texture(filename):
    print(filename)
    im = np.array(Image.open(filename))
    return im

def scale_texture(texture, scale=1):
    if scale > 1 or scale < 0:
        raise Exception("invalid scale: must be in [0, 1]") 
    size_new = int(32 + ((256-32)*(1-scale)))
    res = cv2.resize(texture, dsize=(size_new, size_new), interpolation=cv2.INTER_CUBIC)
    #crop to center
    resize = (size_new-32)//2 
    return res[resize:resize+32, resize:resize+32]


def add_to_filled(filled, texture, offset=(0,0)):
    result = np.zeros(filled.shape)
    ry = 0
    rx = 0
    for y in range(offset[0], filled.shape[0]+offset[0]):
        rx = 0
        for x in range(offset[1], filled.shape[1]+offset[1]):
            if (filled[ry][rx] == 255):
                result[ry][rx] = texture[y][x]
            rx += 1
        ry += 1
    return result

def interpolate_textures(texture_list):
    t1 = texture_list[0]
    result = np.zeros(t1.shape)
    for y in range(t1.shape[0]):
        for x in range(t1.shape[1]):
            val = int(
                sum((int(t[y][x]) for t in texture_list)) / len(texture_list)
            )
            #Watch out for overflow
            #   convert to int explicitly before interpolation
            result[y][x] = val
    return result

def show_bitmap(arr1):
    # plot the sample
    plt.imshow(arr1, cmap='gray')
    plt.show()

def show_bitmap_sbs(arr1, arr2):
    # plot the sample
    plt.imshow([np.concatenate([arr1[i], arr2[i]]) for i in range(arr1.shape[0])], cmap='gray')
    plt.show()

def add_noise(arr, intensity=0):
    NOISE_LEVEL = 64
    result = np.zeros(arr.shape)
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            delta = int(intensity * random.randint(-NOISE_LEVEL, NOISE_LEVEL))
            val = arr[y][x] + delta
            result[y][x] = max(0, min(255, val))
    return result

class TexturedFMNIST():
    def __init__(self, texture_dir=PATH_TO_TEXTURES, fmnist_dir='.'):
        self.imgs = read_idx(os.path.join(fmnist_dir, "train-images-idx3-ubyte"))
        self.labels = read_idx(os.path.join(fmnist_dir, "train-labels-idx1-ubyte"))


        self.textures = [load_texture(os.path.join(texture_dir, i)) for i in os.listdir(texture_dir)]


    def build_class(self, class_num, texture_choices=[], texture_interpolation=0, texture_rescale=True, texture_aug=True, aug_intensity=0.5):
        '''
        texture_rescale: either bool or float oor string? controls how to rescale textures when applying
        texture_aug: whether to add some noise to texture scaling and orientation before application
        '''
        # Loop over self.get_textured_sample randomly sampling noise for texture application if texture_aug is True
        return #[np arrays of images with textures applied]

    def get_textured_sample(self, img_index, sample_fname, texture_choices=[], texture_interpolation=0, texture_rescale=True, texture_aug=True):
        offset = (0, 0)
        noise = 0
        if texture_aug:
            offset = (random.randint(0, 4), random.randint(0, 4))
            noise = 1
            
        texture = interpolate_textures([
            add_noise(scale_texture(random.choice(self.textures)), noise),
            add_noise(scale_texture(random.choice(self.textures)), noise)
        ])

        i1 = add_to_filled(
            fill_bitmap(self.imgs[img_index]), texture, offset
        )
        return i1, self.labels[img_index]

#with PyCallGraph(output=GraphvizOutput()):
 #   test()
def test():
    tf = TexturedFMNIST()
    for i in range(1000):
        show_bitmap(tf.get_textured_sample(0, False, False)[0])
        print(i)