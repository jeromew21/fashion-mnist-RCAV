import matplotlib.pyplot as plt
import struct

import numpy as np

THRESHOLD = 10

def binary_threshold(x):
    if x < THRESHOLD:
        return 0
    return 255

def read_idx(filename):
    with open(filename, 'rb') as f:
        _, _, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

def in_bounds(coords, w, h):
    return all((coords[0] >= 0, coords[0] < h, coords[1] >= 0, coords[1] < w))

def is_boundary_pixel(coords, bitmap):
    for deltaX in (-1, 0, 1):
        for deltaY in (-1, 0, 1):
            if deltaX or deltaY:
                neighbors = (coords[0] + deltaX, coords[1] + deltaY)
                if in_bounds(neighbors, bitmap.shape[1], bitmap.shape[0]):
                    if bitmap[neighbors[1]][neighbors[0]] == 0:
                        return True
                else:
                    return True
    return False

def inside_boundary(coord, boundary_set):
    x0 = coord[0]
    y0 = coord[1]
    values = [False, False, False, False]
    for iv, delta in enumerate(((1, 0), (0, 1), (-1, 0), (0, -1))):
        dx = delta[0]
        dy = delta[1]
        x = x0
        y = y0
        while True:
            coord = (x, y)
            if in_bounds(coord, boundary_set.shape[1], boundary_set.shape[0]):
                if boundary_set[coord[1]][coord[0]] == 1:
                    values[iv] = True
                x += dx
                y += dy
            else:
                break
    return all(values)

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
        for deltaX in (-1, 0, 1):
            for deltaY in (-1, 0, 1):
                if deltaX or deltaY:
                    neighbors = (coords[0] + deltaX, coords[1] + deltaY)
                    if in_bounds(neighbors, width, height) \
                        and color == binary_set[neighbors[1]][neighbors[0]] \
                        and visited[neighbors[1]][neighbors[0]] == 0:
                        explore(neighbors, _set)

    disjoint_sets = []
    for y in range(height):
        for x in range(width):
            if visited[y][x] == 0:
                _set = set()
                explore((x, y), _set)
                disjoint_sets.append(_set)
    
    print(disjoint_sets)
    def key(_set):
        coord = list(_set)[0]
        print(binary_set[coord[1]][coord[0]])
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
        if is_boundary_pixel(coord, binary_set):
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
    show_bitmap_sbs(arr, result)
    return result

def show_bitmap(arr1):
    # plot the sample
    plt.imshow(arr1, cmap='gray')
    plt.show()

def show_bitmap_sbs(arr1, arr2):
    # plot the sample
    plt.imshow([np.concatenate([arr1[i], arr2[i]]) for i in range(arr1.shape[0])], cmap='gray')
    plt.show()

imgs = read_idx("train-images-idx3-ubyte")
labels = read_idx("train-labels-idx1-ubyte")

for i in range(100):
    fill_bitmap(imgs[i])
