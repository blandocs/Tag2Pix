"""
modified from https://github.com/lllyasviel/sketchKeras
"""
import cv2
import numpy as np
import os
from keras.models import load_model
from scipy import ndimage
from .xdog_blend import *


def get_light_map_single(img):
    gray = img
    gray = gray[None]
    gray = gray.transpose((1,2,0))
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    gray = gray.reshape((gray.shape[0],gray.shape[1]))
    highPass = gray.astype(int) - blur.astype(int)
    highPass = highPass.astype(np.float)
    highPass = highPass / 128.0
    return highPass

def normalize_pic(img):
    img = img / np.max(img)
    return img

def resize_img_512_3d(img):
    zeros = np.zeros((1,3,512,512), dtype=np.float)
    zeros[0 , 0 : img.shape[0] , 0 : img.shape[1] , 0 : img.shape[2]] = img
    return zeros.transpose((1,2,3,0))

def to_keras_enhanced(img):
    mat = img.astype(np.float)
    mat[mat<0.1] = 0
    mat = - mat + 1
    mat = mat * 255.0
    mat[mat < 0] = 0
    mat[mat > 255] = 255
    mat=mat.astype(np.uint8)
    mat = ndimage.median_filter(mat, 1)

    return mat

def get_light_map(img):
    from_mat = img
    width = float(from_mat.shape[1])
    height = float(from_mat.shape[0])
    new_width = 0
    new_height = 0

    if (width > height):
        if width != 512:
            from_mat = cv2.resize(from_mat, (512, int(512 / width * height)), interpolation=cv2.INTER_AREA)
        new_width = 512
        new_height = int(512 / width * height)
    else:
        if height != 512:
            from_mat = cv2.resize(from_mat, (int(512 / height * width), 512), interpolation=cv2.INTER_AREA)
        new_width = int(512 / height * width)
        new_height = 512

    from_mat = from_mat.transpose((2, 0, 1))
    light_map = np.zeros(from_mat.shape, dtype=np.float)
    for channel in range(3):
        light_map[channel] = get_light_map_single(from_mat[channel])

    light_map = normalize_pic(light_map)
    light_map = resize_img_512_3d(light_map)

    return new_height, new_width, light_map

mod = None
def get_mod():
    global mod
    if mod is None:
        mod = load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sketchKeras.h5'))
    return mod

def get_keras_enhanced(img):
    mod = get_mod()
    new_height, new_width, light_map = get_light_map(img)

    # TODO: batch this!
    line_mat = mod.predict(light_map, batch_size=1)

    line_mat = line_mat.transpose((3, 1, 2, 0))[0]
    line_mat = line_mat[0:int(new_height), 0:int(new_width), :]
    line_mat = np.amax(line_mat, 2)

    keras_enhanced = to_keras_enhanced(line_mat)
    return keras_enhanced

def get_keras_high_intensity(img, intensity=1.7):
    keras_img = get_keras_enhanced(img)
    return add_intensity(keras_img, intensity)

def batch_keras_enhanced(img_list):
    mod = get_mod()
    light_maps = list(map(get_light_map, img_list))

    hw_list = [(h, w) for h, w, _ in light_maps]
    light_map = [l for _, _, l in light_maps]

    batch_light_map = np.concatenate(light_map, axis=0)
    batch_line_mat = mod.predict(batch_light_map, batch_size=len(img_list))

    mat_list = np.array_split(batch_line_mat, len(img_list))
    batch_line_mat = map(lambda map: map.transpose((3, 1, 2, 0))[0], mat_list)

    result_list = []
    for line_mat, (new_height, new_width) in zip(batch_line_mat, hw_list):
        mat = line_mat[0:int(new_height), 0:int(new_width), :]
        mat = np.amax(mat, 2)
        keras_enhanced = to_keras_enhanced(mat)
        result_list.append(keras_enhanced)
    
    return result_list
