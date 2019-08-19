import cv2
import numpy as np
from scipy import ndimage

def dog(img, size=(0,0), k=1.6, sigma=0.5, gamma=1):
    img1 = cv2.GaussianBlur(img, size, sigma)
    img2 = cv2.GaussianBlur(img, size, sigma * k)
    return (img1 - gamma * img2)

def xdog(img, sigma=0.5, k=1.6, gamma=1, epsilon=1, phi=1):
    aux = dog(img, sigma=sigma, k=k, gamma=gamma) / 255
    for i in range(0, aux.shape[0]):
        for j in range(0, aux.shape[1]):
            if(aux[i, j] < epsilon):
                aux[i, j] = 1*255
            else:
                aux[i, j] = 255*(1 + np.tanh(phi * (aux[i, j])))
    return aux

def get_xdog_image(img, sigma=0.4, k=2.5, gamma=0.95, epsilon=-0.5, phi=10**9):
    xdog_image = xdog(img, sigma=sigma, k=k, gamma=gamma, epsilon=epsilon, phi=phi).astype(np.uint8)
    return xdog_image

def add_intensity(img, intensity):
    if intensity == 1:
        return img
    inten_const = 255.0 ** (1 - intensity)
    return (inten_const * (img ** intensity)).astype(np.uint8)

def blend_xdog_and_sketch(illust, sketch, intensity=1.7, degamma=(1/1.5), blend=0, **kwargs):
    gray_image = cv2.cvtColor(illust, cv2.COLOR_BGR2GRAY)
    gamma_sketch = add_intensity(sketch, intensity)

    if blend > 0:
        xdog_image = get_xdog_image(gray_image, **kwargs)
        xdog_blurred = cv2.GaussianBlur(xdog_image, (5, 5), 1)
        xdog_residual_blur = cv2.addWeighted(xdog_blurred, 0.75, xdog_image, 0.25, 0)

        if gamma_sketch.shape != xdog_residual_blur.shape:
            gamma_sketch = cv2.resize(gamma_sketch, xdog_residual_blur.shape, interpolation=cv2.INTER_AREA)
        
        blended_image = cv2.addWeighted(xdog_residual_blur, blend, gamma_sketch, (1-blend), 0)
    else:
        blended_image = gamma_sketch

    return add_intensity(blended_image, degamma)