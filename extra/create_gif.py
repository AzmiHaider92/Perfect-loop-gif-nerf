import shutil
import cv2
import numpy as np
import glob
import os
from rembg import remove
import matplotlib.pyplot as plt
import imageio


if __name__ == '__main__':
    folder = r"C:\Users\azmih\Desktop\Projects\ComputerVisionLab\TensoRF\data\train"
    images = []
    rimgs = os.listdir(folder)
    for j in range(len(rimgs)):
        image_path = os.path.join(folder,rimgs[j])
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape
        newsize = (int(width / 5), int(height / 5))
        img = cv2.resize(img, newsize)

        #img[np.all(img == (0, 0, 0), axis=-1)] = (255,255,255)

        images.append(img)

    imageio.mimsave(f"{folder}\\rgb_maps.gif", images, format='GIF', fps=8)