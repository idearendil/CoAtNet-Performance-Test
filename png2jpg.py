import numpy as np
import cv2
from random import randint, random
import random
from PIL import Image
import os
import multiprocessing

# random.seed(0)

image_dir = './data/train/others/'
save_dir = './data_jpg/train/others/'
images = os.listdir(image_dir)
process_num = 10

def covering(proc_id):

    for idx, an_image in enumerate(images):
        
        if idx < len(images) // process_num * proc_id:
            continue
        if idx >= len(images) // process_num * (proc_id + 1):
            break

        org_img = Image.open(image_dir + an_image)
        img = org_img.convert('RGB').resize((70, 70))
        img.save(save_dir + an_image[:-3] + 'jpg')
        
        if idx % 1000 == 0:
            print(idx)


if __name__ == '__main__':

    procs = []
    for proc_id in range(process_num):
        p = multiprocessing.Process(target=covering, args=(proc_id, ))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()