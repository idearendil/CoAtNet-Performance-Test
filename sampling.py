import numpy as np
import cv2
from random import randint, random
import random
from PIL import Image
import os
import multiprocessing

src_dir = './covered_data/train/others/'
dest_dir = './mixed_data/train/others/'
images = os.listdir(src_dir)
process_num = 8

def copying(proc_id):

    cnt = 0
    for idx, an_image in enumerate(images):
        
        if idx < len(images) // process_num * proc_id:
            continue
        if idx >= len(images) // process_num * (proc_id + 1):
            break

        img = Image.open(src_dir + an_image)
        img.save(dest_dir + "cov_" + an_image)
        
        if idx % 1000 == 0:
            print(cnt)
            cnt += 1


if __name__ == '__main__':

    procs = []
    for proc_id in range(process_num):
        p = multiprocessing.Process(target=copying, args=(proc_id, ))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
        
    
