import numpy as np
import cv2
from random import randint, random
import random
from PIL import Image
import os
import multiprocessing

# random.seed(0)

image_dir = './data/train/others/'
save_dir = './covered_data/train/others/'
images = os.listdir(image_dir)
process_num = 8

def covering(proc_id):

    for idx, an_image in enumerate(images):
        
        if idx < len(images) // process_num * proc_id:
            continue
        if idx >= len(images) // process_num * (proc_id + 1):
            break

        org_img = Image.open(image_dir + an_image)
        org_img = np.asarray(org_img)
        org_img = org_img[:,:,2::-1].copy()

        width, height = org_img.shape[0], org_img.shape[1]
        q_width1 = int(width / 4)
        q_height1 = int(height / 4)
        q_width2 = int(width * 3 / 4)
        q_height2 = int(height * 3 / 4)
        limit = min(width, height)

        # cover_color = (
        #     int(np.mean(org_img[q_width1:q_width2,q_height1:q_height2,0])),
        #     int(np.mean(org_img[q_width1:q_width2,q_height1:q_height2,1])),
        #     int(np.mean(org_img[q_width1:q_width2,q_height1:q_height2,2]))
        # )

        blue_color = (255, 0, 0)
        green_color = (0, 255, 0)
        red_color = (0, 0, 255)
        white_color = (255, 255, 255)
        cover_color = (0, 0, 0)

        blank_img = np.zeros((width, height, 3), np.uint8)
        center_area = np.array([[q_width1, q_height1], [q_width1, q_height2], [q_width2, q_height2], [q_width2, q_height1]], np.int32)
        only_center = cv2.fillConvexPoly(blank_img, center_area, white_color)

        while True:
            blank_img = np.zeros((width, height, 3), np.uint8)
            cover1 = np.array([[randint(0, limit), randint(0, limit)], [randint(0, limit), randint(0, limit)], [randint(0, limit), randint(0, limit)]], np.int32)
            plus_cover1 = cv2.fillConvexPoly(blank_img, cover1, white_color)
            cover2 = np.array([[randint(0, limit), randint(0, limit)], [randint(0, limit), randint(0, limit)], [randint(0, limit), randint(0, limit)]], np.int32)
            plus_cover2 = cv2.fillConvexPoly(plus_cover1, cover2, white_color)

            cover_pixel_num = np.count_nonzero(plus_cover2) / 3
            cover_ratio = cover_pixel_num / (width * height)
            if cover_ratio < 0.03:      # prevent too small cover
                continue

            center_cover1 = np.logical_and(plus_cover2, only_center).astype(dtype=np.uint8) * 255
            covered_pixel_num = np.count_nonzero(center_cover1) / 3
            covered_ratio = covered_pixel_num / (width * height / 4)
            if covered_ratio < 0.05 or covered_ratio > 0.35:    # prevent useless cover & too large cover
                continue
            threshold = abs(covered_ratio - 0.2) / 0.15
            rand_num = random.random()
            if rand_num > threshold:        # control probability distribution
                continue
            break

        img = cv2.fillConvexPoly(org_img, cover1, cover_color)
        img = cv2.fillConvexPoly(img, cover2, cover_color)
        cv2.imwrite(save_dir + an_image, img)
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