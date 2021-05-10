import json
import h5py
import os
from os.path import dirname
import cv2
import numpy as np
import pycocotools
import random

from cv2 import getRotationMatrix2D, warpAffine
import matplotlib.pyplot as plt
from skimage.measure import label
import detectron2.data.transforms as T
import scipy.io as scio
from detectron2.structures import BoxMode
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train')
    parser.add_argument('--dataset', default='01new')
    parser.add_argument('--oc', type=float, default=0.58)
    parser.add_argument('--hr', type=float, default=0.4)
    parser.add_argument('--is_nohead', type=bool, default=True)
    parser.add_argument('--is_hr', type=bool, default=True)
    return parser.parse_args()

def get_dicts(datadir, phase, meta_dir):
    idx = 0
    dataset_dicts = []
    depth_dir = os.path.join(os.getcwd(), datadir, phase, 'gray')
    seg_dir = os.path.join(os.getcwd(), datadir, phase, 'label')

    for file in os.listdir(depth_dir):
        if 'left' in file:
            oc = args.oc
        else:
            oc = args.oc - 0.8
        record = {}
        ## occulusion ##
        meta_path = os.path.join(meta_dir, file.split('.')[0]+'.mat')
        meta = scio.loadmat(meta_path)
        print(file)
        occulist = meta['oc'][0]
        depth_path = os.path.join(depth_dir, file)
        # seg_path = os.path.join(h5_dir, file.split('.')[0]+'_0.hdf5')
        # file = h5py.File(seg_path, 'r')
        seg_path = os.path.join(seg_dir, file)
        record['file_name'] = depth_path
        record['image_id'] = idx
        idx += 1
        a = cv2.imread(depth_path)
        height, width, _ = a.shape
        record['height'] = height
        record['width'] = width

        seg_img = cv2.imread(seg_path, -1)
        if len(seg_img.shape) == 3:
            seg_img = seg_img[:,:,0]
        seg_ids = np.unique(seg_img)
        # seg_img = file['label'][:]
        # seg_ids = np.unique(seg_img)

        seg_ids = [id for id in seg_ids if id != 0]
        objs = []
        for id in seg_ids:
            if id > len(occulist)-1:
                continue
            occ_r = occulist[id]
            bit_mask = (seg_img == id)
            y_idxs, x_idxs = np.where(bit_mask)
            size_label = np.sum(bit_mask)


            if (y_idxs.size != 0) and (x_idxs.size != 0) and (occ_r > args.oc):
                object_points = np.array([[x,y] for x,y in zip(x_idxs,y_idxs)])
                (c_x, c_y), (w, h), a = cv2.minAreaRect(object_points)
                if h > w:
                    angle = a + 90
                    object_length = h
                else:
                    angle = a
                    object_length = w
                judge_range = int(0.2*object_length)
                rot_mat = getRotationMatrix2D(center=(int(c_x), int(c_y)), angle=angle,
                                              scale=1)
                rotated_m = warpAffine(bit_mask.astype('uint8'), rot_mat, (int(width), int(height)))
                y_ids, x_ids = np.where(rotated_m == 1)
                start_x = np.min(x_ids)
                end_x = np.max(x_ids)
                left_mask = rotated_m[:,start_x:start_x+judge_range]
                right_mask = rotated_m[:,end_x-judge_range+1:end_x+1]
                label_wo_head = rotated_m.copy()
                if args.is_nohead:
                    if np.sum(left_mask) > np.sum(right_mask):
                        head = start_x
                        label_wo_head[:,start_x:start_x+judge_range] = 0
                    else:
                        head = end_x
                        label_wo_head[:,end_x-judge_range+1:end_x+1] = 0
                    rot_matr = getRotationMatrix2D(center=(int(c_x), int(c_y)), angle=-angle,
                                                  scale=1)
                    bit_mask = warpAffine(label_wo_head, rot_matr, (int(width), int(height)))
                    y_idxs, x_idxs = np.where(bit_mask)
                    object_points = np.array([[x,y] for x,y in zip(x_idxs,y_idxs)])
                    (c_x, c_y), (w, h), a = cv2.minAreaRect(object_points)

                # measure the ratio of pickable part
                if args.is_hr:
                    labels = label(rotated_m)
                    for id in np.unique(labels):
                        ys, xs = np.where(labels == id)
                        ll = np.min(xs)
                        rr = np.max(xs)
                        if ll == head or rr == head:
                            length_head = rr - ll
                            head_ratio = length_head/object_length
                            break

                    if head_ratio > args.hr:
                        rle = pycocotools.mask.encode(np.asarray(bit_mask, order="F"))
                        rle['counts'] = rle['counts'].decode()
                        obj = {
                            "bbox": [int(c_x), int(c_y), int(w), int(h), -a],
                            # "bbox_mode": BoxMode.XYXY_ABS,
                            "segmentation": rle,
                            "category_id": 0,
                            "occulusion": occ_r
                        }
                        objs.append(obj)
                else:
                    rle = pycocotools.mask.encode(np.asarray(bit_mask, order="F"))
                    rle['counts'] = rle['counts'].decode()
                    obj = {
                        "bbox": [int(c_x), int(c_y), int(w), int(h), -a],
                        # "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": rle,
                        "category_id": 0,
                        "occulusion": occ_r
                    }
                    objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


if __name__ == "__main__":
    args = get_args()
    meta_dir = os.path.join(args.dataset, 'meta')
    data_dir = args.dataset
    phase_str = args.phase
    out_dir = os.path.join(args.dataset, 'json')
    os.makedirs(out_dir, exist_ok=True)
    jname = phase_str+'.json'
    jdict = get_dicts(data_dir, phase_str, meta_dir)
    with open(out_dir+'/'+ jname,'w') as f:
        json.dump(jdict, f)
        print("json file: %s accomplished!"%jname)



