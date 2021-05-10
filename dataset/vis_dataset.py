import cv2
import json
from matplotlib.pyplot import *
import os
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
import argparse

def get_dicts(json_path):
    with open(json_path, 'r') as f:
        j = json.load(f)
    for i in range(len(j)):
        for k in range(len(j[i]['annotations'])):
            j[i]['annotations'][k]['segmentation']['counts'] = j[i]['annotations'][k]['segmentation']['counts'].encode()
    return j

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train')
    parser.add_argument('--name', default='01c')
    args = parser.parse_args()
    json_path = os.path.join(args.name,'json', args.phase+'.json')
    dicts = get_dicts(json_path)
    DatasetCatalog.register('screw', lambda : dicts)
    MetadataCatalog.get('screw').set(thing_classes=["screw"])
    meta = MetadataCatalog.get('screw')

    for d in dicts:
        img = cv2.imread(d['file_name'])
        visualizer = Visualizer(img[:, :, ::-1], metadata=meta)
        out = visualizer.draw_dataset_dict(d)
        figure()
        imshow(out.get_image()[:,:,::-1])
        show()
