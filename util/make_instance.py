import os, sys, glob, json
from path import Path
import numpy as np
import cv2 as cv
from tqdm import tqdm
from PIL import Image

def make_instance(image_folder, txt_folder, output_json):
    f = {"licenses": [], 
         "info": {},
         'images': [],
         "categories": [{"supercategory": "beverage", "id": 1, "keypoints": ["mean", "xmin", "x2", "x3", "xmax", "ymin", "y2", "y3", "ymax", "cross"], "name": "text"}],
         'annotations': []}
    
    image_folder = Path(image_folder)
    txt_folder = Path(txt_folder)
    output_json = Path(output_json)
    
    image_paths = sorted(list(glob.glob(image_folder/'*.png')))
    txt_paths = sorted(list(glob.glob(txt_folder/'*.txt')))
    
    count_annotation = 0
    
    for i, ( image_path, txt_path) in tqdm(enumerate(zip( image_paths, txt_paths))):
        
        img = cv.imread(image_path)
        f['images'].append({
            'file_name': os.path.split(image_path)[-1],
            'height': img.shape[0],
            'width':  img.shape[1],
            'id': i,
            "date_captured": "", 
            "license": 0, 
            "flickr_url": "", 
            "coco_url": ""
            })
        
        txt = open(txt_path, 'r', encoding='utf-8').readlines()
        txt = [l.rstrip('\n') for l in txt]
        for line in txt:
            splits = line.split(',')[:-1]
            
            points = np.array([int(s) for s in splits]).reshape(-1, 2)
            
            
            label = str(line.split(',')[-1])
            if len(splits) != 8:
                print("label format error ")
                print(os.path.split(image_path)[-1])
                print(label)
                
            l = []
            for c in label:
                o = ord(c)
                if o > 65536:
                    l.append(65536)
                else:
                    l.append(ord(c))
                if len(l)>=25:
                    break
            for x in range(25-len(label)):
                l.append(65536)
            
    
            f['annotations'].append({
                'iscrowd': 0,
                'category_id': 1,
                'bbox': [float(v) for v in cv.boundingRect(points)],
                'area': float(cv.contourArea(points)),
                'polys': [float(s) for s in splits],
                'image_id': i,
                'id': count_annotation,
                'rec': l,
                'iou':1.0
                })
            count_annotation += 1
    
    with open(output_json, 'w') as out:
        json.dump(f, out)
