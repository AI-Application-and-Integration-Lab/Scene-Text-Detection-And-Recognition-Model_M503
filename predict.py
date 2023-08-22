import argparse, os
import subprocess
from visualize import save_result
import  sys, glob, json
from path import Path
import numpy as np
from tqdm import tqdm
import shutil

from CGT.tools.create_lmdb_dataset import createDataset


def main(opt):
    
    subprocess.run(["python", "DPText-DETR/demo/demo.py", "--config-file", "./DPText-DETR/configs/DPText_DETR/TotalText/R_50_poly.yaml", "--input", 
                    opt.input, "--output", opt.output + "/visualize", "--opts", "MODEL.WEIGHTS", opt.detec_model])
    
    createDataset('./samples', opt.output + '/visualize/tmp_result.json', opt.output + '/LMDB')
    
    
    subprocess.run(["python", "CGT/main.py", "--config=./CGT/configs/eval.yaml", "--checkpoint", opt.recog_model, "--phase", "test", 
                    "--image_only", "--test_root", opt.output + '/LMDB'])
    
    # Visualize
    attn_folder = Path("./workdir/train-CSTC/LMDB-alignment/attn")
    gtFile = opt.output + '/visualize/tmp_result.json'
    textimg_paths = sorted(list(glob.glob(attn_folder/'*.jpg')))
    
    with open(gtFile, 'r', encoding='utf-8') as data:
        datalist = json.load(data)
    nSample = len(datalist)

    all_labels = dict()
    
    for textimg_path in textimg_paths:
        
        filename = os.path.split(textimg_path)[-1][:-4]
        names = filename.split("_")
        crop_image = names[0]
        generated_text = names[-1]
        
        
        for i in range(nSample):
            crop_image_name = os.path.split(datalist[i]['crop_image'])[-1][:-4]
            if int(crop_image) == int(crop_image_name):
                img_name = os.path.split(datalist[i]['original_image'])[-1]
                points = datalist[i]['poly']
                x_min = 1000000
                x_max = 0
                y_min = 1000000
                y_max = 0
                for point in points:
                    if x_min > point[0]: x_min = point[0]
                    if x_max < point[0]: x_max = point[0]
                    if y_min > point[1]: y_min = point[1]
                    if y_max < point[1]: y_max = point[1]
                
                label = dict(category = 1, category_id = 1, x_min = x_min, y_min = y_min, x_max = x_max,
                             y_max = y_max, det_conf = 1, text = generated_text)
                if img_name in all_labels.keys():
                    all_labels[img_name].append(label)
                else:
                    labels = list()
                    labels.append(label)
                    all_labels[img_name] = labels
                break
            
        
        
    save_result(opt, all_labels)
    
    file = Path("./tmp_crop")
    shutil.rmtree(file)
    
    file = Path("./workdir")
    shutil.rmtree(file)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input', type=str, default='./samples', help='folder of input images')  # file/folder
    parser.add_argument('--result_dir', default='./exp', help='save results to project/name')
    parser.add_argument('--output', type=str, default='./output', help='output folder')
    parser.add_argument('--font', type=str, default='./simsun.ttc', help='font path')
    parser.add_argument('--recog_model', type=str, default='./CGT/checkpoints/best-check.pth', help='Recognition model path')
    parser.add_argument('--detec_model', type=str, default='./DPText-DETR/checkpoints/model_final.pth', help='Detection model path')

    opt = parser.parse_args()

    main(opt)
