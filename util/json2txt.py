import os, sys, glob, json
from path import Path
import numpy as np
from tqdm import tqdm

def json2txt(json_folder, output_folder):

    json_folder = Path(json_folder)
    output_folder = Path(output_folder)
    output_folder.makedirs_p()

    json_paths = sorted(list(glob.glob(json_folder/'*.json')))

    for json_path in tqdm(json_paths):
        f = open(json_path, 'r', encoding='utf-8')
        data = json.load(f)['shapes']
        #print(data)
        output_path = output_folder/os.path.split(json_path)[-1][:-4] + 'txt'
        out_txt = open(output_path, 'w', encoding='utf-8')
        # group_id 0:ch-str 2:en-str 3:en-ch-str 4:ch-single 
        str_group = [0,2,3,4]
        for item in data:
            if item['group_id'] in str_group and len(item['label']) > 0:
                #print('{}: {}'.format(item['label'], item['points']))
                points = np.array(item['points'], dtype=int).reshape(-1)

            for p in points:
                out_txt.write('{},'.format(p))
            safe_label = item['label'].replace(',','，')
            out_txt.write(safe_label + '\n')


