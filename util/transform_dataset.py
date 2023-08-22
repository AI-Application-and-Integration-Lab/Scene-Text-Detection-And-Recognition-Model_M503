from json2txt import json2txt
from make_instance import make_instance
import zipfile, os

transform_datasets = ["D501/train", "D501/val", "D501/test", "D503/train", "D501/val", "D501/test"]

def zip_dir(path):
    zf = zipfile.ZipFile('{}.zip'.format(path), 'w', zipfile.ZIP_STORED)
    
    for root, dirs, files in os.walk(path):
        for file_name in files:
            print(root, file_name)
            zf.write(os.path.join(root, file_name),os.path.join("./", file_name))
            
def main():
    for dataset in transform_datasets:
        image_folder = "./datasets/" + dataset + "/images"
        json_folder = "./datasets/" + dataset + "/labels"
        output_folder = "./datasets/" + dataset + "/txt_labels"
        output_json = "./datasets/" + dataset + "/poly.json"
        
        json2txt(json_folder, output_folder)
        make_instance(image_folder, output_folder, output_json)

        if "test" in dataset:
            zip_dir(output_folder)
    
if __name__ == "__main__":
    main()