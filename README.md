# Scene Text Detection and Recognition model(Train on real data)

## Installation
``` bash
# create conda environment
conda create -n STDR python=3.8 -y
conda activate STDR

##  Follow DPText-DETR installation
# install required packages and detectron2 
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install opencv-python scipy timm shapely albumentations Polygon3
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
pip install setuptools==59.5.0
git clone https://github.com/ymy-k/DPText-DETR.git
cd DPText-DETR
python setup.py build develop

## install CGT requirement
cd CGT
pip install -r requirements.txt

```

## Data preparation
Download the dataset, and place them into the `datasets` directory.  
We use large synthetic datasets for pre-training(D501) and human-annotated real datasets for fine-tuning(D503)
```
.
├── DPText-DETR
├── CGT
├── datasets
│   ├── D501
│   │   ├── train
│   │   │   ├── images
│   │   │   └── labels
│   │   ├── val
│   │   └── test
│   └── D503
└── ...
```

For the D501 and D503 dataset, run the following command to convert the labels to model required format:
```bash
python util/transform_dataset.py
```

## Testing

DPText-DETR: [`model_final.pth`](https://drive.google.com/file/d/1QEv__YD8wmENPUS73sCXFbctLkN8or_c/view?usp=sharing)  
CGT: [`best-check.pth`](https://drive.google.com/file/d/1_880BO0ucA_EDJAAhLJ6t83gT05zA5PQ/view?usp=sharing)  
Download the model checkpoint and place it into `model(DPText-DETR, CGT)/checkpoints` folder.

### Detection Module
Follow the section **Testing** in the DPText-DETR README.
``` bash
python DPText-DETR/tools/train_net.py --config-file ./DPText-DETR/configs/DPText_DETR/d503/R_50_poly.yaml --eval-only MODEL.WEIGHTS ./DPText-DETR/checkpoints/model_final.pth
```

### Recognition Module
Follow the section **Testing** in the CGT README.
``` bash
CUDA_VISIBLE_DEVICES=0 python CGT/main.py --config=./CGT/configs/eval.yaml --checkpoint ./CGT/checkpoints/best-check.pth --test_root /path/to/dataset --phase test --image_only
``` 
- `--test_root /path/to/dataset` set the path of evaluation dataset

## Training
### Detection Module
Follow the section **Training** in the DPText-DETR README.
1. Pre-train: Pre-train the model for D501. Please adjust the GPU number according to your situation.

``` bash
python tools/train_net.py --config-file ./DPText_DETR/configs/DPText_DETR/Pretrain_d501/R_50_poly.yaml --num-gpus 4
```
2. Fine-tune: With the pre-trained model, use the following command to fine-tune it on D503. For example:

``` bash
python tools/train_net.py --config-file ./DPText_DETR/configs/DPText_DETR/d503/R_50_poly.yaml --num-gpus 4
```

### Recognition Module
Follow the section **Training** in the CGT README.

``` bash    
CUDA_VISIBLE_DEVICES=0 python CGT/main.py --config=./CGT/configs/train.yaml
``` 
## Inference
Please prepare a font file for visualization, for instance, [`Noto Sans Traditional Chinese`](https://fonts.google.com/noto/specimen/Noto+Sans+TC) released by Google. Then run the following command.

``` bash
python predict.py --input <PATH_TO_IMG_OR_FOLDER> --result_dir <PATH_TO_SAVE_RESULT> --detec_model ./DPText-DETR/checkpoints/model_final.pth --recog_model ./CGT/checkpoints/best-check.pth  --font <PATH_TO_FONT_FILE> 
```

## Results
The following measures are expressed as percentages. We only use string categories for training and testing.

### Detection
|    Train    |  Finetune   |  Testing   |  Precision |   Recall   |  F1 score  |
|-------------|-------------|------------|------------|------------|------------|
| D501_train  | D503_train | D503 | 65.7 | 42.7 | 51.8   |

### Recognition
|    Train    |  Finetune   |  Testing   |  Accuracy  |
|-------------|-------------|------------|------------|
| CS  | D503_train | D503 | 28.1   |

## Reference

```
@inproceedings{ye2022dptext,
  title={DPText-DETR: Towards Better Scene Text Detection with Dynamic Points in Transformer},
  author={Ye, Maoyuan and Zhang, Jing and Zhao, Shanshan and Liu, Juhua and Du, Bo and Tao, Dacheng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023}
}
```
