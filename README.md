# ERDCF
A keras implementation.

## 1. Running environment
Training: 
python = 3.5/3.6,
keras = 2.2.4,
tensorflow-gpu = 1.9.0,
cuda = 9.0,
cudnn = 7.6.5,
numpy = 1.18.5,
opencv-python = 4.4.0.42


## 2. Datasets
Download the public crack detection dataset is available [here](https://github.com/yhlleo/DeepCrack/tree/master/dataset/)

DeepCrack:
@article{liu2019deepcrack,
  title={DeepCrack: A deep hierarchical feature learning architecture for crack segmentation},
  author={Liu, Yahui and Yao, Jian and Lu, Xiaohu and Xie, Renping and Li, Li},
  journal={Neurocomputing},
  volume={338},
  pages={139--153},
  year={2019},
  publisher={Elsevier}
}.

We have create the train.txt and test.txt.

To create crack dataset, please follow:
1. extract DeepCrack.zip  to ./dataset/DeepCrack,

## 3. Train
Run train.py

## 4. Predict image
Run predict_img.py

You need change the path, for expamle:
model = load_model(**"./save_model/ERDCF/ERDCF_ep140.h5"** 
  , custom_objects={'dice_loss': dice_loss, 'F_score': F_score})

## 5. Eval
Run eval.py

## 6. Pretrained model

We provid a pretrained model on the public crack detection dataset. 
./pre/[ERDCF_crack.h5](https://drive.google.com/file/d/1h2F6oRANYT6vWGhS_7xCmvwSkEutms4O/view?usp=sharing)

We have uploaded the prediction in ./pre/ERDCF.zip.

***
**If you have any questions, please contact me**
