import numpy as np
from keras.models import load_model, Model
from loss_functions import F_score, dice_loss
import os
import cv2
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from PIL import Image


def read_file_list(filelist):
    pfile = open(filelist)
    filenames = pfile.readlines()
    pfile.close()
    filenames = [f.strip() for f in filenames]
    return filenames

def split_pair_names(filenames, base_dir):
    filenames = [c.split(' ') for c in filenames]
    filenames = [(os.path.join(base_dir, c[0]), os.path.join(base_dir, c[1])) for c in filenames]

    return filenames

def get_batch():
    train_file = os.path.join('./dataset/DeepCrack/', 'test.txt')
    train_data_dir = './dataset/DeepCrack/'
    training_pairs = read_file_list(train_file)
    samples = split_pair_names(training_pairs, train_data_dir)

    return samples


def test():
    image_train = get_batch()
    model = load_model("./save_model/ERDCF/ERDCF_ep102_F_score0.90011.h5"
                       , custom_objects={'dice_loss': dice_loss, 'F_score': F_score})
    layer_model = Model(inputs=model.input, outputs=model.get_layer('ofuse').output)
    model.summary()

    for i in range(len(image_train)):

        im = Image.open(image_train[i][0])
        name = image_train[i][0]
        _, imgname = os.path.split(name)
        imgname, _ = os.path.splitext(imgname)

        y_train = Image.open(image_train[i][1])
        im = np.array(im, dtype=np.uint8)
        y_train = np.array(y_train, dtype=np.uint8)
        h = y_train.shape[0]
        w = y_train.shape[1]

        im1 = np.reshape(im, (1, h, w, 3))
        y_pred = layer_model.predict(im1)

        path = "predict/ERDCF/"
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(path + "gt/"):
            os.makedirs(path + "gt/")
        if not os.path.exists(path + "img/"):
            os.makedirs(path + "img/")
        if not os.path.exists(path + "crack/"):
            os.makedirs(path + "crack/")

        y_pred = y_pred.reshape((h, w))
        save_pred = y_pred * 255
        name_pre_out = path + "crack/" + imgname + '.png'
        cv2.imwrite(name_pre_out, save_pred.astype(np.uint8))

        y_train2 = y_train
        name_pre_gt = path + "gt/" + imgname + '.png'  #
        cv2.imwrite(name_pre_gt, y_train2)

        name_img = path + "img/" + imgname + '.png'  #
        cv2.imwrite(name_img, im)

if __name__ == "__main__":
    test()
