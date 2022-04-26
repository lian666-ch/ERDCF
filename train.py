import numpy as np
from keras.optimizers import Adam
from data_loader import DataParser
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import argparse
from loss_functions import  dice_loss,F_score
from model.ERDCF import ERDCF
import os

def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser(description='Keras Training')
    # ========= paths for training
    ap.add_argument("-mpath", "--model_path",
                    default="save_model/ERDCF/",
                    help="path to save the output model")
    ap.add_argument("-lpath", "--log_path",
                    default="save_model/ERDCF/",
                    help="path to save the 'log' files")
    ap.add_argument("-name","--model_name", default="ERDCF",
                    help="output of model name")
    # ========= parameters for training
    ap.add_argument('-bs', '--batch_size', default=4, type=int, help='batch size')
    ap.add_argument('-ep2', '--epoch', default=250, type=int, help='epoch')
    ap.add_argument('-lr', '--Initial_learning_rate', default=1e-3, help='lr')
    ap.add_argument('-minlr', '--Min_learning_rate', default=1e-8, help='minlr')
    args = vars(ap.parse_args())
    return args

def train(args):
    dataParser = DataParser(args["batch_size"])
    model = ERDCF()
    model.summary()

    lr_decay = ReduceLROnPlateau(monitor='val_ofuse_F_score', factor=0.5, patience=20, verbose=1, min_lr=args["Min_learning_rate"])
    checkpointer = ModelCheckpoint(args["model_path"] +args["model_name"]+'_ep{epoch:03d}.h5',
                                   verbose=1, save_best_only=True,monitor='val_ofuse_F_score',mode='max')
    tensorboard = TensorBoard(log_dir=args["log_path"])
    optimizer = Adam(lr=args["Initial_learning_rate"], beta_1=0.9, beta_2=0.999)
    callback_list = [lr_decay, checkpointer, tensorboard]

    model.compile(loss={'o1':dice_loss,
                        'o2': dice_loss,
                        'o3': dice_loss,
                        'ofuse': dice_loss
                        }, metrics={'ofuse': F_score}, optimizer=optimizer)
    model.fit_generator(
        generate_minibatches(dataParser, model="train"),
        steps_per_epoch=dataParser.train_steps,
        epochs=args["epoch"],
        validation_data=generate_minibatches(dataParser, model="test"),
        validation_steps=dataParser.test_steps,
        callbacks=callback_list)

def generate_minibatches(dataParser, model):
    # pdb.set_trace()
    while True:
        if model=="train":
            batch_ids = np.random.choice(dataParser.train_ids, dataParser.batch_size_train)
        elif model=="test":
            batch_ids = np.random.choice(dataParser.test_ids, dataParser.batch_size_train)
        ims, ems  = dataParser.get_batch(batch_ids)
        yield(ims, [ems, ems, ems, ems])

if __name__ == "__main__":
    args = args_parse()
    if not os.path.exists(args["model_path"]):
        os.makedirs(args["model_path"])
    train(args)