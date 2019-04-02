import os
import glob
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from data import write_data_to_file, open_data_file
from generator import get_training_and_validation_generators
from net import model_3d_2, res_next32,model_3d_1
from training import load_old_model, train_model

import tensorflow as tf
import keras.backend.tensorflow_backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
K.set_session(session)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--opt", type=str, default='Adam')
args = parser.parse_args()

config = dict()
config["image_shape"] = (128, 128, 128)  # This determines what shape the images will be cropped/resampled to.
config["batch_size"] = 8
config["patch_shape"] = None  # switch to None to train on the whole image
config["modalities"] = ["CT"]
config["nb_channels"] = len(config["modalities"])
if "patch_shape" in config and config["patch_shape"] is not None:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["patch_shape"]))
else:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))
config["validation_batch_size"] = config["batch_size"]
config["n_epochs"] = 500  # cutoff the training after this many epochs
config["patience"] = 5  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 10  # training will be stopped after this many epochs without the validation loss improving
config["initial_learning_rate"] = 0.0001
config["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
config["validation_split"] = 1  # portion of the data that will be used for training
config["validation_patch_overlap"] = 0  # if > 0, during training, validation patches will be overlapping
config["training_patch_start_offset"] = (16, 16, 16)  # randomly offset the first patch index by up to this offset
config["skip_blank"] = True  # if True, then patches without any target will be skipped
config["data_file"] = "./train_binary_128_128_128.h5"

config["model_file"] = os.path.abspath('binary_128_128_128_model.h5')
config["training_file"] = os.path.abspath("all.pkl")
config["validation_file"] = os.path.abspath("all.pkl")
config["overwrite"] = False  # If True, will previous files. If False, will use previously written files.


def main(overwrite=False):
    # convert input images into an hdf5 file
    data_file_opened = open_data_file(config["data_file"])

    model = model_3d_1(input_shape=config["input_shape"],
                       initial_learning_rate=config["initial_learning_rate"],
                       opt=args.opt
                       )
    if not overwrite and os.path.exists(config["model_file"]):
        print('load model !!')
        load_old_model(config["model_file"], model)


    # get training and testing generators
    train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_generators(
        data_file_opened,
        batch_size=config["batch_size"],
        data_split=config["validation_split"],
        overwrite=overwrite,
        validation_keys_file=config["validation_file"],
        training_keys_file=config["training_file"],
        patch_shape=config["patch_shape"],
        validation_batch_size=config["validation_batch_size"],
        validation_patch_overlap=config["validation_patch_overlap"],
        training_patch_start_offset=config["training_patch_start_offset"],
        )

    # run training
    train_model(model=model,
                model_file=config["model_file"],
                training_generator=train_generator,
                validation_generator=validation_generator,
                steps_per_epoch=n_train_steps,
                validation_steps=n_validation_steps,
                initial_learning_rate=config["initial_learning_rate"],
                learning_rate_drop=config["learning_rate_drop"],
                learning_rate_patience=config["patience"],
                early_stopping_patience=config["early_stop"],
                n_epochs=config["n_epochs"])
    data_file_opened.close()


if __name__ == "__main__":
    main(overwrite=config["overwrite"])
