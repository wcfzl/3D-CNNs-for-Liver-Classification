import os
import SimpleITK as sitk
import numpy as np
import pandas as pd
import tables
from keras.utils.np_utils import to_categorical
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--W", type=int, default=128)
parser.add_argument("--D", type=int, default=128)
args = parser.parse_args()



def create_data_file(out_file, n_channels, n_samples, image_shape):
    hdf5_file = tables.open_file(out_file, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')
    data_shape = tuple([0, n_channels] + list(image_shape))
    truth_shape = tuple([0, 1])
    data_storage = hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape,
                                           filters=filters, expectedrows=n_samples)
    truth_storage = hdf5_file.create_earray(hdf5_file.root, 'truth', tables.UInt8Atom(), shape=truth_shape,
                                            filters=filters, expectedrows=n_samples)
    return hdf5_file, data_storage, truth_storage


filepath = "./train_binary_"+ str(args.W) + "_" + str(args.W)  + "_" + str(args.D) +"/"
csv_path = './train_label.csv'
out_file = "./train_binary_"+ str(args.W) + "_" + str(args.W)  + "_" + str(args.D)+".h5"

input_shape = (args.W, args.W, args.D)
n_samples=7424
# n_samples=4027

label_list = pd.read_csv(csv_path, index_col=0)
label_list = label_list["ret"]

filename_list = os.listdir(filepath)
filename_list = sorted(filename_list)


try:
    hdf5_file, data_storage, truth_storage = create_data_file(out_file, n_channels=1, n_samples=n_samples, image_shape=input_shape)
except Exception as e:
        # If something goes wrong, delete the incomplete data file
    os.remove(out_file)
    raise e

i = 0
for filename in filename_list:
    label_index = filename.split('.')
    label_index = label_index[0]
    label = int(label_list[label_index])
    #categorical_labels = to_categorical(label, num_classes=2)
    label = np.asarray(label)[np.newaxis][np.newaxis]

    filefullname = os.path.join(filepath, filename)
    image = sitk.ReadImage(filefullname)
    image_data1 = sitk.GetArrayFromImage(image)
    image_data1 = np.asarray(image_data1)[np.newaxis][np.newaxis]
    truth_storage.append(label)
    data_storage.append(image_data1)

    i = i + 1
    print(image_data1.shape,'  ----> ',i)
hdf5_file.close()


