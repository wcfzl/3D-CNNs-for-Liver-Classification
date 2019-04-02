import os
import SimpleITK as sitk
import numpy as np
import pandas as pd
from training import load_old_model
from net import model_3d_2, res_next32,model_3d_1

model = model_3d_1(input_shape=(1, 128,128,128))
model.load_weights("./best_model/train_binary_128_128_128_model.h5")
test_path = "./test_binary_128_128_128/"
file_list = list()
result_list = list()
filename_list = sorted(os.listdir(test_path))
i = 0
for test_file in filename_list:
    i = i+1

    test_full_file = os.path.join(test_path, test_file)
    image = sitk.ReadImage(test_full_file)
    image_data = sitk.GetArrayFromImage(image)
    image_data = np.asarray(image_data)[np.newaxis][np.newaxis]
    result = model.predict(image_data)
    result = result[0, 0]
    if result >= 0.45:
        y_ = 1
    else:
        y_ = 0
    print(i, ': ', result)
    result_list.append(y_)
    file_name = test_file.split('.')[0]
    file_list.append(file_name)

data_frame = pd.DataFrame({"id": file_list, "ret": result_list})
data_frame.to_csv("submit.csv", index=0, header=1)

