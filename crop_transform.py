# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 15:07:43 2019

@author: GKX
"""

from nilearn.image import reorder_img, new_img_like
from sitk_utils import resample_to_spacing, calculate_origin_offset
import SimpleITK as sitk
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage import measure


def save_max_objects(img):
    labels = measure.label(img)  # 返回打上标签的img数组
    jj = measure.regionprops(labels)  # 找出连通域的各种属性。  注意，这里jj找出的连通域不包括背景连通域
    # is_del = False
#    print(len(jj))
    if len(jj) == 1:
        out = img
        # is_del = False
    else:
        # 通过与质心之间的距离进行判断
        num = labels.max()  #连通域的个数
        del_array = np.array([0] * (num + 1))#生成一个与连通域个数相同的空数组来记录需要删除的区域（从0开始，所以个数要加1）
        for k in range(num):#TODO：这里如果遇到全黑的图像的话会报错
            if k == 0:
                initial_area = jj[0].area
                save_index = 1  # 初始保留第一个连通域
            else:
                k_area = jj[k].area  # 将元组转换成array

                if initial_area < k_area:
                    initial_area = k_area
                    save_index = k + 1

        del_array[save_index] = 1
        del_mask = del_array[labels]
        out = img * del_mask
        # is_del = True
    return out



def resize(image, new_shape, interpolation="linear"):
    image = reorder_img(image, resample=interpolation)
    zoom_level = np.divide(new_shape, image.shape)
    new_spacing = np.divide(image.header.get_zooms(), zoom_level)
    new_data = resample_to_spacing(image.get_data(), image.header.get_zooms(), new_spacing,
                                   interpolation=interpolation)
    new_affine = np.copy(image.affine)
    np.fill_diagonal(new_affine, new_spacing.tolist() + [1])
    new_affine[:3, 3] += calculate_origin_offset(new_spacing, image.header.get_zooms())
    return new_img_like(image, new_data, affine=new_affine)


def binaryzation(filefullname):
    ###   二值化
    image = sitk.ReadImage(filefullname)
    image_array = sitk.GetArrayFromImage(image)
    threshold_value = 0
    image_array[image_array<threshold_value] = 0
    image_array[image_array>threshold_value] = 1
    d,h,w= image_array.shape
    ###   去除小的连通区域
    for i in range(d):
        image_array[i, :, :] = save_max_objects(image_array[i, :, :])
    image_array[image_array==0] = 20000
    image_array[image_array==1] = 0
    return image_array
    


def get_bound(Arrays):
 
    H,W,L = Arrays.shape


    for i in range(0,H):
        if np.max(Arrays[i,:,:])>0:
#            print(np.max(Arrays[i,:,:]))
            H_min = i
            break
    
    for i in range(H-1,0,-1):
        if np.max(Arrays[i,:,:])>0:
#            print(np.max(Arrays[i,:,:]))
            H_max = i
            break

    for i in range(0,W):
        if np.max(Arrays[:,i,:])>0:
#            print(np.max(Arrays[:,i,:]))
            W_min = i
            break
            

    for i in range(W-1,0,-1):
        if np.max(Arrays[:,i,:])>0:
#            print(np.max(Arrays[:,i,:]))
            W_max = i
            break

    for i in range(0,L):
        if np.max(Arrays[:,:,i])>0:
#            print(np.max(Arrays[:,:,i]))
            L_min = i
            break
    
    for i in range(L-1,0,-1):
        if np.max(Arrays[:,:,i])>0:
#            print(np.max(Arrays[:,:,i]))
            L_max = i
            break
#    
    return Arrays[H_min:H_max,W_min:W_max,L_min:L_max]
    



    
def _transform(dcm_dir,save_dir):
###   from dcm to nii
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcm_dir)
    reader.SetFileNames(dicom_names)
    image2 = reader.Execute()
    
###   transform 3D image to array
    image_array = sitk.GetArrayFromImage(image2)  #  z,y,x

###   crop the dark voxel
#    new_array,range_list = get_bound(image_array)


###   transform array to 3D image
    image3 = sitk.GetImageFromArray(image_array)

###   save 3D image
    name = dcm_dir.split('/')[-1] + '.nii'
    save_path = os.path.join(save_dir,name)   #   get the save path
    sitk.WriteImage(image3,save_path)



def nib_resize1(save_dir1,name1,image_shape):

    save_path = os.path.join(save_dir1,name1)
    # load
    image1 = nib.load(save_path)
    image_data1 = image1.get_data()
    # print('before resize :',image_data1.shape)
    # resize
    image1 = resize(image1, image_shape)

    ###   transform to array
    image_data1 = image1.get_data()
    # image_data1 = image_data1[:,:,::-1]

    # print('after resize : ',image_data1.shape)
    image1 = sitk.GetImageFromArray(image_data1)
    sitk.WriteImage(image1,save_path)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--d", type=int, default=128)
parser.add_argument("--W", type=int, default=128)
args = parser.parse_args()

image_shape = (args.W,args.W,args.d)


ori_dir = './train_dataset/'
ori_nii_dir = './train_binary_'+ str(args.W) + '_'+ str(args.W)+ '_' + str(args.d) + '/'

isExists = os.path.exists(ori_nii_dir)

if not isExists:
    os.makedirs(ori_nii_dir)
    print(ori_nii_dir + ' build successfully !')
else:
    print(ori_nii_dir + ' already exist !')


if __name__ == '__main__':

    dirnum_names = os.listdir(ori_dir)
    count = 0
    for name in dirnum_names:
        dcm_dir = os.path.join(ori_dir,name)
        count+=1
        _transform(dcm_dir,ori_nii_dir) #   dicom转化为nii
        ori_nii_path = os.path.join(ori_nii_dir,name+'.nii')
        #   二值化和去除连通阈
        new_arrary = binaryzation(ori_nii_path)
        ori_image = sitk.ReadImage(ori_nii_path)
        ori_arrary = sitk.GetArrayFromImage(ori_image)
        #   映像到原图
        new_arrary1 = ori_arrary+new_arrary
        new_arrary1[new_arrary1 >10000] = 0
        crop_arrary = get_bound(new_arrary1)
        crop_image = sitk.GetImageFromArray(crop_arrary)
        sitk.WriteImage(crop_image,ori_nii_path)
        print(image_shape,'--------->>',count)
        nib_resize1(ori_nii_dir,name+'.nii',image_shape)

        
        
        
        
        
