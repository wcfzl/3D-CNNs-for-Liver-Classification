环境配置：
Ubuntu 16.04
python 3.6.4
tensorflow-gpu 1.7.0
nibable 2.3.0
nilearn 0.4.2
SimpleITK 1.1.0
scilit-image 0.13.1

1.下载并解压训练集到./train_dataset/（这里将两部份训练集合并到一起）; 下载并解压测试集到 ./test_dataset/

2.数据预处理（分别对训练集和测试集预处理，处理完成后的数据集保存在 ./train_binary_128_128_128/ 和 ./test_binary_128_128_128/ 中）
运行 crop_transform.py
其中包括：去除背板
		  裁剪黑边
		  将切片组合成3D格式（nii格式）
		  归一化数据集大小到128*128*128
		  校准
		  数据删选

3.数据打包
运行 preprocess.py将训练集打包成keras格式（train_binary_128_128_128.h5）

4.训练,分两步进行
第一步：随机初始化（正态分布），使用Adam优化器，交叉熵作为损失函数，验证精度达到0.98时停止（需要手动停止）
python train.py --opt "Adam"

第二步：前一步作为预训练，使用SGD优化器，focal_loss作为损失函数，验证精度达到0.98时停止（需要手动停止）
python train.py --opt "SGD"

5.测试,我们提供了自己训练的最好模型，直接测试可以得到线上提交结果
运行 python test.py
