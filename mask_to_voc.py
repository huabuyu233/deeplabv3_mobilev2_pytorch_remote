import os
import shutil
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

#-------------------------------------------------------#
#   数据集路径
#-------------------------------------------------------#
source_data_path = r'E:\文件文档\数据集\train_256'
target_voc_path = r'E:\project\pycharm project\deeplabv3-plus-pytorch-main\VOCdevkit\VOC2007'

#-------------------------------------------------------#
#   想要增加测试集修改trainval_percent 
#   修改train_percent用于改变验证集的比例 9:1
#-------------------------------------------------------#
trainval_percent    = 1
train_percent       = 0.9

if __name__ == "__main__":
    random.seed(0)
    print("Creating VOC directory structure...")
    
    # 创建VOC格式目录结构
    jpegimages_path = os.path.join(target_voc_path, 'JPEGImages')
    segmentation_class_path = os.path.join(target_voc_path, 'SegmentationClass')
    imagesets_segmentation_path = os.path.join(target_voc_path, 'ImageSets', 'Segmentation')
    
    os.makedirs(jpegimages_path, exist_ok=True)
    os.makedirs(segmentation_class_path, exist_ok=True)
    os.makedirs(imagesets_segmentation_path, exist_ok=True)
    
    print("Copying images and labels...")
    
    # 复制图片和标签
    images_path = os.path.join(source_data_path, 'images')
    labels_path = os.path.join(source_data_path, 'labels')
    
    image_files = []
    for file in os.listdir(images_path):
        if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.tif'):
            image_files.append(file)
    
    for file in tqdm(image_files):
        # 复制原图
        src_image = os.path.join(images_path, file)
        dst_image = os.path.join(jpegimages_path, file)
        shutil.copy(src_image, dst_image)
        
        # 复制标签
        label_file = file.replace('.jpg', '.tif').replace('.png', '.tif').replace('.tif', '.tif')
        src_label = os.path.join(labels_path, label_file)
        dst_label = os.path.join(segmentation_class_path, label_file.replace('.tif', '.png'))
        if os.path.exists(src_label):
            shutil.copy(src_label, dst_label)
        else:
            print(f"Warning: Label file {label_file} not found.")
    
    print("Generating txt in ImageSets...")
    
    # 生成ImageSets/Segmentation目录下的txt文件
    segfilepath     = segmentation_class_path
    saveBasePath    = imagesets_segmentation_path
    
    temp_seg = os.listdir(segfilepath)
    total_seg = []
    for seg in temp_seg:
        if seg.endswith(".png"):
            total_seg.append(seg)
    
    num     = len(total_seg)  
    list    = range(num)  
    tv      = int(num*trainval_percent)  
    tr      = int(tv*train_percent)  
    trainval= random.sample(list,tv)  
    train   = random.sample(trainval,tr)  
    
    print("train and val size", tv)
    print("train size", tr)
    ftrainval   = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
    ftest       = open(os.path.join(saveBasePath,'test.txt'), 'w')  
    ftrain      = open(os.path.join(saveBasePath,'train.txt'), 'w')  
    fval        = open(os.path.join(saveBasePath,'val.txt'), 'w')  
    
    for i in list:  
        name = total_seg[i][:-4]+'\n'  
        if i in trainval:  
            ftrainval.write(name)  
            if i in train:  
                ftrain.write(name)  
            else:  
                fval.write(name)  
        else:  
            ftest.write(name)  
    
    ftrainval.close()  
    ftrain.close()  
    fval.close()  
    ftest.close()
    print("Generate txt in ImageSets done.")

    print("Checking datasets format...")
    classes_nums        = np.zeros([256], int)
    for i in tqdm(list):
        name            = total_seg[i]
        png_file_name   = os.path.join(segfilepath, name)
        if not os.path.exists(png_file_name):
            raise ValueError("未检测到标签图片%s，请查看具体路径下文件是否存在以及后缀是否为png。"%(png_file_name))
        
        png             = np.array(Image.open(png_file_name), np.uint8)
        if len(np.shape(png)) > 2:
            print("标签图片%s的shape为%s，不属于灰度图或者八位彩图，请仔细检查数据集格式。"%(name, str(np.shape(png))))
            print("标签图片需要为灰度图或者八位彩图，标签的每个像素点的值就是这个像素点所属的种类。"%(name, str(np.shape(png))))

        classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)
            
    print("打印像素点的值与数量。")
    print('-' * 37)
    print("| %15s | %15s |"%("Key", "Value"))
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print("| %15s | %15s |"%(str(i), str(classes_nums[i])))
            print('-' * 37)
    
    print("转换完成！")
    print(f"数据集已转换到: {target_voc_path}")
