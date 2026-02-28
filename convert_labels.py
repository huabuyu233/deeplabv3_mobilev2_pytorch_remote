import os
import numpy as np
from PIL import Image
import shutil

def convert_label(label_path, output_path):
    """将标签从0/255转换为0/1格式"""
    img = Image.open(label_path)
    img_np = np.array(img)
    # 将255转换为1
    img_np[img_np == 255] = 1
    # 确保其他值都是0
    img_np[img_np != 1] = 0
    img_out = Image.fromarray(img_np.astype(np.uint8))
    img_out.save(output_path)

def prepare_voc_dataset(input_dir, output_dir, train_ratio=0.8):
    """
    准备VOC格式数据集
    input_dir: 输入目录，包含image和label文件夹
    output_dir: 输出目录（VOCdevkit/VOC2007）
    """
    # 创建目录结构
    jpeg_dir = os.path.join(output_dir, 'JPEGImages')
    seg_dir = os.path.join(output_dir, 'SegmentationClass')
    sets_dir = os.path.join(output_dir, 'ImageSets', 'Segmentation')
    
    os.makedirs(jpeg_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(sets_dir, exist_ok=True)
    
    image_dir = os.path.join(input_dir, 'image')
    label_dir = os.path.join(input_dir, 'label')
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.tif', '.tiff', '.jpg', '.png'))]
    image_files.sort()
    
    # 打乱并分割数据集
    np.random.seed(42)
    np.random.shuffle(image_files)
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"总图片数: {len(image_files)}")
    print(f"训练集: {len(train_files)}")
    print(f"验证集: {len(val_files)}")
    
    # 处理训练集
    for img_file in train_files:
        base_name = os.path.splitext(img_file)[0]
        
        # 复制图片（转换为jpg）
        src_img = os.path.join(image_dir, img_file)
        dst_img = os.path.join(jpeg_dir, base_name + '.jpg')
        img = Image.open(src_img)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(dst_img, 'JPEG')
        
        # 转换并保存标签
        src_label = os.path.join(label_dir, img_file)
        dst_label = os.path.join(seg_dir, base_name + '.png')
        convert_label(src_label, dst_label)
    
    # 处理验证集
    for img_file in val_files:
        base_name = os.path.splitext(img_file)[0]
        
        # 复制图片（转换为jpg）
        src_img = os.path.join(image_dir, img_file)
        dst_img = os.path.join(jpeg_dir, base_name + '.jpg')
        img = Image.open(src_img)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(dst_img, 'JPEG')
        
        # 转换并保存标签
        src_label = os.path.join(label_dir, img_file)
        dst_label = os.path.join(seg_dir, base_name + '.png')
        convert_label(src_label, dst_label)
    
    # 生成train.txt
    with open(os.path.join(sets_dir, 'train.txt'), 'w') as f:
        for img_file in train_files:
            base_name = os.path.splitext(img_file)[0]
            f.write(base_name + '\n')
    
    # 生成val.txt
    with open(os.path.join(sets_dir, 'val.txt'), 'w') as f:
        for img_file in val_files:
            base_name = os.path.splitext(img_file)[0]
            f.write(base_name + '\n')
    
    print("数据集准备完成！")

if __name__ == "__main__":
    # 请修改这里的路径
    input_dataset_dir = r"D:\文件文档\毕业设计和论文\数据集\3. The cropped image tiles and raster labels\test"  # 你的数据集路径
    output_voc_dir = r"E:\project\deeplabv3-plus-pytorch\VOCdevkit\VOC2007"
    
    prepare_voc_dataset(input_dataset_dir, output_voc_dir, train_ratio=0.8)
