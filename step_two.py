# 第二步处理数据集，使其符合VOC格式
import os
import shutil

#  Step 1: 复制 JPG 图片
img_src_dir = '/kaggle/input/face-mask-detection/images'
img_dst_dir = '/kaggle/working/VOCdevkit/VOC2007/JPEGImages'

os.makedirs(img_dst_dir, exist_ok=True)

for file_name in os.listdir(img_src_dir):
    if file_name.endswith('.jpg'):
        shutil.copy(os.path.join(img_src_dir, file_name), os.path.join(img_dst_dir, file_name))

print(" 图片复制完成")

#  Step 2: 复制 XML 标注文件
anno_src_dir = '/kaggle/input/face-mask-detection/annotations'
anno_dst_dir = '/kaggle/working/VOCdevkit/VOC2007/Annotations'

os.makedirs(anno_dst_dir, exist_ok=True)

for file_name in os.listdir(anno_src_dir):
    if file_name.endswith('.xml'):
        shutil.copy(os.path.join(anno_src_dir, file_name), os.path.join(anno_dst_dir, file_name))

print(" XML 标注文件复制完成")
