# 如需转化PNG和JPG图片时
from PIL import Image
import os

src_dir = '/kaggle/input/face-mask-detection/images'
dst_dir = '/kaggle/working/VOCdevkit/VOC2007/JPEGImages'

os.makedirs(dst_dir, exist_ok=True)

for fname in os.listdir(src_dir):
    if fname.endswith('.png'):
        img = Image.open(os.path.join(src_dir, fname)).convert("RGB")
        new_name = os.path.splitext(fname)[0] + '.jpg'
        img.save(os.path.join(dst_dir, new_name))

print(" PNG 转 JPG 完成")
