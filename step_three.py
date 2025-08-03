# 第三步 生成训练文件trainval.txt
import os

voc_root = "/kaggle/working/VOCdevkit/VOC2007"
jpg_dir = os.path.join(voc_root, "JPEGImages")
xml_dir = os.path.join(voc_root, "Annotations")
main_dir = os.path.join(voc_root, "ImageSets", "Main")
os.makedirs(main_dir, exist_ok=True)

# 获取所有 XML 文件对应的 ID（不带扩展名）
xml_ids = set(os.path.splitext(f)[0] for f in os.listdir(xml_dir) if f.endswith(".xml"))

# 获取所有 .jpg 文件中，存在 XML 的
valid_ids = [os.path.splitext(f)[0] for f in os.listdir(jpg_dir) if f.endswith(".jpg") and os.path.splitext(f)[0] in xml_ids]
valid_ids.sort()

# 生成 trainval.txt
with open(os.path.join(main_dir, 'trainval.txt'), "w") as f:
    for image_id in valid_ids:
        f.write(f"{image_id}\n")

print(f" 共生成 {len(valid_ids)} 条记录到 trainval.txt（全部有 XML）")
