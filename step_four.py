# 第四步 解决标签问题
voc_classes_path = "/kaggle/working/YOLOX/yolox/data/datasets/voc_classes.py"

fixed_content = """# encoding: utf-8
# Copyright (c) Megvii, Inc. and its affiliates.

VOC_CLASSES = (
    "with_mask",
    "without_mask",
    "mask_weared_incorrect",
)
"""

with open(voc_classes_path, "w") as f:
    f.write(fixed_content)

print("标签修复完成")
