# 最后一步，下载模型训练相关依赖，启动训练
# 安装训练所缺依赖
!pip install loguru
!pip install thop

# 训练
!PYTHONPATH=/kaggle/working/YOLOX python /kaggle/working/YOLOX/tools/train.py \
-f /kaggle/working/YOLOX/exps/example/yolox_voc/face_mask_yolox.py \
-d 1 -b 8 --fp16 -o

# -f 配置路径
# -d 使用几个GPU
# -b batch_size
# --fp 半精度训练
# -o 开启省略优化器 warmup
