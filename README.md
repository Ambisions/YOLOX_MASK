# YOLOX_MASK: Face Mask Detection using YOLOX on VOC Dataset
## 项目简介
  本项目基于Larxel发布的 Face Mask Detection 数据集，在 Kaggle 平台利用 GPU T4 环境成功构建并训练了一个 YOLOX 模型，实现了对人脸口罩佩戴情况的高精度检测任务。
- 数据格式：Pascal VOC（包含 PNG 格式图片与对应的 XML 标注文件）
- 检测类别：with_mask、without_mask、mask_weared_incorrect

项目成果与亮点：
- 模型训练与优化：
 - 成功构建并训练 YOLOX 模型，训练轮数 50 epoch，最终达到 mAP@0.5 = 70.17%
 - 利用 YOLOX 原生训练管线并结合自定义 VOC 数据集，实现快速收敛
 - 启用 warmup、Cosine Annealing 学习率调度、EMA 等策略提升精度稳定性
- 数据清洗与异常处理：
  - 自动识别并跳过以下异常文件：无对应XML的图像、标注框缺失或非法的XML文件、标签类别不属于预定义 VOC_CLASSES 的目标
  - 对原始VOC数据读取逻辑进行了系统性重构，实现对边界框、图像尺寸与类别标签的鲁棒解析
- 加速训练与推理：
  - 在 Kaggle 云平台双卡 T4 GPU 上完成训练，最终推理时间低于 10ms，具备实时部署潜力 

