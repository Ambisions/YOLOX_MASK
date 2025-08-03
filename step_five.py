# 7. 读取、修改并保存配置文件
fixed_code = """# encoding: utf-8
import os
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.class_names = ["with_mask", "without_mask", "mask_weared_incorrect"] # 分类名
        self.num_classes = 3
        self.depth = 0.33
        self.width = 0.50
        self.warmup_epochs = 1

        # ---------- transform config ------------ #
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5

        # ---------- dataset config ------------ #
        self.data_dir = "/kaggle/working/VOCdevkit"
        self.train_ann = "trainval.txt"
        self.val_ann = "trainval.txt"  # 若无 val.txt，可复用 trainval.txt
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # ---------- training config ------------ #
        self.max_epoch = 50 # 训练50个 epoch
        self.eval_interval = 5 # 每5个epoch评估一次

    def get_dataset(self, cache: bool, cache_type: str = "ram"):
        from yolox.data import VOCDetection, TrainTransform

        return VOCDetection(
            data_dir=self.data_dir,
            image_sets=[('2007', 'trainval')],
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        from yolox.data import VOCDetection, ValTransform
        legacy = kwargs.get("legacy", False)

        return VOCDetection(
            data_dir=self.data_dir,
            image_sets=[('2007', 'trainval')],
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import VOCEvaluator

        return VOCEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed,
                                            testdev=testdev, legacy=legacy),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )
"""

with open("/kaggle/working/YOLOX/exps/example/yolox_voc/face_mask_yolox.py", "w") as f:
    f.write(fixed_code)

print("配置文件 face_mask_yolox.py 修改完成")
