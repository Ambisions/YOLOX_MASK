# 第六步，读取、修改并保存voc.py文件内容：添加跳过空target功能，自适应模型评估流程
fixed_voc_code = '''
import os
import os.path
import pickle
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from yolox.evaluators.voc_eval import voc_eval
from .datasets_wrapper import CacheDataset, cache_read_img
from .voc_classes import VOC_CLASSES


class AnnotationTransform(object):
    def __init__(self, class_to_ind=None, keep_difficult=True):
        self.class_to_ind = class_to_ind or dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        res = np.empty((0, 5))
        for obj in target.iter("object"):
            difficult = obj.find("difficult")
            if difficult is not None:
                difficult = int(difficult.text) == 1
            else:
                difficult = False
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.strip()
            if name not in self.class_to_ind:
                print(f"[警告] 未知标签: {name}，跳过该目标")
                continue
            bbox = obj.find("bndbox")
            pts = ["xmin", "ymin", "xmax", "ymax"]
            bndbox = []
            for pt in pts:
                try:
                    cur_pt = int(float(bbox.find(pt).text)) - 1
                except Exception as e:
                    print(f"[错误] 边界框解析失败: {e}")
                    continue
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res = np.vstack((res, bndbox))

        width = int(target.find("size").find("width").text)
        height = int(target.find("size").find("height").text)
        img_info = (height, width)

        return res, img_info


class VOCDetection(CacheDataset):
    def __init__(
        self,
        data_dir,
        image_sets=[("2007", "trainval"), ("2012", "trainval")],
        img_size=(416, 416),
        preproc=None,
        target_transform=AnnotationTransform(),
        dataset_name="VOC0712",
        cache=False,
        cache_type="ram",
    ):
        self.root = data_dir
        self.image_set = image_sets
        self.img_size = img_size
        self.preproc = preproc
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = os.path.join("%s", "Annotations", "%s.xml")
        self._imgpath = os.path.join("%s", "JPEGImages", "%s.jpg")
        self._classes = VOC_CLASSES
        self.cats = [{"id": idx, "name": val} for idx, val in enumerate(VOC_CLASSES)]
        self.class_ids = list(range(len(VOC_CLASSES)))
        self.ids = []
        for (year, name) in image_sets:
            self._year = year
            rootpath = os.path.join(self.root, "VOC" + year)
            with open(os.path.join(rootpath, "ImageSets", "Main", name + ".txt")) as f:
                for line in f:
                    self.ids.append((rootpath, line.strip()))
        self.num_imgs = len(self.ids)

        self.annotations = self._load_coco_annotations()

        path_filename = [
            (self._imgpath % self.ids[i]).split(self.root + "/")[1]
            for i in range(self.num_imgs)
        ]
        super().__init__(
            input_dimension=img_size,
            num_imgs=self.num_imgs,
            data_dir=self.root,
            cache_dir_name=f"cache_{self.name}",
            path_filename=path_filename,
            cache=cache,
            cache_type=cache_type
        )

    def __len__(self):
        return self.num_imgs

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(i) for i in range(self.num_imgs)]

    def load_anno_from_ids(self, index):
        img_id = self.ids[index]
        try:
            target_xml = ET.parse(self._annopath % img_id).getroot()
            res, img_info = self.target_transform(target_xml)
        except FileNotFoundError:
            print(f"[缺失 XML 文件] {self._annopath % img_id}")
            res = np.empty((0, 5))
            img_info = (416, 416)  # 默认大小
        if res.shape[0] == 0:
            print(f"[空标注] 跳过图像: {img_id[1]}")
        height, width = img_info
        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r
        resized_info = (int(height * r), int(width * r))
        return (res, img_info, resized_info)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img, (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        img_id = self.ids[index]
        img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
        assert img is not None, f"file named {self._imgpath % img_id} not found"
        return img

    @cache_read_img(use_cache=True)
    def read_img(self, index):
        return self.load_resized_img(index)

    def pull_item(self, index):
        target, img_info, _ = self.annotations[index]
        if target.shape[0] == 0:
            return self.__getitem__((index + 1) % self.num_imgs)
        img = self.read_img(index)
        return img, target, img_info, index

    @CacheDataset.mosaic_getitem
    def __getitem__(self, index):
        img, target, img_info, img_id = self.pull_item(index)
        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, img_id
        
    def evaluate_detections(self, all_boxes, output_dir=None):
        self._write_voc_results_file(all_boxes)
        IouTh = np.linspace(0.5, 0.95, 10, endpoint=True)
        mAPs = []
        for iou in IouTh:
            mAP = self._do_python_eval(output_dir, iou)
            mAPs.append(mAP)

        print("--------------------------------------------------------------")
        print("map_5095:", np.mean(mAPs))
        print("map_50:", mAPs[0])
        print("--------------------------------------------------------------")
        return np.mean(mAPs), mAPs[0]

    def _get_voc_results_file_template(self):
        filename = "comp4_det_test" + "_{:s}.txt"
        filedir = os.path.join(self.root, "results", "VOC" + self._year, "Main")
        os.makedirs(filedir, exist_ok=True)
        return os.path.join(filedir, filename)

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(VOC_CLASSES):
            if cls == "__background__":
                continue
            print(f"Writing {cls} VOC results file")
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, "wt") as f:
                for im_ind, index in enumerate(self.ids):
                    index = index[1]
                    dets = all_boxes[cls_ind][im_ind]
                    if len(dets) == 0:
                        continue
                    for k in range(dets.shape[0]):
                        f.write(
                            f"{index} {dets[k, -1]:.3f} {dets[k, 0]+1:.1f} {dets[k, 1]+1:.1f} {dets[k, 2]+1:.1f} {dets[k, 3]+1:.1f}\\n"
                        )
    


    def _do_python_eval(self, output_dir="output", iou=0.5):
        rootpath = os.path.join(self.root, "VOC" + self._year)
        name = self.image_set[0][1]
        annopath = os.path.join(rootpath, "Annotations", "{:s}.xml")
        imagesetfile = os.path.join(rootpath, "ImageSets", "Main", name + ".txt")
        cachedir = os.path.join(self.root, "annotations_cache", "VOC" + self._year, name)
        os.makedirs(cachedir, exist_ok=True)

        aps = []
        use_07_metric = True if int(self._year) < 2010 else False
        print(f"Eval IoU : {iou:.2f}")
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        for i, cls in enumerate(VOC_CLASSES):
            if cls == "__background__":
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir,
                ovthresh=iou, use_07_metric=use_07_metric,
            )
            aps.append(ap)
            if iou == 0.5:
                print(f"AP for {cls} = {ap:.4f}")
                with open(os.path.join(output_dir, cls + "_pr.pkl"), "wb") as f:
                    pickle.dump({"rec": rec, "prec": prec, "ap": ap}, f)

        if iou == 0.5:
            print("Mean AP = {:.4f}".format(np.mean(aps)))
            print("~~~~~~~~")
            print("Results:")
            for ap in aps:
                print("{:.3f}".format(ap))
            print("{:.3f}".format(np.mean(aps)))
            print("~~~~~~~~")
            print("")
            print("--------------------------------------------------------------")
            print("Results computed with the **unofficial** Python eval code.")
            print("Recompute with `./tools/reval.py --matlab ...` for your paper.")
            print("-- Thanks, The Management")
            print("--------------------------------------------------------------")
        return np.mean(aps)



'''

# 保存覆盖文件
with open("/kaggle/working/YOLOX/yolox/data/datasets/voc.py", "w") as f:
    f.write(fixed_voc_code)

print("voc.py 文件已成功更新（跳过空标注图像）")
