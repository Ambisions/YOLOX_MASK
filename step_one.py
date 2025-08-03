# 第一步创建项目并配置环境
# 克隆 YOLOX 仓库
!git clone https://github.com/Megvii-BaseDetection/YOLOX.git
%cd YOLOX

# 安装基本依赖（不要运行 setup.py）
!pip install -r requirements.txt
!pip install cython pycocotools

# 直接添加 YOLOX 到 Python 路径
import sys
sys.path.append("/kaggle/working/YOLOX")
