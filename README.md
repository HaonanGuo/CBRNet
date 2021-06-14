# CBRNet
## A Coarse-to-fine Boundary Refinement Network for Building Extraction from Remote Sensing Imagery

!!!The paper is in the peer preview process.

Dataset
----
[WHU Building Dataset](https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html)  
[Massachusetts buildings dataset](https://www.kaggle.com/balraj98/massachusetts-buildings-dataset)  
[Inria aerial image dataset](https://project.inria.fr/aerialimagelabeling/)  

The code
----
### Requirements
* torch
* torchvision
* pillow
* cv2

### Usage
Clone the repository:git clone https://github.com/HaonanGuo/CBRNet.git
1. Run [s1_offset_generator.py](https://github.com/HaonanGuo/CBRNet/blob/main/s1_offset_generator.py) to generate dataset
2. Run [s2_Train_CBRNet.py](https://github.com/HaonanGuo/CBRNet/blob/main/s2_Train_CBRNet.py) to train CBR-Net
3. Run [s3_Eval_CBRNet.py](https://github.com/HaonanGuo/CBRNet/blob/main/s3_Eval_CBRNet.py) to evaluate the performance of CBR-Net


Help
----
Any question? Please contact us with: guohnwhu@163.com
