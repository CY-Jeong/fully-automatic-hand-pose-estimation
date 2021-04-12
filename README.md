# fully-automatic-hand-pose-estimation

Pytorch implementation of ***Fully-Automatic-Hand-Pose-Estimation*** which estimate hand pose with multi view automatically.<br>
There are so many papers to get pose annotation without manual method. Those are datasets, and we can use those datasets to raise
performance. You can see link related to make datasets in below. <br>



## Prerequisites
- Linux or macOS
- python3+

## Installation
```bash
git clone https://github.com/CY-Jeong/fully-automatic-hand-pose-estimation.git
cd fully-automatic-hand-pose-estimation
```
- Pip users, ```pip install -r requirements.txt```
- Conda users, ```conda env create -f environment.yml```

## Usage
First you need to get intrinsic, extrinsic files from **[calibration](https://github.com/CY-Jeong/camera-calibration)** which make calibration files with multiview.
If you do calibration, you can have calibration information.
And you must get **[MANO parameter pkl file](https://mano.is.tue.mpg.de/)** and **[CPM pretrained pose parameter](https://github.com/lmb-freiburg/hand3d)**.
So your final directory should be 
```
Calibration/
  result_dir/
    intrinsic.json
    extrinsic.json
fully-automatic-hand-pose-estimation/
  data/
    MANO_RIGHT.pkl
    MANO_LEFT.pkl
    posenet3d-rhd-stb-slr-finetuned.pickle
  optimize.py
```

To optimize a model
```python optimize.py```


## Results
### Segmentation with grabcut, no training with annotation
<img src='imgs/seg/0.jpg' width=150 height=100><img src='imgs/seg/1.jpg' width=150 height=100><img src='imgs/seg/2.jpg' width=150 height=100><img src='imgs/seg/3.jpg' width=150 height=100>

### Improving pose with multi view in first frame
<img src='imgs/improving/0.png' width=300 height=200>              <img src='imgs/improving/0_0.png' width=300 height=200><br>
<img src='imgs/improving/1.png' width=300 height=200>              <img src='imgs/improving/1_1.png' width=300 height=200>

### Optimizing pose with multi view segmentation after first frame
<img src='imgs/orig/0.png' width=150 height=100><img src='imgs/orig/1.png' width=150 height=100><img src='imgs/orig/2.png' width=150 height=100><img src='imgs/orig/3.png' width=150 height=100><img src='imgs/orig/4.png' width=150 height=100>

### Synthesizing background
<img src='imgs/synback/0.png' width=150 height=100><img src='imgs/synback/1.png' width=150 height=100><img src='imgs/synback/2.png' width=150 height=100>

## Related Projects
**[STB](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yujun_Cai_Weakly-supervised_3D_Hand_ECCV_2018_paper.pdf)** | **[FPHAB](https://openaccess.thecvf.com/content_cvpr_2018/papers/Garcia-Hernando_First-Person_Hand_Action_CVPR_2018_paper.pdf)** | **[Honnotatae](https://arxiv.org/pdf/1907.01481.pdf)** | **[Obman](https://ps.is.mpg.de/uploads_file/attachment/attachment/499/obman.pdf)** | **[Freihand](https://arxiv.org/pdf/1909.04349.pdf)** | **[Interhand](https://arxiv.org/pdf/2008.09309.pdf)** | **[Humbi](https://arxiv.org/pdf/1812.00281.pdf)**

# Sponser
**[ETRI](www.etri.re.kr)** (funded instruments)
