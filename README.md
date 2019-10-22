![eyes](https://github.com/bartoszptak/Gaze_Attention_Website_Mapping/blob/master/README_files/head.png?raw=true)
### Author: Bartosz Ptak
If you use a dataset or model, please inform about the source.

### Install requirements

```
conda create --name gaze python=3.6.4 
conda activate gaze
```

```
conda install -c conda-forge dlib
conda install jupyter tensorflow-gpu=1.12
```

```
pip install \
  numpy==1.16 \
  requests \
  imutils \
  opencv-python==3.4.1.15 \
  pandas
```

### Downloads

Use `python get_big_files.py` to download all files  
**or**  
download directly:
  - [shape_predictor_68_face_landmarks.dat](https://drive.google.com/open?id=1TXJn_tAKkgmg9aMAVUrY8E2A9xUpoxLl)
  - [Eye dataset](https://drive.google.com/open?id=1AQ-ToGm4-PG2HlEdnvVEzX73sf-XOBL5)
  - [Train model and logs](https://drive.google.com/open?id=1DyHAYc3qOjl4odaeI9YgW82PhTV5ZVHE)
  - [Inference model](https://drive.google.com/open?id=1wPEBjl6NjpQOhh-J2ZoR3XxxUPb7do4B)

### Dataset

Dataset consists of 2670 training data and 256 test and validation data. Data structure:
- image file with resolution 120x60px:
![samle_eye](https://github.com/bartoszptak/Gaze_Attention_Website_Mapping/blob/master/README_files/eye_8.png?raw=true)
- image landmarks:
![landmarks](https://github.com/bartoszptak/Gaze_Attention_Website_Mapping/blob/master/README_files/description.png?raw=true)

```
file,L_x,L_y,R_x,R_y,CC_x,CC_y,CL_x,CL_y,CR_x,CR_y,CU_x,CU_y,CD_x,CD_y
test_imgs/eye_8.png,14,32,113,41,89,33,72,33,105,38,89,16,90,49
```

### Model
#todo
