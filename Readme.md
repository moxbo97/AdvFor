Implementation of our submited paper

 "Query-Efficient Hard-Label Attacks against Black-Box Image Forgery Localization Model via Reinforcement Learning"

 This program is stable under Python=3.7!

 We recommend that using conda before installing all the requirements. The details of our local conda environment are in:

 - environment.yaml

After your local dependencies are set the same as us, then you can run this command to setup your environment:

 - conda env create -f environment.yaml


 Directories and files included in the implementation:

 'curbest'- The well-trained AdvFor models. 

 'models'-  The code files of OSN model.

 'weights'-  The well-trained OSN model.

Please download the files above from Baidu Netdisk:

Shared file via Baidu Netdisk: AdvFor_Model
Link: https://pan.baidu.com/s/1bpZ3ivcdyjKpFD15iXUcOA?pwd=1024
Extraction code: 1024


 The commands for training and testing are:
 - Training:
 - CUDA_VISIBLE_DEVICES=0,1 python Train_osn.py
 - Testing:
 - CUDA_VISIBLE_DEVICES=0,1 python Tst.py
