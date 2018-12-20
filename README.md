# Team 13 ROB 535 Perception Final Project 

Throughout, we ran on Python 3.5.3 with the following modules: 
* numpy v1.15.4
* scikit-learn v0.19.2
* pandas v0.23.4
* tensorflow-gpu v1.12.0
* keras v2.2.4
* opencv-python v3.4.4.19

The final models we had for task 1 and task2 can be downloaded at the following links (need umich email to access):
* Task 1: https://drive.google.com/file/d/1Yw5loURLevxZ235ayqOcX8S8NFhA2JXO/view?usp=sharing
* Task 2: https://drive.google.com/file/d/16bUOla2ysJe1ROyw-a2pnYFXzfaw6O4Q/view?usp=sharing

Both tasks expect `trainval` and `test` directories containing the training and test data, respectively, with the organization unchanged from how it was distributed.

## Task 1
Task 1 can be run as follows: 
`python3 rob535_task1.py [train|predict]`
If the argument you pass in is `train` (or if you don't pass in anything), then the script will fine-tune a DenseNet121 model on this task and then generate labels on the test set using the resulting model. If you pass in `predict` (or anything else), the script will only produce labels for the test set. 

If you are fine-tuning the model, you need to download the pre-trained DenseNet model at https://drive.google.com/file/d/0Byy2AcGyEVxfSTA4SHJVOHNuTXc/view - the scripts expect the model file to be placed in a directory called `models`. During the training process, the script evaluates the validation accuracy after every epoch and saves the model with the best performance as `best_model_task1.h5` in the working directory. If you are just producing training labels, the script expects there to already exist a `best_model_task1.h5` file in your working directory. In both cases, the labels will be written to a csv called `task1_out.csv`. 

Sidenote: DenseNet uses a lot of memory. You may have to reduce the batch size to get it to run. 

## Task 2
Task 2 has the same interface as task 1, execpt you are running the `rob535_task2.py` script instead. Before you can run it, you will need to setup the YOLOv3 model. Download the official weights at this link: https://pjreddie.com/media/files/yolov3.weights. The repo should already have a `yolo.cfg` file at the top level. Run `python yad2k.py yolo.cfg yolov3.weights yolo.h5` to convert the offical model to a Keras one (`yolov3.weights` should be whatever filename you gave the official model you downloaded). There might be an error at the end about pydot but it is (seems) harmless. Then, the `rob535_task2.py` script will either train a new model that converts bouding boxes to centroids, saving the best one to a file called `best_model_task2.h5`, and producing centroids for the test set, or it will look for the `best_model_task2.h5` file and produce the centroids on the test set, just like in Task 1. In both cases, the outputs are written to a csv called `task2_out.csv`. 

## Credits
`densenet121_mod.py` and `custom_layers.py` are modified versions of code from https://github.com/flyyufelix/DenseNet-Keras

`yad2k.py` and `yolo.cfg` are directly from https://github.com/xiaochus/YOLOv3. `yolo_model.py` is modified from the same repo