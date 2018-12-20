# Team 13 ROB 535 Perception Final Project 

Throughout, we ran on Python 3.5.3 with the following modules: 
* numpy v1.15.4
* scikit-learn v0.19.2
* pandas v0.23.4
* tensorflow-gpu v1.12.0
* keras v2.2.4
* opencv-python v3.4.4.19

## Task 1
Task 1 can be run as follows: 
`python3 rob535_task1.py [train|predict]`
If the argument you pass in is `train` (or if you don't pass in anything)then the script will fine-tune a DenseNet121 model on this task and then generate labels on the test set using the resulting model. If you pass in `predict` (or anything else), the script will only produce labels for the test set. 

If you are fine-tuning the model, you need to download the pre-trained DenseNet model at https://drive.google.com/file/d/0Byy2AcGyEVxfSTA4SHJVOHNuTXc/view - the scripts expect the model file to be placed in a directory called `models`. During the training process, the script evaluates the validation accuracy after every epoch and saves the model with the best performance as `best_model_task1.h5` in the working directory. 

If you are just producing training labels, the script expects there to already exist a `best_model_task1.h5` file in your working directory. 