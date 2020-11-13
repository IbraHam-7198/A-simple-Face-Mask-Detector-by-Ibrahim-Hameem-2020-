**Step 01:** Create a new virtual environment. Then install all python packages listed out in the requirement.txt file 



## **If you want to adjust the underlying model used to train the Face Mask Detector, then you will need to open the following file**

*Fine-tuned Google MobileNet V2 model for Mask or No Mask Prediction - (V1 224 x 224).ipynb* 
          
**Step 01:** Download the dataset from the link provided in the Dataset Link.md file 

**Step 02:** Open the *Fine-tuned Google MobileNet V2 model for Mask or No Mask Prediction - (V1 224 x 224).ipynb* and change the working directory to the location where the downloaded dataset was saved in your local drive *Coding line 03*

**Step 03:** Change the working directory to the location where you want to save the trained model *Coding line 26*





## **If you want to use the pre-trained model (Transfer learning), then ccarryout the following steps**


**Step 01:** Download the pre-trained model from the link provided within the file "Download the pre-trained model.md" file. 

**Step 02:** Open the *Real_time_mask_detection -  Using OpenCV-python for Face detection-Final.ipynb* file or the *Real_time_mask_detection -  Using MTCNN package for Face detection-Final.ipynb* file

**Note that the *Real_time_mask_detection -  Using MTCNN package for Face detection-Final* will display the output in a lagged manner. This is because it uses a more computationally expensive CNN (Convolutional Neural Network) to locate faces**

**Step 03:** Change the working directory to the location where you have saved the pre-trained model that was downloaded in Step 01 *Coding line 02*

