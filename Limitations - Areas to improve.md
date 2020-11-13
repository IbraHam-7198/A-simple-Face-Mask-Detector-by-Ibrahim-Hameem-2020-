
Area to improve - 01

The algorithm does a decent job in indentifying in real time if an individual is wearing a mask or not wearing a mask. However if there are more than one individual in the frame, the algorithm focusses on only one individual and sometimes can get stuck due to sudden rapid movements between the individuals visible on the frame

Area to improve - 02

The algorithm works well when the distance between the webcam and the target individual is approximately 60 - 80 cm away in dim lighting. However in good lighting, the algorithm performs decently at a 1 m distance between the webcam and the target individual.

Using a better camera might help improve the performance 


Area to improve - 03

The algorithm does a decent job in identifying if an individual is wearing a mask or not wearing a mask. However in the scenario that an individual is wearing a mask incorreclty the performance of the algorithm can be affected, depending on the type of mask and the location of the incorrectly worn mask. If the mask is right under the nose, the algorithm will start classifying some frames as "Mask" and the others as "No Mask or Incorrectly Masked". If the mask is about 1 cm below the nose then in most cases the algorithm is able to pick up that the mask is worn incorrectly. 

This issue primarily arises from the fact that there are a very few images in the training set that are wearing the mask incorrectly and the mask lies right below the nose. Most incorrectly masked images in the training set have the mask located right above the mouth (lips) but at least 1-1.5 cms below the nose. 

Hence this issue can be fixed by populating the training set with more incorreclty masked images, where the mask lies right under the nose 
