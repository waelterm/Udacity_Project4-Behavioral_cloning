# **Behavioral Cloning** 

## Writeup by Philipp Waeltermann

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image2]: ./images/straight.jpg "Grayscaling"
[image3]: ./images/left.jpg "Left Image"
[image4]: ./images/right.jpg "Right Image"

[image5]: ./images/rare1.jpg "Recovery Image"
[image6]: ./images/straight1.jpg "Recovery Image"
[image7]: ./images/rare2.jpg "Recovery Image"
[image12]: ./images/normal.png "Normal Image"
[image13]: ./images/flipped.png "Flipped Image"

[image8]: ./images/challenge2_2_training.png "Training Loss"
[image9]: ./images/challenge2_2_validation.png "Validation Loss"
[image10]: ./images/correction.png "Finding Correction"
[image11]: ./images/architecture.png "Architecture"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

There are also two videos included called trackA.mp4 and trackB.mp4. These videos show the vehicle driving 
two laps on each track. 

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
The model.h5 file was used to generate the videos trackA and trackB. It is capable to drive on both tracks.
A backup model called mynet_model.h5 has also been created. This model has been trained and can drive on the first 
track only.


#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. 
The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
Before entering the main network, Keras Lambda and Cropping2D layers have been used to normalize and crop the image.

My main model consists of two convolution neural network with 5x5 filter sizes and two convolutional layers with 5x5 filters.
The depths of the layers is 12.  In between the convolutional layers, relu activation functions, max pooling, batch normalization and dropout layers are applied
Five fully connected layers follow the convolutional layers. They decrease in size from 120 to 1.
In between the fully connected layers, relu activation functions, batch normalization and dropout are applied.
For a detailed view of the layers please refer to the table in section three.

The combination of dropout layers and batch normalization have been implemented to avoid overfitting. This was necessary because the validation loss started oscillating after just one epoch.
Relu layers have been used to introduce nonlinarity. 
#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers as well as batch normalization to reduce overfitting. 
Additionally, only every 5th image from the collected data has been used. This helps prevent overfitting because
many subsequent images will be very similar with similar steering angles. By taking only every fifth image, the 
diversity in the data can be increased without having to increase the total number of images. 
This proved very effective when I quickly wanted to iterate through training the model. For thraining the final model,
I have used all of the available data though.


*Note: This is not the case for the rare cases. The datasets with rare situations have been used completely. This decision was made because these situations are already underrepresented in the general test set.*

Throughout the training, the model was validated with a different data set than the test set. And the validation loss was plotted after each training.
This allowed for quick detection of overfitting.
The model was tested by running it through the simulator and ensuring that the vehicle could stay track #1.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

The rate for the dropout layers was tested for 5%, 10%, and 20%. Because the final loss was increasing above 10%, 20% was not considered.
I have then decided that 5% will be sufficient when also adding batch normalization layers.
Batch normalization layers can help decrease the need for high dropout rates by making sure that no single value can dominate the input into the next layer.

By observing the training and validation loss, the number of epochs without was set to 12 because until then both validation and training loss decrease.
The validation and training loss of the model.h5 can be seen below:

![alt text][image8]

![alt text][image9]



#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.
I decided to only collect data driving on the center of the road. Recovery data was not collected separately because it was generated using the side cameras.
After some initial runs on the simulator, I noticed that some spots are difficult for the model. This mostly included situations that were underrepresented in the dataset.
Examples of such situations are bridges, patches without lane lines, and dirt roads.
I decided to collect some data of just these situations. By including the data in the training set, the model learned to handle those situations.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a simple LeNet network.
This was used to make sure the architecture, and the image preprocessing steps work. 
With this architecture it has been validated that the input from all three cameras could be used, and the correction factor for the steering angles had been found to be 0.04
This was done by looking at the validation loss for different correction factors. It can be seen that there is a minimum at 0.04.
These steps have been done in the import_sample_data.py script. This script later migrated to the model.py script.

![alt text][image10]

Using the LeNet architecture resulted in overfitting after just a few epochs. I decided to change the architecture.
I was inspired by Nvidia's end to end network mentioned in this paper https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
While I was inspired by that network, my network ended up being quite different. I decided that my network could be smaller because learning to drive on roads is more complex than driving on two tracks in simulation.
I ended up using less convolutional layers with less filters. I used four fully connected layers, but with fewer parameters.
Lastly, I added Dropout and Batch Normalization to avoid further overfitting. 
All of these methods enabled the network to be trained for 12 epochs without overfitting.
This number was choosen based on the loss values.

After including some more test data, this architecture worked well on Track A. However, I had to include testing data
from Track B, as well as additional data of rare situations. This caused my computer to run out of memory.
Therefore, I added a generator method to my training model. This can be seen in model.py.


#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...


My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Lambda                | Normalizes input                              |
| Cropping              | Cropping top and bottom of image              |
| Convolution 5x5     	| Depth 12                                  	|
| RELU					|												|
| Batch Normalization   | Normalizes batch to avoid overfitting         |
| Max pooling           | 2x2 stride                                    |
| Dropout               | KP: 0.95                                      |
| Convolution 5x5     	| Depth 12                                  	|
| RELU					|												|
| Batch Normalization   | Normalizes batch to avoid overfitting         |
| Max pooling           | 2x2 stride                                    |
| Dropout               | KP: 0.95                                      |
| Convolution 3x3     	| Depth 12                                  	|
| RELU					|												|
| Batch Normalization   | Normalizes batch to avoid overfitting         |
| Max pooling           | 2x2 stride                                    |
| Dropout               | KP: 0.95                                      |
| Convolution 3x3     	| Depth 12                                  	|
| RELU					|												|
| Batch Normalization   | Normalizes batch to avoid overfitting         |
| Max pooling           | 2x2 stride                                    |
| Dropout               | KP: 0.95                                      |
| Flatten       		| output 400        						    |
| Fully Connected		| output 120        							|
| RELU					|												|
| Batch Normalization   | Normalizes batch to avoid overfitting         |
| Dropout               | KP: 0.95                                      |
| Fully Connected		| output 120        						    |
| RELU					|												|
| Batch Normalization   | Normalizes batch to avoid overfitting         |
| Dropout               | KP: 0.95                                      |
| Fully Connected		| output 50        							    |
| RELU					|												|
| Batch Normalization   | Normalizes batch to avoid overfitting         |
| Dropout               | KP: 0.95                                      |
| Fully Connected		| output 25        							    |
| RELU					|												|
| Batch Normalization   | Normalizes batch to avoid overfitting         |
| Dropout               | KP: 0.95                                      |
| Fully Connected		| output 1        							    |


 

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image11]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I did not record any recovering data, but instead used images from the left and the right sidewith a correction factor.
Images from the left and right can be seen below:

![alt text][image3]
![alt text][image4]

I realized that there were some situations on the track that were rare situations. Because those situations were underrepresented in the dataset, the vehicle had a harder time driving correctly.
I decided to collect some data of just these rare situations. Examples include: bridges, dirt on the side of the road, and tight curves.
An example of this can be seen below:

![alt text][image5]

For the second track I repeated this. I first drove in the center of the road:

![alt text][image6]

Then I collected data of rare situations. Most of those were situations of tight curves or strong inclines:

![alt text][image7]






Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles. This helped to not overfit the network to one direction (the direction of more turns) and drive left and right turns the same.

![alt text][image12]
![alt text][image13]

After the collection process, I had 39743 number of data points. I used the generator function (line 54 in model.py)
to shuffle the data, calculate the corrected steering angle for the left and right cameras, load the images in each batch and transform them to RGB.
The training to validation ration was set to 80/20.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 12 as was shown in the graphs of the Section 3. Model parameter tuning.
I used an adam optimizer so that manually training the learning rate wasn't necessary.
