# Behavioral Cloning

### Training data
- Dataset was created by driving around the track a few times while making sure the car is staying within the lanes.
- A few instances where the car recovers from almost going out of the lane are captured for better training.

##### Distribution of angle for the entire dataset
About 90% of the time the car is driving straight or nearly straight (less than 10 degrees). 

![Steering histogram](resources/steering_histogram.png "Steering histogram"")

For about 1% of the driving data a recovery maneuver is performed. This is when the car is about to go off the road and the driver sharply steers it back. This pattern is repeated a handful of times to teach the model how to recover from almost going off the road in various spots in the course.

The following example shows the car steering with 0 degrees and about to go off the road:

![Sample about to go off road](resources/sample_about_to_go_off_road.jpg "Sample about to go off road"")

And the next image shows the car back to normal (recovered) after a sharp 60 degrees steer for a couple of frames:

![Sample recovered](resources/sample_recovered.jpg "Sample recovered"")

### Processing
- Dataset is split into train (2/3), validation (2/9) and test sets (1/9)
- Generators are used for feeding data to the model
- Dataset is enhanced by adding vertically flipped images
- Images corresponding to speeds lower than 20mph are filtered out to reduce noise
- Dataset is shuffled to avoid time-dependent learning by the model

### Architecture and training
- A pre-trained VGG16 was chosen over Comma.ai and NVIDIA (better performance)
- Fixed all layers except for the last two, added 2 dropout layers for reducing overfitting, an average pooling layer, and regularized ELU dense layers. 
- Final model ran for over 50 epochs which took a couple of hours on an AWS `g2.8xlarge` instance
- Used the Adam optimizer
- Model summary:
```
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
input_4 (InputLayer)             (None, 80, 80, 3)     0                                            
____________________________________________________________________________________________________
block1_conv1 (Convolution2D)     (None, 80, 80, 64)    1792        input_4[0][0]                    
____________________________________________________________________________________________________
block1_conv2 (Convolution2D)     (None, 80, 80, 64)    36928       block1_conv1[0][0]               
____________________________________________________________________________________________________
block1_pool (MaxPooling2D)       (None, 40, 40, 64)    0           block1_conv2[0][0]               
____________________________________________________________________________________________________
block2_conv1 (Convolution2D)     (None, 40, 40, 128)   73856       block1_pool[0][0]                
____________________________________________________________________________________________________
block2_conv2 (Convolution2D)     (None, 40, 40, 128)   147584      block2_conv1[0][0]               
____________________________________________________________________________________________________
block2_pool (MaxPooling2D)       (None, 20, 20, 128)   0           block2_conv2[0][0]               
____________________________________________________________________________________________________
block3_conv1 (Convolution2D)     (None, 20, 20, 256)   295168      block2_pool[0][0]                
____________________________________________________________________________________________________
block3_conv2 (Convolution2D)     (None, 20, 20, 256)   590080      block3_conv1[0][0]               
____________________________________________________________________________________________________
block3_conv3 (Convolution2D)     (None, 20, 20, 256)   590080      block3_conv2[0][0]               
____________________________________________________________________________________________________
block3_pool (MaxPooling2D)       (None, 10, 10, 256)   0           block3_conv3[0][0]               
____________________________________________________________________________________________________
block4_conv1 (Convolution2D)     (None, 10, 10, 512)   1180160     block3_pool[0][0]                
____________________________________________________________________________________________________
block4_conv2 (Convolution2D)     (None, 10, 10, 512)   2359808     block4_conv1[0][0]               
____________________________________________________________________________________________________
block4_conv3 (Convolution2D)     (None, 10, 10, 512)   2359808     block4_conv2[0][0]               
____________________________________________________________________________________________________
block4_pool (MaxPooling2D)       (None, 5, 5, 512)     0           block4_conv3[0][0]               
____________________________________________________________________________________________________
block5_conv1 (Convolution2D)     (None, 5, 5, 512)     2359808     block4_pool[0][0]                
____________________________________________________________________________________________________
block5_conv2 (Convolution2D)     (None, 5, 5, 512)     2359808     block5_conv1[0][0]               
____________________________________________________________________________________________________
block5_conv3 (Convolution2D)     (None, 5, 5, 512)     2359808     block5_conv2[0][0]               
____________________________________________________________________________________________________
averagepooling2d_2 (AveragePooli (None, 2, 2, 512)     0           block5_conv3[0][0]               
____________________________________________________________________________________________________
dropout_56 (Dropout)             (None, 2, 2, 512)     0           averagepooling2d_2[0][0]         
____________________________________________________________________________________________________
batchnormalization_2 (BatchNorma (None, 2, 2, 512)     2048        dropout_56[0][0]                 
____________________________________________________________________________________________________
dropout_57 (Dropout)             (None, 2, 2, 512)     0           batchnormalization_2[0][0]       
____________________________________________________________________________________________________
flatten_29 (Flatten)             (None, 2048)          0           dropout_57[0][0]                 
____________________________________________________________________________________________________
dense_62 (Dense)                 (None, 4096)          8392704     flatten_29[0][0]                 
____________________________________________________________________________________________________
dropout_58 (Dropout)             (None, 4096)          0           dense_62[0][0]                   
____________________________________________________________________________________________________
dense_63 (Dense)                 (None, 2048)          8390656     dropout_58[0][0]                 
____________________________________________________________________________________________________
dense_64 (Dense)                 (None, 2048)          4196352     dense_63[0][0]                   
____________________________________________________________________________________________________
dense_65 (Dense)                 (None, 1)             2049        dense_64[0][0]                   
====================================================================================================
Total params: 35,698,497
Trainable params: 35,697,473
Non-trainable params: 1,024
____________________________________________________________________________________________________
```

### Training
```
python model.py
```

### Trained model
- [model.json](https://www.dropbox.com/s/i704xuffua2k8p6/model.json?dl=0)
- [model.h5](https://www.dropbox.com/s/j3cragmf87dl4y6/model.h5?dl=0)

### Driving
```
python drive.py model.json
```

