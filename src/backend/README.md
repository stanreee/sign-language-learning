# Overview

The contents of this folder is responsible for data collection modules (```gather``` folder) and training machine learning models (```train.py```).

## Data Collection

### Static Signs Data Processing

A total of 30 frames are captured, with each frame being a separate data point for the corresponding sign.

For each frame, one landmark is designated as the 'base' landmark. All other landmark coordinates are shifted relative to that base landmark. After this, the landmark data is flattened to a 1-dimensional array, and coordinates are normalized based on the largest coordinate value.

Sample landmark data for one frame:
|[0.31, 0.58, -3.16e-07]|[0.38, 0.55, -0.02]|0.43, 0.45, -0.02|...|[0.32, 0.46, -0.01]|
|---|---|---|---|---|

(3x21 array)

After processing:

|0|0|0|-0.067|0.03|0.02|-0.12|0.13|0.02|...|-0.00|0.12|0.01|
|---|---|---|---|---|---|---|---|---|---|---|---|---|

(1x63 array)

### Dynamic Signs Data Processing

Similar to static sign processing, but instead a single data point consists of the processing of all 30 frames captured. The landmark data of all frames is known as the landmark history. Each landmark data point is processed almost identically to static landmark processing, however, the first landmark of the first frame is designated as the base landmark, and all other landmarks in all other frames are shifted relative to that.

After the frames are captured, the landmark history is processed further and are sorted into an array where the ```0```th entry consists of all the x-coordinates, the ```1```st entry consists of all the y-coordinates, and the ```2```nd entry consists of all the z-coordinates.

```
[
  [x_0_0, x_0_1, x_0_2, ..., x_30_20],
  [y_0_0, y_0_1, y_0_2, ..., y_30_20],
  [z_0_0, z_0_1, z_0_2, ..., z_30_20],
]
```
Where ```x_0_0``` is the 1st frame's 1st landmark's x-coordinate, and ```x_30_20``` is the last frame's last landmark's x-coordinate.

All x, y, and z coordinates are then normalized based on the largest respective coordinate in the array. The array is then flattened to a 1x1890 array.

Landmark coordinates are calculated relative to the base landmark coordinates ```(x_0_0, y_0_0, z_0_0)```, as a result, ```(x_0_0, y_0_0, z_0_0) = (0, 0, 0)``` for all signs.

The processing is identical for two handed signs, but the array size will be (1x126) for static signs and (1x3780) for dynamic signs.

## The model training process
The application uses the machine learning library [Pytorch](https://pytorch.org/). 

To achieve optimal results, the training process runs in multiple rounds for each sign language model. The program trains the models with varying numbers of layers, with varying numbers of neurons in each layer. The program then selects the combination of layers and neurons that results in the highest accuracy when the test dataset is ran through the trained model. These models are then saved in the ```trained_models``` folder for later use.

To train the models, uncomment any of these options in ```train.py```

<img width="465" alt="image" src="https://github.com/stanreee/sign-language-learning/assets/77902731/47a2473d-ce50-42d9-9730-e4f01b2cf243">

Then run ```python3 train.py```. 

Sample output:

<img width="952" alt="image" src="https://github.com/stanreee/sign-language-learning/assets/77902731/187a9df9-da7e-4fdf-a132-3978bfffaba8">


