# Overview

The contents of this folder is responsible for data collection modules (```gather``` folder) and training machine learning models (```train.py```).

## Data Collection

**For details on how to add new signs, read the README file in the ```gather``` folder.**

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

After the frames are captured, all the landmark data from each frame is flattened to a 1x1890 array. The landmark data is then normalized and saved to a CSV file for model training.

## Two Hand Dynamic Signs
The two hand dynamic sign model actually consists of two one-hand dynamic sign models, one model being for the right hand and the other for the left hand. As such, training and data processing is identical to the one-hand dynamic sign model, but done twice for each hand.

## The model training process
The application uses the machine learning library [Pytorch](https://pytorch.org/). 

To achieve optimal results, the training process runs in multiple rounds for each sign language model. The program trains the models with varying numbers of layers, with varying numbers of neurons in each layer. The program then selects the combination of layers and neurons that results in the highest accuracy when the test dataset is ran through the trained model. These models are then saved in the ```trained_models``` folder for later use.

To train the models, uncomment any of these options in ```train.py```

<img width="354" alt="image" src="https://github.com/stanreee/sign-language-learning/assets/77902731/d2056567-a424-4590-b710-e7d58b5db394">

Then run ```python3 train.py```. 

Sample output:

<img width="952" alt="image" src="https://github.com/stanreee/sign-language-learning/assets/77902731/187a9df9-da7e-4fdf-a132-3978bfffaba8">

## To test out the model

Run one of the following

```python3 dynamic_cam_one_hand.py``` for one hand dynamic signs,
```python3 dynamic_cam.py``` for two hand dynamic signs,
```python3 static_cam_one_hand.py``` for one hand static signs

