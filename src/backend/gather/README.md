### This folder contains python scripts to gather landmark data manually through your webcam

### How to use

Open command line, navigate to the ```gather``` folder, then run ```python3 main.py```.

<img width="532" alt="image" src="https://github.com/stanreee/sign-language-learning/assets/77902731/bbaa3b0d-ecfb-4f51-86fa-45e6c0fbaa3b">

Follow the instructions in the command line, which will prompt you to enter the number of hands and whether the sign is static or dynamic.

<img width="821" alt="image" src="https://github.com/stanreee/sign-language-learning/assets/77902731/a541aa5a-c88e-438f-a7be-77f8fd5a5cf4">

When the capturing window opens, pressing C will start the capturing process. Capturing begins when green dots appear on the captured hand.

<img width="959" alt="Screenshot 2024-03-11 at 11 06 37 PM" src="https://github.com/stanreee/sign-language-learning/assets/77902731/1cc8b38c-3493-46ec-a87f-811b7714afe7">

### Static Sign Capturing

During the capture process, the program will gather and process the hand landmark data (as indicated by the green dots) at each frame. After 30 frames have been processed, the capturing process will end automatically. Each of the 30 frames will be used a separate data point for the corresponding sign.

### Dynamic Sign Capturing

Dynamic sign capturing is exactly the same as static sign capturing, but the compilation of all 30 frames will be processed and used as one data point for the corresponding sign.

### End Capturing

After the capturing process ends, the command line will prompt whether you would like to save the captured frames. If yes, the command line will prompt you to enter the corresponding ID for the sign.

Depending on the type of sign and the number of hands specified, the data points will be saved in ```dynamic.csv``` or ```static.csv``` for one hand signs, and ```dynamic_2.csv``` or ```static_2.csv``` for two hands, which will be used for training the machine learning models.
