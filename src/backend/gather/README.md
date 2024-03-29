### This folder contains python scripts to gather landmark data manually through your webcam

### IMPORTANT: Before doing any training or data collection, make sure to *pull* from Rev1_app branch, this will help prevent any potential merge conflicts

### ALSO (especially important for newly introduced IDs): Before pushing your collected data, *make sure* to train the model and test how well the model works (noting the consistency and the confidence). Details on how to train the model can be found in the parent folder ```backend```

## How to use

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

For newly added IDs (i.e. IDs that are not listed in the file src > backend > server > id_mapping.py), add them into their corresponding Python dictionaries in ```id_mapping.py```

<img width="405" alt="image" src="https://github.com/stanreee/sign-language-learning/assets/77902731/a33b3ea6-fb2a-4252-8670-daad326b15a4">

For example, ```DYNAMIC[1][0]``` corresponds to the "no" handsign, which is a dynamic sign that uses one hand.

Depending on the type of sign and the number of hands specified, the data points will be saved in ```dynamic.csv``` or ```static.csv``` for one hand signs, and ```dynamic_2.csv``` or ```static_2.csv``` for two hands, which will be used for training the machine learning models.

## Tips and tricks

### For editing .csv dataset files (to remove potential faulty data)

Good VSCode extension:

<img width="528" alt="image" src="https://github.com/stanreee/sign-language-learning/assets/77902731/84727553-b565-4693-9115-de45abeac4cc">

https://marketplace.visualstudio.com/items?itemName=janisdd.vscode-edit-csv

Can go from viewing .csv files from this

<img width="1051" alt="image" src="https://github.com/stanreee/sign-language-learning/assets/77902731/00906057-fd91-4ff0-8d0e-6ee21f33c1a3">

To this

<img width="1132" alt="image" src="https://github.com/stanreee/sign-language-learning/assets/77902731/007ec3be-2c18-4772-b38b-a6ea904f8e2d">

### Static sign capturing

When capturing static signs, during the capturing process, move your hand in different orientations to account for different angles.

### After capturing signs

After collecting landmark data for signs, train the model (instructions to train the model found in the ```backend``` folder) and then test the confidence of the newly trained signs. 
