Audio.py: Takes training data and translates the data making the sound delayed.
AudioWN.py: Takes the data and adds noise to the data
Audio_Spec.py: Takes the data and transforms it into a spectogram. Saves it as individual images.
CNN2D.py: Takes a sequential model and trains it on the com_spec images created.
CNN1D.py: Takes data created by Audio.py and AudioWN and concatenates those with the original data. A model is trained on all three of the data sets.
predict.py: Load a trained model and predict on the test data. Save the predictions in a csv file.

RUNNING CODE: 
The test and train data sets in the working directory of Audio.py, AudioWN.py, and Audio_Spec.py are required to run the python files
Audio_Spec requires a folder named 'specs' to run
Comp1 requires Audio.py and AudioWN.py to have been ran before working.
Comp1-2D.py requires com_spec.py to have been ran.
testComp.py requires a model to have been saved in the working directory.

All the python files load and/or save data. Filepaths may need to be changed for the files to run properly.
