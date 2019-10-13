# Transfer Learning
Dog Breed Classification using Machine Learning

Download data set from http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar

Use create validation script to create a separate validaton set randomly from within the images and move them to a separate validation folder. Just run it wuthout any parameters, make sure current images are extracted into "train" folder AND you are running this script from withint "train" folder.

Once validation data set is created. You can start the model to train and then predict...

python transfer_learning.py --mode train

python transfer_learning.py --mode predict --image tests/dina.jpg
