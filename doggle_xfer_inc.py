import numpy as np
import argparse
import matplotlib.pyplot as plt

from keras.models import Model
from keras.preprocessing import image
from keras.applications import imagenet_utils
from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input as incep_preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications.resnet50 import decode_predictions
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from time import time

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

num_of_classes = 120
preprocess_input = incep_preprocess_input

ap = argparse.ArgumentParser()

ap.add_argument("-m", "--mode", required=True, help="train or predict")
ap.add_argument("-i", "--image", help="path to the input image")
args = vars(ap.parse_args())

mode = args["mode"]

#tensorboard = TensorBoard(log_dir="tensor_logs/{}".format(time()))
tensorboard = TensorBoard(log_dir="tensor_logs/inception")

target_size = (299, 299)
batch_size = 32
base_model = InceptionV3(weights="imagenet", include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
#x = Dropout(0.2)(x)
output = Dense(num_of_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = False

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

train_generator = train_datagen.flow_from_directory('./train/',
                                                    target_size=target_size,
                                                    color_mode='rgb',
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True,
                                                    seed=42)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_generator = val_datagen.flow_from_directory('./validation/',
                                                target_size=target_size,
                                                color_mode="rgb",
                                                batch_size=batch_size,
                                                class_mode="categorical")

if mode == "train":

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    step_size_train = train_generator.n // train_generator.batch_size
    step_size_valid = val_generator.n // val_generator.batch_size

    model.fit_generator(generator=train_generator,
                        steps_per_epoch=step_size_train,
                        validation_data=val_generator,
                        validation_steps=step_size_valid,
                        callbacks=[tensorboard],
                        epochs=50)

    model.save_weights('doggle_xfer_inc.h5')

elif mode == "predict":

    model.load_weights('doggle_xfer_inc.h5')

    orig_img = image.load_img(args["image"], target_size=target_size)

    img = np.expand_dims(orig_img, axis=0)
    img = preprocess_input(img)

    preds = model.predict(img)

    results = []
    maxRows = 3
    classes = train_generator.class_indices

    for pred in preds:

        top_indices = pred.argsort()[-maxRows:][::-1]
        for i in top_indices:
            clsName = list(classes.keys())[list(classes.values()).index(i)]
            result = "{}: {:.2f}%".format(clsName, pred[i] * 100)
            results.append(result)

    for res in results:
        print(res)

    plt.imshow(orig_img)
    plt.suptitle("\n".join(results), fontsize=9)
    plt.show()

