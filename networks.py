import json

from keras.models import Model, Sequential, model_from_json
from keras.regularizers import l2
from keras.applications import VGG16
from keras.layers import AveragePooling2D, Conv2D
from keras.layers import Input, Flatten, Dense, Lambda
from keras.layers import Dropout, BatchNormalization, ELU


def train_model(model, generator_train, size_train, nb_epoch, generator_validation, size_validation):
    return model.fit_generator(generator_train, samples_per_epoch=size_train,
                               nb_epoch=nb_epoch,
                               validation_data=generator_validation, nb_val_samples=size_validation)


def load_model(model_file):
    model = model_from_json(json.load(open(model_file)))
    weight_file = model_file.replace('.json', '.h5')
    model.load_weights(weight_file)
    return model


def save_model(model, model_file):
    json.dump(model.to_json(), open(model_file, 'w'))
    weight_file = model_file.replace('.json', '.h5')
    model.save_weights(weight_file)


def comma_ai(input_shape):
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=input_shape, output_shape=input_shape))
    model.add(Conv2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Conv2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Conv2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    return model


def nvidia(input_shape):
    input_ = Input(shape=input_shape)
    x = Conv2D(24, 5, 5, border_mode="valid", subsample=(2, 2), activation="elu")(input_)
    x = Conv2D(36, 5, 5, border_mode="valid", subsample=(2, 2), activation="elu")(x)
    x = Conv2D(48, 5, 5, border_mode="valid", subsample=(2, 2), activation="elu")(x)
    x = Conv2D(64, 3, 3, border_mode="valid", subsample=(1, 1), activation="elu")(x)
    x = Conv2D(64, 3, 3, border_mode="valid", subsample=(1, 1), activation="elu")(x)
    x = Flatten()(x)
    x = Dense(1164, activation="elu")(x)
    x = Dropout(0.8)(x)
    x = Dense(100, activation="elu")(x)
    x = Dropout(0.8)(x)
    x = Dense(50, activation="elu")(x)
    x = Dropout(0.8)(x)
    x = Dense(10, activation="elu")(x)
    x = Dropout(0.8)(x)
    x = Dense(1, activation="linear")(x)
    return Model(input=input_, output=x)


def vgg16(input_shape):
    input_ = Input(shape=input_shape)
    base_model = VGG16(input_tensor=input_, include_top=False)
    for layer in base_model.layers[:-3]:
        layer.trainable = False
    x = base_model.get_layer("block5_conv3").output
    x = AveragePooling2D((2, 2))(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(4096, activation="elu", W_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation="elu", W_regularizer=l2(0.01))(x)
    x = Dense(2048, activation="elu", W_regularizer=l2(0.01))(x)
    x = Dense(1, activation="linear")(x)

    return Model(input=input_, output=x)
