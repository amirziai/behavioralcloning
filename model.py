import pandas as pd
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

from networks import vgg16, train_model, save_model, load_model
from processing import create_generators, create_flipped, pre_processor

# config
nb_epoch = 50
batch_size = 256
image_size = (80, 80, 3)
model_to_use = vgg16
base_folder = 'resources/'
driving_log = '{}driving_log.csv'.format(base_folder)
model_file = "model.json"
split_size = 1/3
min_speed_filter = 20
adam_learning_rate = 0.001
loss_function = 'mse'
columns_to_use = ['center', 'steering', 'speed']
speed_column = 'speed'


if __name__ == '__main__':
    # Processing
    data = pd.read_csv(driving_log, usecols=columns_to_use)
    data_low_speed_removed = data[data[speed_column] >= min_speed_filter]
    data_with_mirrors = create_flipped(data_low_speed_removed)
    data_shuffled = data_with_mirrors.sample(frac=1)
    train, validation_and_test = train_test_split(data_shuffled, test_size=split_size)
    validation, test = train_test_split(validation_and_test, test_size=split_size)
    size_train = len(train)
    size_validation = len(validation)
    size_test = len(test)

    # Create generators
    generator_train, generator_validation, generator_test = create_generators(train, validation, test,
                                                                              batch_size, pre_processor, image_size,
                                                                              base_folder)

    # Train and save model
    model = model_to_use(image_size)
    model.compile(loss=loss_function, optimizer=Adam(adam_learning_rate))
    history = train_model(model, generator_train, size_train, nb_epoch, generator_validation, size_validation)
    save_model(model, model_file)
    try:
        model = load_model(model_file)
        print('Model successfully saved')
    except Exception as e:
        print('ERROR loading back the model')
        print(e)
