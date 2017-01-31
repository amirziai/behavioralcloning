import numpy as np
import pandas as pd
from functools import partial
from keras.preprocessing.image import load_img, img_to_array


def pre_processor(image, image_size, base_folder):
    height, weight, _ = image_size
    image_path = '{}{}'.format(base_folder, image)
    mirrored = False
    if image_path.endswith('_flipped'):
        image_path = image_path.replace('_flipped', '')
        mirrored = True
    image_content = load_img(image_path, target_size=(height, weight))
    image_array = img_to_array(image_content)
    if mirrored:
        image_array = image_array[:, ::-1, :]
    return image_array


def generator(data, batch_size, processor, image_size, base_folder):
    index = 0
    max_index = len(data)
    while True:
        if index >= len(data):
            index = 0  # wrap around once the end of index is reached
        indices = range(index, min(index + batch_size, max_index))  # make sure we're not going past the last index
        rows = data.iloc[indices]
        images = np.array([processor(img, image_size, base_folder) for img in rows['center']])
        targets = rows['steering'].values
        yield tuple((images, targets))
        index += batch_size


def create_flipped(dataframe):
    """Creates a new dataframe with images tagged flipped and appeneds it to the original"""
    dataframe_mirror = dataframe.copy()
    dataframe_mirror['center'] += '_flipped'
    dataframe_mirror['steering'] = -dataframe_mirror['steering'].astype(np.float32)
    return pd.concat([dataframe, dataframe_mirror], axis=0, ignore_index=True)


def create_generators(train, validation, test, batch_size, processor, image_size, base_folder):
    """Creates 3 generators for train, validation, and test"""
    generator_ = partial(generator, batch_size=batch_size, processor=processor, image_size=image_size,
                         base_folder=base_folder)
    generator_train = generator_(train)
    generator_validation = generator_(validation)
    generator_test = generator_(test)
    return generator_train, generator_validation, generator_test
