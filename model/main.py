import tensorflow as tf
from DataGenerator import data_generator
import argparse
from model import define_model_architecture
import fsspec
import s3fs
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split


def get_files(start='',end=''):
    fs = fsspec.filesystem('s3', anon=True)
    s3 = s3fs.S3FileSystem(anon=False)
    all_files = []
    years = range(int(start[-2:]),int(end[-2:]))

    for yr in years:
        file_paths_p = f"s3://maap-ops-workspace/shared/jiannamar07/wildfire_ids_{yr}/*_perimeter.nc"
        print(file_paths_p)
        file_paths = s3.glob(file_paths_p)
        all_fire_ids = []

        for file in file_paths:
            fire_id = os.path.basename(file).split('_')[0]
            all_fire_ids.append(fire_id)

        for fire in all_fire_ids:
            file_path = f"s3://maap-ops-workspace/shared/jiannamar07/wildfire_ids_{yr}/{fire}_all.nc"
            all_files.append(file_path)

    return all_files


def create_and_fit_model(train_data, validation_data):
    model = define_model_architecture(input_shape=(None, None, 1, 5))
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=7,
        verbose=1,
        mode='auto',
        restore_best_weights=True
    )

    model.fit(train_data, epochs=10, validation_data=validation_data, verbose=1,
              callbacks=[early_stopping])
    return model


def generate_train_and_test_data(train_data, validation_data, output_signature):
    batch_size = 1
    training_generator = lambda: data_generator(train_data, 1, 2)
    validation_generator = lambda: data_generator(validation_data, 1, 2)

    train_gen_ds = tf.data.Dataset.from_generator(
        training_generator,
        output_signature=output_signature
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE).repeat()
    
    val_gen_ds = tf.data.Dataset.from_generator(
        validation_generator,
        output_signature=output_signature
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE).repeat()

    # train_gen_ds = train_gen_ds.skip(4000).take(1000)
    # val_gen_ds = val_gen_ds.skip(800).take(200)
    return train_gen_ds, val_gen_ds


def main(args):
    output_signature = (
        tf.TensorSpec(shape=(None, None, 1, 5), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, 1, 2), dtype=tf.float32)
    )
    all_files = get_files(args.start_year, args.end_year)
    
    np.random.seed(42)
    random.seed(42)
 
    training_files = all_files[0:int(0.75 * len(all_files))]
    testing_files = all_files[int(0.75 * len(all_files)):]
 
    train_data, validation_data = train_test_split(all_files, test_size = 0.2, random_state=42)
 

    train_gen_ds, val_gen_ds = generate_train_and_test_data(train_data,
                                                            validation_data,
                                                            output_signature)
    train_gen_ds = train_gen_ds.take(4)
    val_gen_ds = val_gen_ds.take(1)
    
    model = create_and_fit_model(train_gen_ds, val_gen_ds)
    model.save(args.model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='model_v1.keras')
    parser.add_argument('--start_year', type=str, required=True)
    parser.add_argument('--end_year', type=str, required=True)
    args = parser.parse_args()
    main(args)
