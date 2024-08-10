import tensorflow as tf
from DataGenerator import data_generator
import argparse
from model import define_model_architecture


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

    model.fit(train_data, epochs=1, validation_data=validation_data, verbose=1,
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

    train_gen_ds = train_gen_ds.skip(4000).take(1000)
    val_gen_ds = val_gen_ds.skip(800).take(200)
    return train_gen_ds, val_gen_ds


def main(args):
    output_signature = (
        tf.TensorSpec(shape=(None, None, 1, 5), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, 1, 2), dtype=tf.float32)
    )
    train_gen_ds, val_gen_ds = generate_train_and_test_data(args.train_data,
                                                            args.validation_data,
                                                            output_signature)
    model = create_and_fit_model(train_gen_ds, val_gen_ds)
    model.save(args.model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='fcn_model.keras')
    parser.add_argument('--train_data', nargs='+', required=True)
    parser.add_argument('--validation_data', nargs='+', required=True)
    args = parser.parse_args()
    main(args)