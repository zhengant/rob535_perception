import sys

import tensorflow as tf
import numpy as np

from densenet121_mod import DenseNet
from custom_layers import Scale
import rob535_input


def create_model(n_classes):
    model = DenseNet(classes=n_classes, reduction=0.5, dropout_rate=0.5, weights_path='models/densenet121_weights_tf.h5')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def train_model(model, train_gen, val_gen):
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=5)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='best_model.h5', monitor='val_acc',
                                                          save_best_only=True)
    model.fit_generator(train_gen, epochs=100,
                        callbacks=[early_stop, model_checkpoint], validation_data=val_gen,
                        use_multiprocessing=True, workers=4)

    return model


def predict_test(model, test_input):
    probs = model.predict_generator(test_input)
    labels = np.argmax(probs, axis=1)

    return labels


def output_predicted_labels(test_df, labels, filename):
    with open(filename, 'w') as out:
        out.write('guid/image,label\n')
        for i, label in enumerate(labels):
            id = test_df['guid/image'].values[i]
            out.write(id + ',' + str(label) + '\n')


def train_new_model():
    model = create_model(3)
    train, val, test = rob535_input.generate_df('trainval', 'test', 0.2)
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=15, width_shift_range=20,
                                                                      height_shift_range=20,
                                                                      brightness_range=[0.5, 1.5],
                                                                      shear_range=15, zoom_range=0.5,
                                                                      horizontal_flip=True)

    train_gen = rob535_input.PerceptionDataGenerator1('trainval', train, image_generator)
    val_gen = rob535_input.PerceptionDataGenerator1('trainval', val, image_generator)
    test_gen = rob535_input.PerceptionDataGenerator1('test', test, label_col=None)

    train_model(model, train_gen, val_gen)
    labels = predict_test(model, test_gen)

    output_predicted_labels(test, labels, 'task1_out.csv')


def predict_using_best():
    model = tf.keras.models.load_model('best_model.h5', custom_objects={'Scale':Scale})

    _, _, test = rob535_input.generate_df('trainval', 'test', 0)
    test_gen = rob535_input.PerceptionDataGenerator1('test', test, label_col=None)
    labels = predict_test(model, test_gen)

    output_predicted_labels(test, labels, 'task1_out.csv')


def main():
    if len(sys.argv) < 2 or sys.argv[1] == 'train':
        train_new_model()
    else:
        predict_using_best()

if __name__ == '__main__':
    main()