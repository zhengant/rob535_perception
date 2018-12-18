import sys

# import tensorflow as tf
import keras

from yolo_model import YOLO
from rob535_input import generate_df2, generate_xywh_task2


def create_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(64, input_dim=4, activation='elu', use_bias=True))
    model.add(keras.layers.Dense(128, activation='relu', use_bias=True))
    model.add(keras.layers.Dense(3, use_bias=True))

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    return model


def train_model(model, yolo, target_classes, train_df):
    xywh_train = generate_xywh_task2(yolo, train_df, 'trainval', target_classes)
    centroids_train = train_df.as_matrix(columns=['x', 'y', 'z'])

    early_stop = keras.callbacks.EarlyStopping(monitor='val_acc', patience=5)
    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath='best_model_task2.h5', monitor='val_acc',
                                                          save_best_only=True)
    model.fit(xywh_train, centroids_train, epochs=1000, callbacks=[early_stop, model_checkpoint],
              validation_split=0.2)

    return model


def predict_test(model, yolo, target_classes, test_df):
    xywh_test = generate_xywh_task2(yolo, test_df, 'trainval', target_classes)

    centroids = model.predict(xywh_test)
    return centroids


def output_predicted_centroids(test_df, centroids, filename):
    axes = ['x', 'y', 'z']
    with open(filename, 'w') as out:
        out.write('guid/image/axis,value\n')
        for i in range(len(test_df)):
            id = test_df['guid/code'].values[i]
            for axis in range(3):
                out.write(id + '/' + axes[axis] + ',' + str(centroids[i,axis]) + '\n')


def train_new_model(yolo, target_classes):
    model = create_model()
    train, _, test = generate_df2('trainval', 'test', 0)

    model = train_model(model, yolo, target_classes, train)
    centroids = predict_test(model, yolo, target_classes, test)

    output_predicted_centroids(test, centroids, 'task2_out.csv')


def predict_using_best(yolo, target_classes):
    model = keras.models.load_model('best_model_task2.h5')
    _, _, test = generate_df2('trainval', 'test', 0)
    centroids = predict_test(model, yolo, target_classes, test)

    output_predicted_centroids(test, centroids, 'task2_out.csv')


def main():
    yolo = YOLO(0.25, 0.5)

    target_classes = {
        1: 'bicycle',
        2: 'car',
        3: 'motorbike',
        4: 'aeroplane',
        5: 'bus',
        6: 'train',
        7: 'truck',
        9: 'boat',
    }

    if len(sys.argv) < 2 or sys.argv[1] == 'train':
        train_new_model(yolo, target_classes)
    else:
        predict_using_best(yolo, target_classes)


if __name__ == '__main__':
    main()