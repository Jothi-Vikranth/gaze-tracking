import os
from enum import Enum
import json
import random
import cv2
import mediapipe as mp
from datetime import datetime
import numpy as np
import tensorflow as tf
from typing import Optional
from abc import ABC, abstractmethod
import logging
# from typinf import fl
import pyttsx3
from pyttsx3 import Engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from collections import defaultdict
import joblib
from sklearn.metrics import silhouette_score


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
logging.getLogger('absl').setLevel(logging.ERROR)


face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
# unique_files = ["top", "bottom", "left", "right",  "center"]


class Speak:

    # Class-level attribute to store the singleton instance
    _instance: Optional['Speak'] = None
    engine: Optional[Engine]

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Speak, cls).__new__(
                cls)  # Create the singleton instance
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'engine'):  # Initialize only once
            self.engine = pyttsx3.init()

    def speak(self, message: str, wait=False):
        if self.engine is None:
            raise ValueError("Engine is not initialized.")
        self.engine.say(message)
        self.engine.runAndWait()


class Direction:
    @abstractmethod
    def get_direction_ls(self) -> list:
        raise NotImplementedError

    @abstractmethod
    def get_sample_weight(self) -> list:
        raise NotImplementedError

    @abstractmethod
    def get_model_save_name(self):
        raise NotImplementedError

    @abstractmethod
    def process_input_data(self, data: list) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_direction(self, inp_data: list) -> str:
        raise NotImplementedError

    def get_input_shape(self, data: np.ndarray) -> int:
        return data.shape[1]


class X_Dir(Enum):
    LEFT = -1
    H_CENTER = 0
    RIGHT = 1


class Y_Dir(Enum):
    TOP = -1
    V_CENTER = 0
    BOTTOM = 1


class Vertical(Direction):

    def get_direction_ls(self):
        return [member.name for member in Y_Dir]

    def get_sample_weight(self):
        return [1, 1, 1]

    def get_model_save_name(self):
        return "vertical_model"

    def process_input_data(self, data):
        return np.array(data)[:, -1:]/100

    def get_direction(self, inp_data: list) -> str:
        model_name = self.get_model_save_name()

        loaded_model = tf.keras.models.load_model(
            f'saved_model/{model_name}.h5')

        predictions = loaded_model.predict(
            (self.process_input_data(inp_data)), verbose=0)

        prediction_classes = tf.argmax(predictions, axis=1)
        return self.get_direction_ls()[
            prediction_classes.numpy()[0]]
        # model_name = self.get_model_save_name()

        # # Load model
        # kmeans = joblib.load(f'saved_model/{model_name}.pkl')

        # # Load data from a JSON file
        # with open('mapper.json', 'r') as json_file:
        #     mapper = json.load(json_file)

        # # print(mapper)
        # # print(inp_data)
        # predictions = kmeans.predict(self.process_input_data([inp_data]))
        # # print(predictions,)
        # return self.get_direction_ls()[mapper[str(predictions[0])]]


class Horizontal(Direction):

    def get_direction_ls(self):
        return [member.name for member in X_Dir]

    def get_sample_weight(self):
        return [1, 1, 4]

    def get_model_save_name(self):
        return "horizontal_model"

    def process_input_data(self, data):
        return np.array(data)

    def get_direction(self, inp_data: list) -> str:
        model_name = self.get_model_save_name()

        loaded_model = tf.keras.models.load_model(
            f'saved_model/{model_name}.h5')

        predictions = loaded_model.predict(
            (self.process_input_data(inp_data)), verbose=0)

        prediction_classes = tf.argmax(predictions, axis=1)
        return self.get_direction_ls()[
            prediction_classes.numpy()[0]]


def get_co_ord_landmark(landmark, frame_w, frame_h):
    x = int(landmark.x * frame_w)
    y = int(landmark.y * frame_h)
    return x, y


def get_distance(point_1, point_2):
    # x1, y1 = point_1
    # x2, y2 = point_2
    # print(x1, "  ", y1)
    # print(x2, "  ", y2)
    dist = np.linalg.norm(np.array(point_1)-np.array(point_2))
    return np.round(dist, 2)


def get_points_delta(point_1, point_2):

    return point_1[0] - point_2[0], point_1[1] - point_2[1]


def show_camera_display(cam, seconds: float):

    start_time = datetime.now()
    while (datetime.now() - start_time).total_seconds() < seconds:

        _, frame = cam.read()
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('Eye Controlled Mouse', frame)

        key = cv2.waitKey(10) & 0xFF
        if key == 27:  # 27 is the ASCII code for Esc
            break

    cv2.destroyAllWindows()


def get_landmark_points(frame):

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks

    return landmark_points


def get_data(cam, data_collection_fn, file_ls: list, DATA_COLLECTION_TIME=10):

    # Speak().speak("collecting data")
    c_time = datetime.now()
    res = []
    whole_data = []

    while (datetime.now() - c_time).total_seconds() < DATA_COLLECTION_TIME:
        _, frame = cam.read()
        # frame = cv2.flip(frame, 1)
        # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # output = face_mesh.process(rgb_frame)
        # landmark_points = output.multi_face_landmarks
        landmark_points = get_landmark_points(frame)
        frame_h, frame_w, _ = frame.shape

        if landmark_points:
            # print("Collecting data")

            landmarks = landmark_points[0].landmark

            whole_data.append(collect_landmark_data_1(
                landmarks, frame_w, frame_h))
            res.append(data_collection_fn(landmarks, frame_w, frame_h))

    for file_name in file_ls:
        file_path = f"{file_name}.json"

        whole_data_folder = "whole_data"
        whole_data_file_path = os.path.join(whole_data_folder, file_path)

        with open(whole_data_file_path, 'r') as file:
            existing_data = json.load(file)
        existing_data.extend(whole_data)

        with open(whole_data_file_path, 'w') as file:
            json.dump(existing_data, file, indent=4)

        # Check if the file exists
        if os.path.exists(file_path):
            # File exists, read the existing data
            with open(file_path, 'r') as file:
                existing_data = json.load(file)

            # Append the new list to existing data
            existing_data.extend(res)

            # Write back the updated data to the file
            with open(file_path, 'w') as file:
                json.dump(existing_data, file, indent=4)
        else:
            # File doesn't exist, create it with the list of lists
            with open(file_path, 'w') as file:
                json.dump(res, file, indent=4)


def collect_landmark_data_1(landmarks, frame_w, frame_h):
    tmp = []
    for idx, landmark in enumerate(landmarks):
        x, y = get_co_ord_landmark(landmark, frame_w, frame_h)
        tmp.append([x, y])

    return tmp


def collect_landmark_data_2(landmarks, frame_w, frame_h):
    res = []

    LEFT_EYE_CENTER = 468
    RIGHT_EYE_CENTER = 473
    LEFT_HEAD_LS = [21, 162, 127, 234, 93]
    RIGHT_HEAD_LS = [251, 389, 356, 454, 323]

    eye_left = {"l": 33, "t": 159, "r": 133, "b": 145}
    eye_right = {"l": 362, "t": 386, "r": 263, "b": 374}

    diamond_left = {"l": 471, "t": 470, "r": 469, "b": 472}
    diamond_right = {"l": 476, "t": 475, "r": 474, "b": 477}

    head_mid_points = []
    for LEFT_HEAD, RIGHT_HEAD in zip(LEFT_HEAD_LS, RIGHT_HEAD_LS):

        # Left head
        l_x, l_y = get_co_ord_landmark(
            landmarks[LEFT_HEAD], frame_w, frame_h)
        # Right head
        r_x, r_y = get_co_ord_landmark(
            landmarks[RIGHT_HEAD], frame_w, frame_h)

        # head center
        head_mid_points.append((int((l_x+r_x)/2), int((l_y+r_y)/2)))

    # Left eye
    l_x, l_y = get_co_ord_landmark(
        landmarks[LEFT_EYE_CENTER], frame_w, frame_h)

    # Right eye
    r_x, r_y = get_co_ord_landmark(
        landmarks[RIGHT_EYE_CENTER], frame_w, frame_h)

    left_right_eye_center = (l_x+r_x)/2, (l_y+r_y)/2

    # Left eye ratio
    left_dist_eye = get_distance(get_co_ord_landmark(landmarks[eye_left["t"]], frame_w, frame_h), get_co_ord_landmark(
        landmarks[eye_left["b"]], frame_w, frame_h))

    left_dist_diamond = get_distance(get_co_ord_landmark(landmarks[diamond_left["t"]], frame_w, frame_h), get_co_ord_landmark(
        landmarks[diamond_left["b"]], frame_w, frame_h))

    left_eye_ratio = np.round(left_dist_eye/left_dist_diamond*100, 2)

    # Right eye ratio
    right_dist_eye = get_distance(get_co_ord_landmark(landmarks[eye_right["t"]], frame_w, frame_h), get_co_ord_landmark(
        landmarks[eye_right["b"]], frame_w, frame_h))

    right_dist_diamond = get_distance(get_co_ord_landmark(landmarks[diamond_right["t"]], frame_w, frame_h), get_co_ord_landmark(
        landmarks[diamond_right["b"]], frame_w, frame_h))

    right_eye_ratio = np.round(right_dist_eye/right_dist_diamond*100, 2)

    for head_mid_point in head_mid_points:
        res.extend(get_points_delta(head_mid_point, left_right_eye_center))

    res.append((left_eye_ratio+right_eye_ratio)/2)
    return res


def read_json_file(file_name):
    # Read the JSON file
    with open(file_name, "r") as file:
        data = json.load(file)
    return data


def train_model(direction: Direction):

    def evaluate_model(data_name, dataset):
        # Evaluate the model on the test set
        loss, accuracy = model.evaluate(dataset)
        print(f"{data_name} Loss: {loss}, {data_name} Accuracy: {accuracy}")

        predictions = model.predict(dataset)
        predicted_classes = tf.argmax(predictions, axis=1)

        y_values = tf.concat([y for _, y, _ in dataset], axis=0)

        # y_values = tf.stack([y for _, y in dataset])  # Assuming eager execution is enabled
        true_classes = tf.argmax(y_values, axis=1)
        accuracy = tf.reduce_mean(
            tf.cast(predicted_classes == true_classes, tf.float32))
        print(
            f"Custom Computed {data_name} Accuracy: {accuracy.numpy() * 100:.2f}%")

        conf_mx = confusion_matrix(true_classes, predicted_classes)
        print(conf_mx)

    unique_files = direction.get_direction_ls()
    model_save_name = direction.get_model_save_name()

    inputs = []
    outputs = []
    for idx, file_ in enumerate(unique_files):
        one_hot = np.zeros(len(unique_files), dtype=int)
        one_hot[idx] = 1
        file_data = read_json_file(f"{file_}.json")

        for face_landmark_data in file_data:
            raw_input = np.array(face_landmark_data)

            inputs.append(raw_input)
            outputs.append(one_hot)

    np_inputs = direction.process_input_data(inputs)

    print(len(inputs))
    print(np.array(inputs).shape)
    X_train, X_test, y_train, y_test = train_test_split(
        np_inputs, outputs, test_size=0.2, random_state=42)

    # Define class weights
    class_weights = direction.get_sample_weight()  # Adjust weights as needed

    # Compute sample weights based on class labels
    train_sample_weights = np.array(
        [class_weights[np.argmax(label)] for label in y_train])
    test_sample_weights = np.array(
        [class_weights[np.argmax(label)] for label in y_test])

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (X_train, y_train, train_sample_weights))
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (X_test, y_test, test_sample_weights)).batch(32).prefetch(tf.data.AUTOTUNE)

    # Apply data augmentation
    train_dataset = (
        train_dataset
        .batch(32)
        .prefetch(tf.data.AUTOTUNE)
    )

    DROPOUT = 0.4
    HIDDEN_UNITS = 10

    kernel_initializer = tf.keras.initializers.GlorotUniform()
    kernel_regularizer = tf.keras.regularizers.L2(l2=0.00001)

    # kernel_regularizer=None
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(direction.get_input_shape(np_inputs),)),
        # tf.keras.layers.Dense(HIDDEN_UNITS, activation='relu', use_bias=True,kernel_initializer=kernel_initializer,
        #                       kernel_regularizer=kernel_regularizer),  # Hidden layer
        # tf.keras.layers.Dropout(DROPOUT),  # Dropout for regularization
        tf.keras.layers.Dense(
            len(unique_files), use_bias=True, activation="softmax", kernel_initializer=kernel_initializer)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.categorical_crossentropy

    # Compile the model
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.CategoricalCrossentropy(
                      from_logits=False),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])

    # Train the model
    model.fit(train_dataset, epochs=250, validation_data=test_dataset)

    evaluate_model("test", test_dataset)
    model.save(f'saved_model/{model_save_name}.h5')
    print("model saved")


def train_vertical():
    vertical = Vertical()

    unique_files = vertical.get_direction_ls()
    model_save_name = vertical.get_model_save_name()

    inputs = []
    outputs = []
    for idx, file_ in enumerate(unique_files):
        # one_hot = np.zeros(len(unique_files), dtype=int)
        # one_hot[idx] = 1
        file_data = read_json_file(f"{file_}.json")

        for face_landmark_data in file_data:
            raw_input = np.array(face_landmark_data)

            inputs.append(raw_input)
            outputs.append(idx)

    np_inputs = vertical.process_input_data(inputs)

    X_train, X_test, y_train, y_test = train_test_split(
        np_inputs, outputs, test_size=0.2, random_state=42)

    # # Define class weights
    # class_weights = vertical.get_sample_weight()  # Adjust weights as needed

    # # Compute sample weights based on class labels
    # train_sample_weights = np.array(
    #     [class_weights[np.argmax(label)] for label in y_train])
    # test_sample_weights = np.array(
    #     [class_weights[np.argmax(label)] for label in y_test])

    # Fit the K-Means model
    kmeans = KMeans(n_clusters=len(unique_files), random_state=42)
    kmeans.fit(X_train)

    # Cluster assignments
    labels = kmeans.labels_
    # print("Cluster labels:", labels)

    # Cluster centers
    # print("Cluster centers:\n", kmeans.cluster_centers_)

    # print(confusion_matrix(y_train, labels))

    mapper = defaultdict(float)
    count = defaultdict(int)

    for predicted, true_y in zip(labels, y_train):
        mapper[int(predicted)] += true_y
        count[int(predicted)] += 1

    for _key in mapper.keys():
        mapper[_key] = int(np.round(mapper[_key] / count[_key]))

    # print("mapper = ", mapper)
    # print("count = ", count)

    predictions = kmeans.predict(X_test)
    mapped_predictions = [mapper[el] for el in predictions]

    # print(confusion_matrix(y_test, mapped_predictions))
    joblib.dump(kmeans, f'saved_model/{model_save_name}.pkl')

    # Save to a JSON file
    with open('mapper.json', 'w') as json_file:
        json.dump(mapper, json_file, indent=4)

# def train_model():

#     def evaluate_model(data_name, dataset):
#         # Evaluate the model on the test set
#         loss, accuracy = model.evaluate(dataset)
#         print(f"{data_name} Loss: {loss}, {data_name} Accuracy: {accuracy}")

#         predictions = model.predict(dataset)
#         predicted_classes = tf.argmax(predictions, axis=1)

#         y_values = tf.concat([y for _, y, _ in dataset], axis=0)

#         # y_values = tf.stack([y for _, y in dataset])  # Assuming eager execution is enabled
#         true_classes = tf.argmax(y_values, axis=1)
#         accuracy = tf.reduce_mean(
#             tf.cast(predicted_classes == true_classes, tf.float32))
#         print(
#             f"Custom Computed {data_name} Accuracy: {accuracy.numpy() * 100:.2f}%")

#         conf_mx = confusion_matrix(true_classes, predicted_classes)
#         print(conf_mx)

#     def get_sample_weights(data):

#         # Define class weights
#         class_weights = {0: 1, 1: 1, 2: 0.5}

#         # Generate sample weights based on class labels
#         sample_weights = np.array([class_weights[np.argmax(label)]
#                                    for label in y_train])
#     inputs = []
#     outputs = []
#     for idx, file_ in enumerate(unique_files):
#         one_hot = np.zeros(len(unique_files), dtype=int)
#         one_hot[idx] = 1
#         file_data = read_json_file(f"{file_}.json")

#         for face_landmark_data in file_data:
#             raw_input = np.array(face_landmark_data)

#             inputs.append(raw_input)
#             outputs.append(one_hot)

#     print(len(inputs))
#     print(np.array(inputs).shape)
#     X_train, X_test, y_train, y_test = train_test_split(
#         inputs, outputs, test_size=0.2, random_state=42)

#     # Define class weights
#     class_weights = {idx: weight for idx, weight in enumerate(
#         [1, 1, 3, 3, 5])}  # Adjust weights as needed

#     # Compute sample weights based on class labels
#     train_sample_weights = np.array(
#         [class_weights[np.argmax(label)] for label in y_train])
#     test_sample_weights = np.array(
#         [class_weights[np.argmax(label)] for label in y_test])

#     train_dataset = tf.data.Dataset.from_tensor_slices(
#         (X_train, y_train, train_sample_weights))
#     test_dataset = tf.data.Dataset.from_tensor_slices(
#         (X_test, y_test, test_sample_weights)).batch(32).prefetch(tf.data.AUTOTUNE)

#     # Apply data augmentation
#     train_dataset = (
#         train_dataset
#         .batch(32)
#         .prefetch(tf.data.AUTOTUNE)
#     )

#     DROPOUT = 0.4
#     HIDDEN_UNITS = 20

#     kernel_initializer = tf.keras.initializers.GlorotUniform()
#     kernel_regularizer = tf.keras.regularizers.L2(l2=0.00001)
#     # kernel_regularizer=None
#     model = tf.keras.Sequential([
#         tf.keras.Input(shape=(12,)),
#         # tf.keras.layers.Dense(HIDDEN_UNITS, activation='relu', kernel_initializer=kernel_initializer,
#         #                       kernel_regularizer=kernel_regularizer),  # Hidden layer
#         # tf.keras.layers.Dropout(DROPOUT),  # Dropout for regularization
#         # tf.keras.layers.Dense(HIDDEN_UNITS, activation='relu', kernel_initializer=kernel_initializer,
#         #                       kernel_regularizer=kernel_regularizer),  # Hidden layer
#         # tf.keras.layers.Dropout(DROPOUT),   # Dropout for regularization
#         # tf.keras.layers.Dense(HIDDEN_UNITS, activation='relu',kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer),  # Hidden layer
#         # tf.keras.layers.Dropout(DROPOUT),   # Dropout for regularization
#         # tf.keras.layers.Dense(64, activation='relu',kernel_initializer=kernel_initializer),  # Hidden layer
#         # tf.keras.layers.Dropout(DROPOUT),   # Dropout for regularization
#         # tf.keras.layers.Dense(20, activation='relu',kernel_initializer=kernel_initializer),  # Hidden layer
#         tf.keras.layers.Dense(5, activation="softmax")
#     ])

#     # model = tf.keras.Sequential([
#     #     tf.keras.layers.Flatten(input_shape=(478, 2)),  # Flatten the input
#     #     tf.keras.layers.Dense(64, activation='relu'),  # Hidden layer
#     #     tf.keras.layers.Dropout(DROPOUT),  # Dropout for regularization
#     #     tf.keras.layers.Dense(64, activation='relu'),  # Hidden layer
#     #     tf.keras.layers.Dropout(DROPOUT),   # Dropout for regularization
#     #     tf.keras.layers.Dense(64, activation='relu'),  # Hidden layer
#     #     tf.keras.layers.Dropout(DROPOUT),   # Dropout for regularization
#     #     tf.keras.layers.Dense(64, activation='relu'),  # Hidden layer
#     #     tf.keras.layers.Dropout(DROPOUT),   # Dropout for regularization
#     #     tf.keras.layers.Dense(20, activation='relu'),  # Hidden layer
#     #     tf.keras.layers.Dense(5, activation="linear")
#     # ])

#     optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
#     loss_fn = tf.keras.losses.categorical_crossentropy

#     # Compile the model
#     model.compile(optimizer=optimizer,
#                   loss=tf.keras.losses.CategoricalCrossentropy(
#                       from_logits=False),
#                   metrics=[tf.keras.metrics.CategoricalAccuracy()])

#     # Train the model
#     model.fit(train_dataset, epochs=500, validation_data=test_dataset)

#     # # Evaluate the model on the test set
#     # test_loss, test_accuracy = model.evaluate(test_dataset)
#     # print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

#     # # Make predictions on the test set
#     # # Predictions will be probabilities
#     # predictions = model.predict(test_dataset)
#     # # Convert probabilities to class indices
#     # predicted_classes = tf.argmax(predictions, axis=1)

#     # loss = loss_fn(
#     #     y_test, predictions
#     # )
#     # # Compare with ground truth
#     # # Convert one-hot encoding to class indices
#     # true_classes = tf.argmax(y_test, axis=1)
#     # accuracy = tf.reduce_mean(
#     #     tf.cast(predicted_classes == true_classes, tf.float32))

#     # print(f"Custom Computed Test Accuracy: {accuracy.numpy() * 100:.2f}%")
#     # conf_mx = confusion_matrix(true_classes, predicted_classes)
#     # # plt.matshow(conf_mx,	cmap=plt.cm.gray)
#     # print(conf_mx)

#     # unique, counts = np.unique(y_train, return_counts=True)
#     # class_distribution = dict(zip(unique, counts))
#     # print("Class Distribution:", class_distribution)

#     evaluate_model("test", test_dataset)
#     model.save('saved_model/my_model.h5')
#     print("model saved")


def evaluate_model(input_data):

    # def get_direction(model: Direction):

    #     model_name = model.get_model_save_name()
    #     loaded_model = tf.keras.models.load_model(
    #         f'saved_model/{model_name}.h5')

    #     predictions = loaded_model.predict(
    #         np.expand_dims(np.array(input_data), axis=0), verbose=0)

    #     prediction_classes = tf.argmax(predictions, axis=1)
    #     return model.get_direction_ls()[
    #         prediction_classes.numpy()[0]]

    vertical_direction = Vertical().get_direction(input_data)
    horizontal_direction = Horizontal().get_direction(input_data)
    # vertical_direction = get_direction(Vertical())
    # horizontal_direction = get_direction(Horizontal())

    # vertical_model_name = vertical_direction_cls.get_model_save_name()

    # vertical_loaded_model = tf.keras.models.load_model(
    #     f'saved_model/{vertical_model_name}.h5')

    # predictions = vertical_loaded_model.predict(
    #     np.expand_dims(np.array(input_data), axis=0), verbose=0)

    # prediction_classes = tf.argmax(predictions, axis=1)

    # vertical_direction = vertical_direction_cls.get_direction()[
    #     prediction_classes.numpy()[0]]
    # print(unique_files[prediction_classes.numpy()[0]],
    #       "  ", predictions)

    # print(input_data)
    # print(prediction_classes, "  ", predictions)
    # horizontal_model_name = horizontal_direction_cls.get_model_save_name()
    # horizontal_loaded_model = tf.keras.models.load_model(
    #     f'saved_model/{horizontal_model_name}.h5')

    print(f"{vertical_direction}  ,  {horizontal_direction}")

    return horizontal_direction, vertical_direction


class Camera:

    camera = None

    def __new__(cls):
        return cls

    @ classmethod
    def get_camera(cls):
        assert cls.camera is not None, "Camera is not initialized"
        return cls.camera

    @ classmethod
    def set_camera(cls):
        cls.camera = cv2.VideoCapture(0)

    @ classmethod
    def check_camera_opened(cls):
        if not cls.camera:
            return False
        return cls.camera.isOpened()

    @ classmethod
    def get_frame(cls):
        if not cls.check_camera_opened():
            cls.set_camera()
        assert cls.camera is not None
        _, frame = cls.camera.read()
        return frame
