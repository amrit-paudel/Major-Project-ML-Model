

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score

# from keras.models import load_model
import matplotlib.pyplot as plt
# from softmax import SoftMax
import numpy as np
import argparse
import pickle

# Construct the argumet parser and parse the argument
from src.detectfaces_mtcnn.Configurations import get_logger
from src.training.softmax import SoftMax

from keras import backend as K

from keras.callbacks import Callback



class LearningRateHistory(Callback):
    def on_train_begin(self, logs={}):
        self.lr = []

    def on_epoch_end(self, batch, logs={}):
        self.lr.append(K.get_value(self.model.optimizer.lr))


class TrainFaceRecogModel:

    def __init__(self, args, graph, session):

        self.args = args
        self.graph = graph
        self.session = session
        self.logger = get_logger()
        # Load the face embeddings
        # self.data = pickle.loads(open(args["embeddings"], "rb").read())

        # Load the face embeddings
        self.data = pickle.loads(open(args.embeddings, "rb").read())
        print("EMBEDDINGS DATA", self.data)

    def trainKerasModelForFaceRecognition(self):
        # Encode the labels
        le = LabelEncoder()
        labels = le.fit_transform(self.data["names"])
        print("labels", labels)
        num_classes = len(np.unique(labels))
        labels = labels.reshape(-1, 1)

        one_hot_encoder = OneHotEncoder(categorical_features = [0])
        # one_hot_encoder = OneHotEncoder()
        labels = one_hot_encoder.fit_transform(labels).toarray()

        embeddings = np.array(self.data["embeddings"])

        # Initialize Softmax training model arguments
        # BATCH_SIZE = 8
        BATCH_SIZE = 16
        EPOCHS = 50
        input_shape = embeddings.shape[1]

        with self.graph.as_default():
            K.set_session(self.session)

            # Build sofmax classifier
            softmax = SoftMax(input_shape=(input_shape,), num_classes=num_classes)
            # model = softmax.build()
            model, lr_scheduler, early_stopping = softmax.build()

            # Create KFold
            cv = KFold(n_splits = 5, random_state = 42, shuffle=True)
            history = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': [], 'lr': []}

            # Create an instance of the callback
            lr_history = LearningRateHistory()

            # Train
            for train_idx, valid_idx in cv.split(embeddings):
                X_train, X_val, y_train, y_val = embeddings[train_idx], embeddings[valid_idx], labels[train_idx], labels[valid_idx]
                his = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(X_val, y_val), callbacks=[lr_scheduler, early_stopping, lr_history])
                print(his.history['acc'])

                # ADDED PERFORMANCE MATRIX
                # ADDED PERFORMANCE MATRIX
                # Get your model's predictions (y_pred) on the validation data
                y_pred = model.predict(X_val)

                # Convert the predictions and the labels from one-hot encoding to labels
                y_pred = np.argmax(y_pred, axis=1)
                y_val = np.argmax(y_val, axis=1)

                # Calculate the metrics
                precision = precision_score(y_val, y_pred, average='macro')
                recall = recall_score(y_val, y_pred, average='macro')
                f1 = f1_score(y_val, y_pred, average='macro')

                print(f'Precision: {precision}')
                print(f'Recall: {recall}')
                print(f'F1 Score: {f1}')
                # ADDED PERFORMANCE MATRIX
                # ADDED PERFORMANCE MATRIX


                history['acc'] += his.history['acc']
                history['val_acc'] += his.history['val_acc']
                history['loss'] += his.history['loss']
                history['val_loss'] += his.history['val_loss']
                history['lr'] += lr_history.lr  # Add the learning rate history here

                self.logger.info(his.history['acc'])

        # write the face recognition model to output
        model.save(self.args.training_model)


        print("SAVING TRAINING HISTORY")
        # Save the training history
        with open(self.args.training_model + '_history.pkl', 'wb') as f:
            pickle.dump(history, f)
            print("TRAINING HISTORY SAVED")



        f = open(self.args.le, "wb")
        f.write(pickle.dumps(le))
        f.close()

    #
    # def trainKerasModelForFaceRecognition(self):
    #     # Encode the labels
    #     le = LabelEncoder()
    #     labels = le.fit_transform(self.data["names"])
    #     num_classes = len(np.unique(labels))
    #     labels = labels.reshape(-1, 1)
    #
    #     # ADDED JUST RECENTLY
    #     # one_hot_encoder = OneHotEncoder(sparse=False)  # Ensure the output is a dense array
    #     # labels = one_hot_encoder.fit_transform(labels)  # One-hot encode the labels
    #
    #     one_hot_encoder = OneHotEncoder(categorical_features = [0])
    #     labels = one_hot_encoder.fit_transform(labels).toarray()
    #
    #     embeddings = np.array(self.data["embeddings"])
    #
    #     # Initialize Softmax training model arguments
    #     BATCH_SIZE = 8
    #     EPOCHS = 5
    #     input_shape = embeddings.shape[1]
    #
    #
    #     # Build sofmax classifier
    #     softmax = SoftMax(input_shape=(input_shape,), num_classes=num_classes)
    #     model = softmax.build()
    #
    #     # Create KFold
    #     cv = KFold(n_splits = 5, random_state = 42, shuffle=True)
    #     history = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []}
    #
    #     # Train
    #     for train_idx, valid_idx in cv.split(embeddings):
    #         X_train, X_val, y_train, y_val = embeddings[train_idx], embeddings[valid_idx], labels[train_idx], labels[valid_idx]
    #         his = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(X_val, y_val))
    #         print(his.history['acc'])
    #
    #         history['acc'] += his.history['acc']
    #         history['val_acc'] += his.history['val_acc']
    #         history['loss'] += his.history['loss']
    #         history['val_loss'] += his.history['val_loss']
    #
    #         self.logger.info(his.history['acc'])
    #
    #     # write the face recognition model to output
    #     # model.save(self.args['model'])
    #     # f = open(self.args["le"], "wb")
    #     # f.write(pickle.dumps(le))
    #     # f.close()
    #
    #     # write the face recognition model to output
    #     # model.save(self.args.model)
    #     model.save(self.args.training_model)
    #     f = open(self.args.le, "wb")
    #     f.write(pickle.dumps(le))
    #     f.close()
    #
