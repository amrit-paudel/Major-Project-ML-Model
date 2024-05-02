#
#
#
#
#
#
# from keras.layers import Dense, Dropout
# from keras.models import Sequential
# from keras.optimizers import Adam
# import keras
#
# class SoftMax():
#     def __init__(self, input_shape, num_classes):
#         self.input_shape = input_shape
#         self.num_classes = num_classes
#
#     def build(self):
#         model = Sequential()
#         model.add(Dense(1024, activation='relu', input_shape=self.input_shape))
#         model.add(Dropout(0.5))
#         model.add(Dense(1024, activation='relu'))
#         model.add(Dropout(0.5))
#         model.add(Dense(self.num_classes, activation='softmax'))
#
#
#         optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#         model.compile(loss=keras.losses.categorical_crossentropy,
#                       optimizer=optimizer,
#                       metrics=['accuracy'])
#         return model



#
# from keras.layers import Dense, Dropout, BatchNormalization
# from keras.models import Sequential
# from keras.optimizers import Adam
# import keras
#
# class SoftMax():
#     def __init__(self, input_shape, num_classes):
#         self.input_shape = input_shape
#         self.num_classes = num_classes
#
#     def build(self):
#         model = Sequential()
#         model.add(Dense(512, activation='relu', input_shape=self.input_shape))  # Reduced number of neurons
#         model.add(BatchNormalization())  # Added batch normalization
#         model.add(Dropout(0.6))  # Increased dropout rate
#
#         model.add(Dense(256, activation='relu'))  # Added an additional dense layer with fewer neurons
#         model.add(BatchNormalization())
#         model.add(Dropout(0.6))
#
#         model.add(Dense(128, activation='relu'))  # Added another dense layer with fewer neurons
#         model.add(BatchNormalization())
#         model.add(Dropout(0.6))
#
#         model.add(Dense(self.num_classes, activation='softmax'))
#
#         optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#         model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
#
#         return model




#
# from keras.layers import Dense, Dropout, BatchNormalization
# from keras.models import Sequential
# from keras.optimizers import Adam
# from keras.regularizers import l2
# from keras.callbacks import ReduceLROnPlateau, EarlyStopping
# import keras
#
# class SoftMax():
#     def __init__(self, input_shape, num_classes):
#         self.input_shape = input_shape
#         self.num_classes = num_classes
#
#     def build(self):
#         model = Sequential()
#         model.add(Dense(512, activation='relu', input_shape=self.input_shape, kernel_regularizer=l2(0.001)))  # L2 regularization
#         model.add(BatchNormalization())
#         model.add(Dropout(0.5))  # Adjusted dropout rate
#
#         model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))  # L2 regularization
#         model.add(BatchNormalization())
#         model.add(Dropout(0.5))
#
#         model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))  # L2 regularization
#         model.add(BatchNormalization())
#         model.add(Dropout(0.5))
#
#         model.add(Dense(self.num_classes, activation='softmax'))
#
#         # Learning rate scheduler
#         lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto', cooldown=0, min_lr=1e-7)
#
#         # Early stopping
#         # early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
#         early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
#
#         optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#         model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
#
#         return model, lr_scheduler, early_stopping



from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import LeakyReLU
import keras

class SoftMax():
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self):
        model = Sequential()
        model.add(Dense(512, kernel_regularizer=l2(0.01), input_shape=self.input_shape))  # L2 regularization
        model.add(LeakyReLU(alpha=0.1))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))  # Adjusted dropout rate

        model.add(Dense(256, kernel_regularizer=l2(0.01)))  # L2 regularization
        model.add(LeakyReLU(alpha=0.1))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(128, kernel_regularizer=l2(0.01)))  # L2 regularization
        model.add(LeakyReLU(alpha=0.1))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(self.num_classes, activation='softmax'))

        # Learning rate scheduler
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto', cooldown=0, min_lr=1e-7)

        # Early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

        return model, lr_scheduler, early_stopping



