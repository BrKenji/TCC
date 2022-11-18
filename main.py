## Data manipulation packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Data visualization package
import seaborn as sns

# Tensorflow and Keras for creating neural network models
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler

# import NN layers and others components
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.utils import np_utils
from keras.callbacks import EarlyStopping


def main():

    # Data pre-selection ----------------------------------------------------------------------------
    df_L10 = pd.read_excel("./database/L10_values_treated(6)_sem_NaN.xlsx")

    df_L10 = df_L10.sample(frac=1).reset_index(drop=True)

    encoder = LabelEncoder()
    #std_scaler = StandardScaler()
    scaler = MinMaxScaler()

    y = df_L10["Diag"]
    X = df_L10.drop(['Paciente', 'Diag', 'TOTAL'], axis=1)

    # Fitting SOMA feature large values
    #std_scaler = std_scaler.fit(X["SOMA"].values.reshape(-1, 1))
    scaler = scaler.fit(X["SOMA"].values.reshape(-1, 1))
    X["SOMA"] = scaler.transform(X["SOMA"].values.reshape(-1, 1))
    print(X["SOMA"])
    # Encoding Gender feature and Diag multi-class label
    gender_encoded_X = X.copy()

    for col in gender_encoded_X.select_dtypes(include='O').columns:
        gender_encoded_X[col] = encoder.fit_transform(gender_encoded_X[col])
    
    encoder.fit(y)
    encoded_diag = encoder.transform(y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_diag = np_utils.to_categorical(encoded_diag)

    # convert to numpy arrays
    gender_encoded_X = np.array(gender_encoded_X)
    # ----------------------------------------------------------------------------------------------
    
    # Defining Model -------------------------------------------------------------------------------
    # Build a network
    model = Sequential()
    model.add(Dense(8, input_shape=(X.shape[1],), activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.summary()

    # Compile neural network
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss',
                        mode='min',
                        patience=10,
                        restore_best_weights=True)

    history = model.fit(gender_encoded_X,
                        dummy_diag,
                        callbacks=[es],
                        epochs=800000,
                        batch_size=10,
                        shuffle=True,
                        validation_split=0.3,
                        verbose=1)

    # ----------------------------------------------------------------------------------------------

    # Evaluating Model - Accuracy and Loss ---------------------------------------------------------
    history_dict = history.history
    # Learning Curve - Accuracy
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']

    # loss
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    # Plotting
    plt.plot(epochs, acc, 'r', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    #plt.figure()
    #plt.plot(epochs, loss, 'r', label='Training Loss')
    #plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    #plt.title('Training and Validation Loss')
    #plt.xlabel('Epochs')
    #plt.ylabel('Loss')
    #plt.legend()

    plt.show()
    # ----------------------------------------------------------------------------------------------

    # Evaluating the model - Confusio Matrix--------------------------------------------------------
    preds = model.predict(gender_encoded_X)
    print(preds[0])
    print(np.sum(preds[0]))

    matrix = confusion_matrix(dummy_diag.argmax(axis=1), preds.argmax(axis=1))
    print(matrix)

    print(classification_report(dummy_diag.argmax(axis=1), preds.argmax(axis=1)))

    # ----------------------------------------------------------------------------------------------

    print("Here Working")

main()
