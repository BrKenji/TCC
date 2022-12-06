## Data manipulation packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Data visualization package
import seaborn as sns

# Sklearn Modules
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# import NN layers and others components
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

from itertools import cycle

def main():

    # Data pre-selection ----------------------------------------------------------------------------
    df_L10 = pd.read_excel("./database/L10_values_treated(6)_sem_NaN.xlsx")
    df_L10 = df_L10.sample(frac=1)

    features = []

    while (True):
        desiredFeature = str(input("Feature a ser usada: "))
        if (desiredFeature == ""):
            break
        else:
            features.append(desiredFeature)
        

    n_hidden = int(input("Número de nós na camada oculta: "))

    dict = {0: "CMD", 1: "CMH", 2: "SEM"}

    encoder = LabelEncoder()
    scaler = MinMaxScaler()

    # Encoding Gênero feature
    df_L10["Gênero"] = encoder.fit_transform(df_L10["Gênero"])
    # Fitting SOMA feature large values
    df_L10["SOMA"] = scaler.fit_transform(df_L10["SOMA"].values.reshape(-1, 1))
    # Encoding Diag label
    encoder.fit(df_L10["Diag"])
    df_L10["Diag"] = encoder.transform(df_L10["Diag"])

    y = df_L10["Diag"]
    X = pd.DataFrame()
    for i in features:
        X[i] = df_L10[i]

    #X = df_L10.drop(['Paciente', 'Diag', 'TOTAL'], axis=1)    

    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_diag = np_utils.to_categorical(y)
    n_classes = dummy_diag.shape[1]
    # convert to numpy arrays
    X = np.array(X)

    X_train, X_test, y_train, y_test = train_test_split(X, dummy_diag, random_state=1)

    # ----------------------------------------------------------------------------------------------
    
    # Defining Model -------------------------------------------------------------------------------
    # Build a network
    model = Sequential()
    model.add(Dense(n_hidden, input_shape=(X.shape[1],), activation='relu'))
    #model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))
    model.summary()

    # Compile neural network
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss',
                        mode='min',
                        patience=10,
                        restore_best_weights=True)

    history = model.fit(X_train,
                        y_train,
                        callbacks=[es],
                        epochs=800000,
                        batch_size=10,
                        shuffle=True,
                        validation_data=(X_test, y_test),
                        verbose=2)

    # ----------------------------------------------------------------------------------------------

    # Evaluating Model -----------------------------------------------------------------------------
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Loss: {score[0]}')
    print(f'Test Accuracy: {score[1]}')

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
    plt.title(f'Training and Validation Accuracy with {n_hidden} Hidden Layer Nodes')
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
    y_pred = model.predict(X)

    matrix = confusion_matrix(dummy_diag.argmax(axis=1), y_pred.argmax(axis=1))
    print(matrix)
    cm = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis] 
    for i in range(len(cm.diagonal())):
        print(f'{dict.get(i)} accuracy: {cm.diagonal()[i]}')
        
    print(classification_report(dummy_diag.argmax(axis=1), y_pred.argmax(axis=1)))
    
    print(f'AUROC score: {roc_auc_score(dummy_diag, y_pred, average="weighted", multi_class="ovr")}')
    
    # ----------------------------------------------------------------------------------------------

    # ROC Evaluation Metric ------------------------------------------------------------------------
    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr ={}
    roc_auc = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(dummy_diag[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Micro-average ROC Curve and ROC Area
    fpr["micro"], tpr["micro"], _ = roc_curve(dummy_diag.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Interpolate all ROC Curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    # Avarage it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure()
    lw = 2

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(dict.get(i), roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Some extension of ROC to multiclass with {n_hidden} hidden layer neurons")
    plt.legend(loc="lower right")
    plt.show()
    # ----------------------------------------------------------------------------------------------
    print("Here Working")

main()
