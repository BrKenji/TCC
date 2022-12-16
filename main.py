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

def buildMLP(n_hidden, input_number):
    model = Sequential()
    model.add(Dense(n_hidden, input_shape=(input_number,), activation='relu'))
    #model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))
    model.summary()

    # Compile neural network
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def trainMLP(model, X_train, X_test, y_train, y_test):

    es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)

    history = model.fit(X_train,
                        y_train,
                        callbacks=[es],
                        epochs=800000,
                        batch_size=10,
                        shuffle=True,
                        validation_data=(X_test, y_test),
                        verbose=0)

    return history

# Get user input
def userChooseFeatures():
    features = []
    features_for_results_file = []

    while (True):
        desiredFeature = str(input("Feature a ser usada: ")).upper()
        if (desiredFeature == ""):
            break
        else:
            features_for_results_file.append(desiredFeature)
            if (desiredFeature == "OCTANTE"):
                for i in range(8):
                    features.append(f"L10_O{i + 1}")
            else:
                features.append(desiredFeature)
    
    return features, features_for_results_file

# Save the MLP info and performance
def saveDataToCsv(dictionary, df):
    df = df.append(dictionary, ignore_index=True)
    df.to_csv("testing_results.csv", index=False)

# Get or Create and get results test
def getTestingResultsDf():
    try:
        results_df = pd.read_csv("./testing_results.csv")
    except FileNotFoundError:
        df = pd.DataFrame(columns=['features', 'neurons', 'epochs', 'CMD_Accuracy', 'CMH_Accuracy', 'SEM_Accuracy', 'CMD_ROC', 'CMH_ROC', 'SEM_ROC'])
        df.to_csv("testing_results.csv", index=False)
        results_df = pd.read_csv("./testing_results.csv")

    return results_df

# Plot accuracy or loss curves
def plotAccuracyLoss(training_value, validation_value, epochs, mode, n_hidden):
    plot_title, plot_ylabel , val_label, training_label= "", "", "", ""

    match (mode):
        case "acc":
            plot_ylabel = "Accuracy"
            plot_title = f'Training and Validation {plot_ylabel} with {n_hidden} Hidden Layer Nodes'
            val_label, training_label = f'Training {plot_ylabel}', f'Validation {plot_ylabel}'
        case "loss":
            plot_ylabel = "Loss"
            plot_title = f'Training and Validation {plot_ylabel} with {n_hidden} Hidden Layer Nodes'
            val_label, training_label = f'Training {plot_ylabel}', f'Validation {plot_ylabel}'

    plt.figure()
    plt.plot(epochs, training_value, 'r', label=training_label)
    plt.plot(epochs, validation_value, 'b', label=val_label)
    plt.title(plot_title)
    plt.xlabel('Epochs')
    plt.ylabel(plot_ylabel)
    plt.legend()
    plt.show()

# Plot ROC Curves for each class and return the areas
def rocModelEvaluation(diag_dict, n_classes, dummy_y, y_pred, n_hidden):
    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(dummy_y[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro-average ROC Curve and ROC Area
    fpr["micro"], tpr["micro"], _ = roc_curve(dummy_y.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
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

    cmd_roc, cmh_roc, sem_roc = 0, 0, 0
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(diag_dict.get(i), roc_auc[i]),
        )
        
        roc_pred_diag = diag_dict.get(i)
        pred_roc = round(roc_auc[i] * 100, 2)
        match (roc_pred_diag):
            case 'SEM':
                sem_roc = pred_roc
            case 'CMD':
                cmd_roc = pred_roc
            case 'CMH':
                cmh_roc = pred_roc        

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Some extension of ROC to multiclass with {n_hidden} hidden layer neurons")
    plt.legend(loc="lower right")
    plt.show()

    return cmd_roc, cmh_roc, sem_roc

def main():
    # Data pre-selection ----------------------------------------------------------------------------
    df_L10 = pd.read_excel("./database/L10_values_treated(8)_sem_NaN.xlsm").sample(frac=1)

    results_df = getTestingResultsDf()

    encoder = LabelEncoder()
    scaler = MinMaxScaler()

    # Encoding Gênero feature
    df_L10["GENDER"] = encoder.fit_transform(df_L10["GENDER"])
    # Fitting SOMA feature large values
    df_L10["SOMA"] = scaler.fit_transform(df_L10["SOMA"].values.reshape(-1, 1))
    # Encoding Diag label
    encoder.fit(df_L10["DIAG"])
    df_L10["DIAG"] = encoder.transform(df_L10["DIAG"])

    for i in range(8):
        df_L10[f"L10_O{i + 1}"] = scaler.fit_transform(df_L10[f"L10_O{i + 1}"].values.reshape(-1, 1))

    features, results_features = userChooseFeatures()

    n_hidden = int(input("Número de nós na camada oculta: "))

    diag_dict = {0: "CMD", 1: "CMH", 2: "SEM"}

    y = df_L10["DIAG"]
    X = pd.DataFrame()

    for i in features:
        X[i] = df_L10[i]    

    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_diag = np_utils.to_categorical(y)
    n_classes = dummy_diag.shape[1]
    # convert to numpy arrays
    X = np.array(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, dummy_diag, random_state=1)

    # ----------------------------------------------------------------------------------------------
    
    # Defining Model -------------------------------------------------------------------------------
    model = buildMLP(n_hidden, X.shape[1])

    history = trainMLP(model, X_train, X_test, y_train, y_test)

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

    n_epochs = len(acc) + 1
    epochs = range(1, n_epochs)

    # Plotting
    plotAccuracyLoss(acc, val_acc, epochs, "acc", n_hidden)
    #plotAccuracyLoss(loss, val_loss, epochs, "loss", n_hidden)

    # ----------------------------------------------------------------------------------------------
    # Evaluating the model - Confusio Matrix--------------------------------------------------------
    y_pred = model.predict(X)

    matrix = confusion_matrix(dummy_diag.argmax(axis=1), y_pred.argmax(axis=1))
    print(matrix)
    cm = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis] 

    cmd_acc, cmh_acc, sem_acc = 0, 0, 0

    for i in range(len(cm.diagonal())):
        diagnose_predicted = diag_dict.get(i)
        pred_acc = round(cm.diagonal()[i] * 100, 2)
        match (diagnose_predicted):
            case 'SEM':
                sem_acc = pred_acc
            case 'CMD':
                cmd_acc = pred_acc
            case 'CMH':
                cmh_acc = pred_acc
                
        print(f'{diagnose_predicted} accuracy: {cm.diagonal()[i]}')
        
    print(classification_report(dummy_diag.argmax(axis=1), y_pred.argmax(axis=1)))
    
    print(f'AUROC score: {roc_auc_score(dummy_diag, y_pred, average="weighted", multi_class="ovr")}')
    
    # ----------------------------------------------------------------------------------------------

    # ROC Evaluation Metric ------------------------------------------------------------------------
    
    cmd_roc, cmh_roc, sem_roc = rocModelEvaluation(diag_dict, n_classes, dummy_diag, y_pred, n_hidden)

    # ----------------------------------------------------------------------------------------------

    # Saving results
    d = {
        "features": results_features,
        "neurons": n_hidden,
        "epochs": n_epochs,
        "CMD_Accuracy": cmd_acc,
        "CMH_Accuracy": cmh_acc,
        "SEM_Accuracy": sem_acc,
        "CMD_ROC": cmd_roc,
        "CMH_ROC": cmh_roc,
        "SEM_ROC": sem_roc
    }

    saveDataToCsv(d, results_df)

    print("Here Working")

main()
