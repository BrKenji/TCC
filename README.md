# Objective
Develop a simple artificial neural network capable of predicting the 
cardiomyopathy diagnosis of a patient

# Functionality
The user can choose what features to use, and how many neurons to use in the hidden layer.
Every time the user runs the code it'll save the information about the MLP and the evaluation performace of the model in a CSV file called "testing_results.csv"

# Activation Function

## Softmax Function: 

* it calculates the probabilities distribution of the event over K different events
* this function will calculate the probabilities of each target class over all possible target classes
* the range will be 0 to 1, and the sum of all the probabilities will be equal to one

# Dataset
The dateset provide the age, gender, somaSpharm, octantSPHARMS (which will be represented from O1 to O8) and diagnostics of 394 pacients, each diagnosed with either CMD, CMH or No anomaly.

Paciente| Idade | Gênero | L10_O1 | L10_O2 | L10_O3 | L10_O4 | L10_O5 | L10_O6 | L10_O7 | L10_O8 | SOMA | Diag |
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
1 | 29 | M | -3.164135918705866E8 | -242647.26112422068 | 120228.86522188243 | 441092.93842605286 | 2.106433302094846E7 | -7509872.094907303 | 245855.62114164056 | -1165122.4041963895 | -17691052837199900 | SEM |
102 | 55 | M | -961332.8130004894 | 1201938.0552942532 | -8268.272625276992 | -380325.27013825893 | 1.167387101949857E8 | 5208.737216613921 | 69268.24292192739  | 117137.50198502 | -27792643841918000 | CMH |
286 | 49 | F | -4.258348567755413E8 | -7700.317699685315 | 16437.398808804315 | -18016.66847471462 | -7.093095948563371E8 | -14109.739594618277 | 13836.174181875955 | -241159.66459508796 | 37452431625789200 | CMD |

## Dataset Diagnostics Values
* 0 -> CMD
* 1 -> CMH
* 2 -> Sem anomalia