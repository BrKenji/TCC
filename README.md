# Objective
Develop a simple artificial neural network capable of predicting the 
cardiomyopathy diagnosis of a patient

# Activation Function

## Softmax Function: 

* it calculates the probabilities distribution of the event over K different events
* this function will calculate the probabilities of each target class over all possible target classes
* the range will be 0 to 1, and the sum of all the probabilities will be equal to one

# Dataset
The dateset provides somaSpharm and diagnostics of 394 pacients, each diagnosed with either CMD, CMH or No anomaly.

Example:

Paciente | SOMA | Diag |
--- | --- | --- |
1 | -17691052837199900 | SEM |
102 | -27792643841918000 | CMH |
284 | -17404489207322700 | CMD |

## Dataset Diagnostics Values
* 0 -> CMD
* 1 -> CMH
* 2 -> Sem anomalia