import pandas as pd
import numpy as np

df = pd.read_excel("./database/L10_values_treated.xlsx")

#print(df.head())
#print(type(df["TOTAL"][1]))


# This method receives the TotalSPHARM of a pacient converts it from string to list,
# and every coeficient from said list from string to float and returns the list 
# with float type elements
def getCoeficientList(coeficient_string):
    coeficient_string = coeficient_string.split(",")
    coeficient_string = list(filter(None, coeficient_string))
    return list(map(float, coeficient_string))
    
# ----------------------------------------------------------------------------------