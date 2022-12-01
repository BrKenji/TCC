import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.preprocessing import LabelEncoder

df = pd.read_excel("./database/L10_values_treated(6)_sem_NaN.xlsx")

encoder = LabelEncoder()
df["Gênero"] = encoder.fit_transform(df["Gênero"])

sns.set_style("darkgrid")
sns.jointplot(data=df, x="Idade", y="Gênero", hue="Diag")
plt.show()