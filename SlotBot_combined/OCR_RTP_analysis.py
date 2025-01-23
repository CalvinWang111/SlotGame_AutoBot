import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('test.csv',encoding='utf-8',dtype=str,keep_default_na=False)

fig, ax = plt.subplots(figsize=(10, 6))

#each column
Y = 0
for column in df.columns:
    df[column] = df[column].astype(str)
    Y += 1
    X = 0
    for data in df[column]:
        X += 1
        if data != "NULL":
            ax.barh(Y, 1, left=X, height=0.4, color="blue")

plt.tight_layout()
plt.show()
#print(df[df.columns[0]][7])