import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('mp-non-metals.csv')
df = df.iloc[:, 1:]
df.columns = ['formula', 'target']

df_train, df_val = train_test_split(df, train_size=0.7)
df_val, df_test = train_test_split(df_val, train_size=0.5)


df_train.to_csv('train.csv', index=False)
df_val.to_csv('val.csv', index=False)
df_test.to_csv('test.csv', index=False)
