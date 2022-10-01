import pandas as pd

target_name = './BreastCancer_test'

df = pd.read_csv(f'{target_name}.csv')
df.drop(columns=['Id'],inplace=True)
df.fillna(value=df[df.columns[:-1]].mean(),inplace=True)
df.to_csv(f'{target_name}_preprocessed.csv',index=False)