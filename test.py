import pandas as pd

df = pd.read_excel("Main_data_NotEncoded.xlsx")

print(df['khoangls'].unique())