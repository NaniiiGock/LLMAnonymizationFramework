import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_path = 'models/NER_models/bert/ner_datasetreference.csv'  
df = pd.read_csv(data_path, encoding='ISO-8859-1') 

print("Basic Overview of the Data:")
print(f"Dataset Shape: {df.shape}")
print(f"Columns: {df.columns}")
print(f"Data Types:\n{df.dtypes}\n")
print("First 5 Rows:")
print(df.head())

print("\nSummary Statistics:")
print(df.describe(include='all'))

print("\nNER Tag Distribution:")
ner_tag_counts = df['Tag'].value_counts()
print(ner_tag_counts)

plt.figure(figsize=(10, 6))
sns.countplot(y='Tag', data=df, order=ner_tag_counts.index, palette='Set2')
plt.title('Distribution of NER Tags')
plt.xlabel('Count')
plt.ylabel('NER Tag')
plt.show()

print("\nMissing Data Analysis:")
missing_data = df.isnull().sum()
print(missing_data[missing_data > 0])

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.show()

print("\nSample Data (First 20 rows):")
print(df.head(20))



df['Sentence #'] = df['Sentence #'].fillna(method='ffill')

sentence_ner_tag_counts = df.groupby('Sentence #')['Tag'].value_counts().unstack(fill_value=0)

print("NER Tag Counts per Sentence:")
print(sentence_ner_tag_counts.head())

with open("data_exploration_output.txt", "w") as f:
    f.write(f"Basic Overview of the Data:\n{df.shape}\n")
    f.write(f"Columns: {df.columns}\n")
    f.write(f"Data Types:\n{df.dtypes}\n")
    f.write("Summary Statistics:\n")
    f.write(str(df.describe(include='all')))
    f.write("\nNER Tag Distribution:\n")
    f.write(str(ner_tag_counts))
    f.write("\nMissing Data Analysis:\n")
    f.write(str(missing_data[missing_data > 0]))
