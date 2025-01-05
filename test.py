import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the uploaded CSV file
file_path = 'https://raw.githubusercontent.com/athirahwahhab/fyp/refs/heads/main/data/combined_output_latest.csv'
df = pd.read_csv(file_path)

# Display the first few rows to understand its structure
df.head()


# Check the data types of each column
print(df.info())

# Generate descriptive statistics for numerical columns
print(df.describe())

# Identify missing values in the dataset
print(df.isnull().sum())

# Visualize the distribution of numerical features using histograms
df.hist(figsize=(15, 10))
plt.show()

# Create a correlation matrix to observe relationships between numerical features
correlation_matrix = df.select_dtypes(include=['number']).corr()  
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()
st.pyplot(plt.gcf())
# Explore categorical features using bar plots (replace 'categorical_column' with actual column name)
for col in df.select_dtypes(include=['object']):
    plt.figure(figsize=(10, 6))
    df[col].value_counts().plot(kind='bar')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()
  st.pyplot(plt.gcf())
