
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the uploaded CSV file
file_path = '/content/combined_output_latest.csv'
df = pd.read_csv(file_path)

# Display the first few rows to understand its structure
df.head()
