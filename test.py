import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Example data processing script with Streamlit integration
def main():
    # Load your dataset
    df = pd.read_csv("https://raw.githubusercontent.com/athirahwahhab/fyp/refs/heads/main/data/combined_output_latest.csv")

    # Display dataset information
    st.write("## Exploratary Data Analysis")

    # Generate and display descriptive statistics
    st.write("## Descriptive Statistics")
    st.write(df.describe())

    # Identify and display missing values
    st.write("## Missing Values")
    missing_values = df.isnull().sum()
    st.write(missing_values)

    # Visualize the distribution of numerical features using histograms
    st.write("## Numerical Feature Distributions")
    df.hist(figsize=(15, 10))
    plt.tight_layout()  # Adjust layout for better visualization
    st.pyplot(plt.gcf())  # Use this in Streamlit to display plots
    # Create and display a correlation matrix heatmap
    st.write("## Correlation Matrix")
    correlation_matrix = df.select_dtypes(include=['number']).corr()
    # Plot the heatmap
    plt.figure(figsize=(10, 8))  # Adjust figure size as needed
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot(plt.gcf())  # Render the heatmap in Streamlit

if __name__ == "__main__":
    main()
