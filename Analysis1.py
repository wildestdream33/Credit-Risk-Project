# Import packages
import json
import glob
import pandas as pd
import numpy as np
import datetime as dt
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA

# Output files for saving detailed analysis
output_file_before = "before_pca_NO_Skewness.txt"
output_file_after = "after_pca_NO_Skewness.txt"

# Helper function to write output to a file
def write_output(text, file):
    with open(file, "a") as f:
        f.write(text + "\n")

# Load the dataset
file_path = r'D:\5450\application_data.csv'
df_application = pd.read_csv(file_path)
print("Loaded dataset preview:\n", df_application.head())

# Initial shape of the dataset
# This step shows the number of rows and columns before any processing
print("\nInitial Shape of the Dataset:")
print(f"Rows: {df_application.shape[0]}, Columns: {df_application.shape[1]}")
write_output(f"Initial Shape: Rows: {df_application.shape[0]}, Columns: {df_application.shape[1]}", output_file_before)

# Drop columns with more than 20% missing values
# This reduces the dimensionality by removing columns with a significant amount of missing data
na_percentage = (df_application.isna().sum() / len(df_application)) * 100
na_percentage_df2 = na_percentage.to_frame(name='NaN_Percentage').sort_values(by='NaN_Percentage', ascending=False).loc[na_percentage > 20]
df_application_clean = df_application.drop(na_percentage_df2.index, axis=1)
print("\nShape after dropping high-missing columns:")
print(f"Rows: {df_application_clean.shape[0]}, Columns: {df_application_clean.shape[1]}")
write_output(f"Shape after dropping columns: Rows: {df_application_clean.shape[0]}, Columns: {df_application_clean.shape[1]}", output_file_before)

# Drop remaining rows with missing values
# This step removes any remaining rows that contain missing data
df_application_clean = df_application_clean.dropna()
print("\nShape after dropping missing rows:")
print(f"Rows: {df_application_clean.shape[0]}, Columns: {df_application_clean.shape[1]}")
write_output(f"Shape after dropping missing rows: Rows: {df_application_clean.shape[0]}, Columns: {df_application_clean.shape[1]}", output_file_before)

# Convert binary columns (e.g., 'Y'/'N' to 1/0)
# This standardizes binary data for analysis
flag_columns = [col for col in df_application_clean.columns if set(df_application_clean[col].unique()) <= {0, 1} or set(df_application_clean[col].unique()) <= {'Y', 'N'}]
flag_columns.remove('TARGET')
for col in flag_columns:
    df_application_clean[col] = df_application_clean[col].replace({'Y': 1, 'N': 0}).astype(int)

# Remove 'XNA' values from CODE_GENDER (fix)
# This step ensures that 'XNA' values are removed, as they are likely outliers
df_application_clean2 = df_application_clean.drop(df_application_clean[df_application_clean['CODE_GENDER'] == 'XNA'].index)
df_application_clean2.reset_index(drop=True, inplace=True)
print("\nShape after removing 'XNA' in CODE_GENDER:")
print(f"Rows: {df_application_clean2.shape[0]}, Columns: {df_application_clean2.shape[1]}")
write_output(f"Shape after removing 'XNA' in CODE_GENDER: Rows: {df_application_clean2.shape[0]}, Columns: {df_application_clean2.shape[1]}", output_file_before)

# Group 'XNA' under 'Other' in ORGANIZATION_TYPE
# This step consolidates 'XNA' values under 'Other' to simplify the analysis
df_application_clean2['ORGANIZATION_TYPE'] = df_application_clean2['ORGANIZATION_TYPE'].replace('XNA', 'Other')

# Verify the 'Other' grouping in ORGANIZATION_TYPE
# This check ensures that the grouping of 'XNA' into 'Other' was successful
check = df_application_clean2[['SK_ID_CURR', 'TARGET', 'ORGANIZATION_TYPE']].groupby(['ORGANIZATION_TYPE', 'TARGET']).count()
print("\nVerification after grouping 'XNA' under 'Other':\n", check.loc[['Other']])
write_output("\nVerification after grouping 'XNA' under 'Other':\n" + str(check.loc[['Other']]), output_file_before)

# One-hot encoding for categorical variables
# This step transforms categorical variables into binary columns
non_numeric_columns = df_application_clean2.select_dtypes(exclude='number').columns
encoded_application_clean = pd.get_dummies(df_application_clean2, columns=non_numeric_columns)
print("\nShape after one-hot encoding:")
print(f"Rows: {encoded_application_clean.shape[0]}, Columns: {encoded_application_clean.shape[1]}")
write_output(f"Shape after one-hot encoding: Rows: {encoded_application_clean.shape[0]}, Columns: {encoded_application_clean.shape[1]}", output_file_before)

# Standardize the dataset
# This scales the numeric features to have a mean of 0 and variance of 1
numeric_columns = df_application_clean2.select_dtypes(include='number').columns
numeric_columns = numeric_columns.drop(['SK_ID_CURR', 'TARGET'])
numeric_columns = numeric_columns.drop(flag_columns)

scaler = StandardScaler()
encoded_application_clean_std = encoded_application_clean.copy()
encoded_application_clean_std[numeric_columns] = scaler.fit_transform(encoded_application_clean_std[numeric_columns])

# Shape after standardization
print("\nShape after standardization:")
print(f"Rows: {encoded_application_clean_std.shape[0]}, Columns: {encoded_application_clean_std.shape[1]}")
write_output(f"Shape after standardization: Rows: {encoded_application_clean_std.shape[0]}, Columns: {encoded_application_clean_std.shape[1]}", output_file_before)

# Variance, Skewness, and Kurtosis before and after standardization
# These statistics help assess the effect of standardization
def calculate_statistics(data, label):
    variance = data.var().sum()
    skewness = data.apply(lambda x: skew(x))
    kurtosis_values = data.apply(lambda x: kurtosis(x))
    print(f"\n### {label} ###")
    print(f"Total Variance: {variance}")
    print(f"Skewness:\n{skewness}")
    print(f"Kurtosis:\n{kurtosis_values}")
    return variance, skewness, kurtosis_values

# Before Standardization
variance_before, skewness_before, kurtosis_before = calculate_statistics(encoded_application_clean[numeric_columns], "Before Standardization")

# After Standardization
variance_after_std, skewness_after_std, kurtosis_after_std = calculate_statistics(encoded_application_clean_std[numeric_columns], "After Standardization")

# PCA Analysis
# Principal Component Analysis for dimensionality reduction
pca = PCA()
pca_transformed = pca.fit_transform(encoded_application_clean_std[numeric_columns])
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
optimal_components = np.argmax(cumulative_variance >= 0.95) + 1

# Apply PCA with the optimal number of components
pca_optimal = PCA(n_components=optimal_components)
pca_transformed_optimal = pca_optimal.fit_transform(encoded_application_clean_std[numeric_columns])

# After PCA
variance_after_pca, skewness_after_pca, kurtosis_after_pca = calculate_statistics(pd.DataFrame(pca_transformed_optimal), "After PCA")

# After PCA Shape
print("\nShape after PCA:")
print(f"Rows: {pca_transformed_optimal.shape[0]}, Columns: {pca_transformed_optimal.shape[1]}")
write_output(f"Shape after PCA: Rows: {pca_transformed_optimal.shape[0]}, Columns: {pca_transformed_optimal.shape[1]}", output_file_after)


# Summary and completion
print("\nScript executed successfully. Detailed outputs are saved in 'before_pca_detailed.txt' and 'after_pca_detailed.txt'.")










