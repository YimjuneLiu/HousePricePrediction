import numpy as np
import pandas as pd
import random
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import missingno as msno
import warnings

from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import skew

from model import HousePrice

seed = 42
random.seed(seed)
np.random.seed(seed)
os.environ["PYTHONASHSEED"] = str(seed)

# for dirname, _, filenames in os.walk('./2024smarcleks2house-price'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


sample = pd.read_csv("./data/sample_submission.csv")
test_df = pd.read_csv("./data/test.csv")
train_df = pd.read_csv("./data/train.csv")

test_df_id = test_df['Id']

# delete id column
train_df.drop(columns=['Id'], inplace=True)
test_df.drop(columns=['Id'], inplace=True)

# Statistics of Null Value 
# msno.matrix(train_df)
# msno.bar(train_df)
# plt.show()

def visualization(train_df):
    print(train_df['SalePrice'].describe())

    plt.figure(figsize=(9, 8))
    plt.hist(train_df['SalePrice'], bins=100, color='skyblue', alpha=0.4)  # bins is the number of bars
    plt.xlabel('SalePrice')
    plt.ylabel('Frequency')
    plt.title('Distribution of SalePrice')
    plt.grid(True)
    plt.show()

# visualization(train_df)

# the number of missing values 
numeric_columns = train_df.select_dtypes(include=['int64', 'float64'])
missing_values_count1 = numeric_columns.isnull().sum()

# calculate the number of missing value
object_columns = train_df.select_dtypes(include=['object'])
missing_values_count2 = object_columns.isnull().sum()

# Filter out columns with 0 missing values
missing_values_count1 = missing_values_count1[missing_values_count1 > 0]
missing_values_count2 = missing_values_count2[missing_values_count2 > 0]

# # print the number of missing values for numeric and object columns
# print("Missing values in numeric columns:")
# print(missing_values_count1)
# print("\nMissing values in object columns:")
# print(missing_values_count2)

# check correlation heatmap
# select numeric columns
numeric_columns = train_df.select_dtypes(include=['int64', 'float64'])

# calculate the correlation matrix
correlation_matrix = numeric_columns.corr().abs() # take the absolute values

# draw the correlation heatmap
# plt.figure(figsize=(50, 40))
# sns.heatmap(correlation_matrix, annot=False, cmap='crest')
# plt.title('absolute correlation heatmap of numeric columns in train_df')
# plt.show()

missing_values_info = train_df.isnull().sum()
missing_values_info = missing_values_info[missing_values_info > 0]  # select column with missing values
missing_values_info = pd.DataFrame({'Column Name': missing_values_info.index, 'Missing Values': missing_values_info.values, 'Data Type': train_df[missing_values_info.index].dtypes.values})
missing_values_info = missing_values_info.sort_values(by='Missing Values', ascending=False)
# print(missing_values_info)

# visualize the relationship between each numerical column in train_df and 'SalePrice'
# using scatter plots and regression lines
# sns.set(font_scale=1.5)
# sns.set_style("white")
# plt.figure(figsize=(20, 40))
# plt.subplots_adjust(hspace=0.5)
# for idx, feature in enumerate(numeric_columns):
#     if idx % 12 == 0:
#         plt.figure(figsize=(20, 20))
#         plt.subplots_adjust(hspace=0.5)
#     plt.subplot(6, 2, idx % 12 + 1)
#     sns.regplot(data=train_df, x=feature, y='SalePrice', scatter_kws={'color': 'blue'})
#     plt.show()

# float_columns = train_df.select_dtypes(include=['float64'])
# warnings.filterwarnings("ignore")
# i = 1
# sns.set(font_scale=2)
# sns.set_style("white")
# sns.set_palette("bright")
# plt.figure(figsize=(40, 40))
# plt.subplots_adjust(hspace=1)
# for feature in float_columns:
#     plt.subplot(9,4,i)
#     sns.histplot(train_df[feature], palette='Blues_r')
#     i = i +1

# int_columns = train_df.select_dtypes(include=['int64'])

# i = 1
# sns.set(font_scale = 2)
# sns.set_style("white")
# sns.set_palette("bright")
# plt.figure(figsize=(40, 40))
# plt.subplots_adjust(hspace=1)
# for feature in int_columns:
#     plt.subplot(9,4,i)
#     sns.histplot(train_df[feature], palette='Blues_r')
#     i = i +1

# calculate the average saleprice for each yearbuilt
average_price_by_year = train_df.groupby('YearBuilt')['SalePrice'].mean()

# visualization
# plt.figure(figsize=(12, 6))
# sns.lineplot(data=average_price_by_year, palette='Blues_r')
# plt.title('Average SalePrice by YearBuilt')
# plt.xlabel('YearBuilt')
# plt.ylabel('Average SalePrice')
# plt.grid(True)
# plt.show()

# calculate skewness
with np.errstate(divide='ignore', invalid='ignore'):
    skewness = train_df.select_dtypes(include=['int64', 'float64']).apply(lambda x: x.skew())
skewed_columns = skewness[abs(skewness > 1)]

# print list and count of columns with skewness greater than 1
skewed_columns_list = skewed_columns.index.tolist()
# print(f"Number of columns with skewness greater than 1: {len(skewed_columns_list)}")
# print("Columns with skewness greater than 1:")
# for col in skewed_columns_list:
#     print(f"{col}: Skewness = {skewness[col]}")

# Visualize data of columns with skewness greater than 1
# plt.figure(figsize=(40, 30))
# rows = len(skewed_columns_list) // 2 + (len(skewed_columns_list) % 2 > 0)
# for i, col in enumerate(skewed_columns_list, 1):
#     plt.subplot(rows, 2, i)
#     sns.histplot(train_df[col], kde=True)
#     plt.title(f'{col} Distribution (Skewness: {skewness[col]})')
# plt.tight_layout()
# plt.show()

# lease columns with skewness > 1
# for col in skewed_columns_list:
#     print(col)

# using KNN to handle missing values
# List of columns with missing values to be filled
columns_to_impute = ['MSSubClass', 'LotFrontage', 'LotArea', 'MasVnrArea', 
                     'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', '1stFlrSF', 
                     'LowQualFinSF', 'GrLivArea', 'BsmtHalfBath', 'KitchenAbvGr', 
                     'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 
                     'ScreenPorch', 'PoolArea', 'MiscVal', 'SalePrice']

# create a KNNImputer object
imputer = KNNImputer(n_neighbors=5)

# use the fit_transform method to fill missing values in the specified columns
imputed_data = imputer.fit_transform(train_df[columns_to_impute])
train_df.loc[:, columns_to_impute] = imputed_data

# check if any missing values are left after imputation
missing_values_after_imputation = train_df[columns_to_impute].isnull().sum()
# print("Number of missing values after imputation:")
# print(missing_values_after_imputation)

# test_df
# List of columns with missing values to be filled
columns_to_impute = ['MSSubClass', 'LotFrontage', 'LotArea', 'MasVnrArea', 
                     'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', '1stFlrSF', 
                     'LowQualFinSF', 'GrLivArea', 'BsmtHalfBath', 'KitchenAbvGr', 
                     'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 
                     'ScreenPorch', 'PoolArea', 'MiscVal']

# Create a KNNImputer object
imputer = KNNImputer(n_neighbors=5)

# Use the fit_transform method to fill missing values in the specified columns
imputed_data = imputer.fit_transform(test_df[columns_to_impute])
test_df.loc[:, columns_to_impute] = imputed_data

# Check if any missing values are left after imputation
missing_values_after_imputation = test_df[columns_to_impute].isnull().sum()
# print("Number of missing values after imputation:")
# print(missing_values_after_imputation)

# still missing values
missing_values_info = train_df.isnull().sum()
missing_values_info = missing_values_info[missing_values_info > 0] 
missing_values_info = pd.DataFrame({'Column Name': missing_values_info.index,
                                    'Missing Values': missing_values_info.values,
                                    'Data Type': train_df[missing_values_info.index].dtypes.values})
# print(missing_values_info)

# List of column names to check
missing_columns = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
           'BsmtFinType2', 'Electrical', 'FireplaceQu', 'GarageType',
           'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond',
           'Fence', 'MiscFeature', 'PoolQC']

#  Print unique values for each column in the list
# for column in missing_columns:
#     unique_values = train_df[column].unique()
#     print(f"Unique values in column '{column}':")
#     print(unique_values)
#     print()


# 
check = ['GarageYrBlt', 'YearBuilt', 'TotalBsmtSF', 'Fireplaces']

# for col in check:
#     print(train_df[col].describe())
#     print()

# the non-existed cases are filled with non-existent cases
def fill_garage_yr_blt(df):
    """
    Fill missing values in 'GarageYrBlt' column where the corresponding 'YearBuilt' value is less than or equal to 1900.

    Args:
    - df: DataFrame containing the data

    Returns:
    - DataFrame with missing values filled in 'GarageYrBlt' column
    """
    # Calculate median of GarageYrBlt
    garage_yr_blt_median = df['GarageYrBlt'].median()

    # Fill missing values in GarageYrBlt where the corresponding YearBuilt value is less than or equal to 1900
    df['GarageYrBlt'] = df.apply(lambda row: row['YearBuilt'] if row['YearBuilt'] <= 1900 else garage_yr_blt_median if pd.isnull(row['GarageYrBlt']) else row['GarageYrBlt'], axis=1)

    return df

# Apply the function to train_df_filled
train_df = fill_garage_yr_blt(train_df)

# Apply the function to test_df_filled
test_df = fill_garage_yr_blt(test_df)

# print(train_df['GarageYrBlt'].isnull().sum())

# reduce skewness, use Log Transformation or Box-Cox transformation.
# However, Box-Cox does not work when the data contains zeros or negative numbers.
# since data often contains zeros, -->use Log Transformation.
# The percentage of negative numbers is so high that a logarithmic transformation doesn't seem appropriate.
# It need to do a little more culling. Or to find another way.
# convert the missing values to the object type data 'none'.


# label encoding of object data.
def preprocess_data(df):
    # List of columns with missing values to be filled with 'none'
    columns_to_fill_none = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 
                            'BsmtFinType1','BsmtFinType2', 'Electrical', 
                            'FireplaceQu', 'GarageType', 'GarageFinish', 
                            'GarageQual', 'GarageCond','Fence', 'MiscFeature']

    # Fill missing values with 'none'
    for col in columns_to_fill_none:
        df[col] = df[col].fillna('none')
    
    # Perform label encoding
    label_encoder = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = label_encoder.fit_transform(df[col])
    
    return df

# Preprocess test_df
test_df = preprocess_data(test_df)

# Preprocess train_df
train_df = preprocess_data(train_df)
# print(train_df.info())


# for ease of modeling, change them all to float32 variables first
def convert_to_float32(df):
    df = df.astype('float32')
    return df

test_df = convert_to_float32(test_df)
train_df = convert_to_float32(train_df)

# # verify all missing values in train_df
# missing_values_train = train_df.isnull().sum()
# print("Missing values in train_df:")
# print(missing_values_train)

# # Check for missing values in test_df
missing_values_test = test_df.isnull().sum()
print("\nMissing values in test_df:")
print(missing_values_test)

x_train = train_df.drop(columns=['SalePrice'])
y_train = train_df['SalePrice']

model = HousePrice()
model.train_ridge(x_train, y_train)

x_test = test_df
y_test_pred = model.pred_ridge(x_test)

output = pd.DataFrame({'Id': test_df_id.astype('int'), 'SalePrice': y_test_pred})
output.to_csv('submission.csv', index=False)
