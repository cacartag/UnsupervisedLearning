import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# Set styling
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


# Load the data
df = pd.read_excel("datasets/online_retail.xlsx")
print("Dataset shape:", df.shape)
print("\nDataset info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())




## Data preprocesssing and EDA

# Clean the data
print("Original dataset shape:", df.shape)

# Remove missing CustomerIDs
df_clean = df.dropna(subset=['CustomerID'])

# Remove cancelled orders (negative quantities)
df_clean = df_clean[df_clean['Quantity'] > 0]

# Remove orders with zero price
df_clean = df_clean[df_clean['UnitPrice'] > 0]

# Create TotolAmount 
df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['UnitPrice']

# Convert InvoiceDate to datetime
df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])

print("Cleaned dataset shape:", df_clean.shape)

# Basic statistics
print("\n Basic Statistics:")
print(df_clean[['Quantity', 'UnitPrice', 'TotalAmount']].describe())

# Plot some exploratory visualizations
# fig, axes = plt.subplots(2,2, figsize=(15, 10))

# # Sales over time
# monthly_sales = df_clean.groupby(df_clean['InvoiceDate'].dt.to_period('M'))['TotalAmount'].sum()
# monthly_sales.plot(ax=axes[0,0], title='Monthly Sales Trend')
# axes[0,0].set_ylabel('Total Sales (£)')

# # Top countries by sales
# country_sales = df_clean.groupby('Country')['TotalAmount'].sum().sort_values(ascending=False).head(10)
# country_sales.plot(kind='bar', ax=axes[0,1], title='Top 10 Countries by Sales')
# axes[0,0].set_ylabel('Total Sales (£)')
# axes[0,1].tick_params(axis='x', rotation=45)

# # Distribution of order values - BEST VERSION
# # Use percentile-based filtering for more realistic view
# q95 = df_clean['TotalAmount'].quantile(0.95)  # 95th percentile
# order_values_filtered = df_clean[df_clean['TotalAmount'] <= q95]['TotalAmount']

# axes[1,0].hist(order_values_filtered, bins=50, alpha=0.7, color='lightcoral')
# axes[1,0].set_title(f'Distribution of Order Values (up to 95th percentile: £{q95:.2f})')
# axes[1,0].set_xlabel('Order Value (£)')
# axes[1,0].set_ylabel('Number of Orders')
# axes[1,0].grid(True, alpha=0.3)

# # Top products
# top_products = df_clean.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
# top_products.plot(kind='barh', ax=axes[1,1], title='Top 10 Products by Quantity')
# axes[1,1].set_xlabel('Total Quantity Sold')

# plt.tight_layout()
# plt.show()


# Calculate RFM (Recency, Frequency, Monetary) metrics for each customer
# Define analysis date (day after last transaction)
analysis_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days=1)


# Create RFM table
rfm_table = df_clean.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (analysis_date - x.max()).days, # Recency
    'InvoiceNo': 'count', # Frequency
    'TotalAmount': 'sum'  # Monetary
}).round(2)

# Rename columns
rfm_table.columns = ['Recency', 'Frequency', 'Monetary']

# Remove outliers using IQR method
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Apply outlier removal 
for col in ['Recency', 'Frequency', 'Monetary']:
    rfm_table = remove_outliers(rfm_table, col)

print("RFM Table shape after cleaning:", rfm_table.shape)
print("\nRFM Statistics:")
print(rfm_table.describe())



# Visualize RFM distributions
# fig, axes = plt.subplots(1,3, figsize=(15, 5))

# rfm_table['Recency'].hist(bins=50, ax=axes[0], alpha=0.7)
# axes[0].set_title('Recency Distribution')
# axes[0].set_xlabel('Days since last purchase')


# rfm_table['Frequency'].hist(bins=50, ax=axes[1], alpha=0.7)
# axes[1].set_title('Frequency Distribution')
# axes[1].set_xlabel('Number of purchases')

# rfm_table['Monetary'].hist(bins=50, ax=axes[2], alpha=0.7)
# axes[2].set_title('Monetary Distribution')
# axes[2].set_xlabel('Total amount spent (£)')

# plt.tight_layout()
# plt.show()





























