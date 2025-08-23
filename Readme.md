# Real-World Unsupervised Learning Tutorial with Actual Data

## Overview
This tutorial uses real datasets to demonstrate unsupervised learning techniques. Each case study includes complete code that you can run with actual data downloads.

## Prerequisites & Setup

```python
# Install required packages
# pip install pandas numpy matplotlib seaborn scikit-learn plotly umap-learn wordcloud

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
```

---

# Case Study 1: Customer Segmentation with Online Retail Data

## Dataset: UCI Online Retail Dataset
**Source**: https://archive.ics.uci.edu/ml/datasets/Online+Retail
**Description**: Real transactions from UK-based online retailer (2010-2011)

### Download and Load Data

```python
# Download the dataset
import urllib.request

# Download Online Retail dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
urllib.request.urlretrieve(url, "online_retail.xlsx")

# Load the data
df = pd.read_excel("online_retail.xlsx")
print("Dataset shape:", df.shape)
print("\nDataset info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())
```

### Data Preprocessing and EDA

```python
# Clean the data
print("Original dataset shape:", df.shape)

# Remove missing CustomerIDs
df_clean = df.dropna(subset=['CustomerID'])

# Remove cancelled orders (negative quantities)
df_clean = df_clean[df_clean['Quantity'] > 0]

# Remove orders with zero price
df_clean = df_clean[df_clean['UnitPrice'] > 0]

# Create TotalAmount column
df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['UnitPrice']

# Convert InvoiceDate to datetime
df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])

print("Cleaned dataset shape:", df_clean.shape)

# Basic statistics
print("\nBasic Statistics:")
print(df_clean[['Quantity', 'UnitPrice', 'TotalAmount']].describe())

# Plot some exploratory visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Sales over time
monthly_sales = df_clean.groupby(df_clean['InvoiceDate'].dt.to_period('M'))['TotalAmount'].sum()
monthly_sales.plot(ax=axes[0,0], title='Monthly Sales Trend')
axes[0,0].set_ylabel('Total Sales (£)')

# Top countries by sales
country_sales = df_clean.groupby('Country')['TotalAmount'].sum().sort_values(ascending=False).head(10)
country_sales.plot(kind='bar', ax=axes[0,1], title='Top 10 Countries by Sales')
axes[0,1].set_ylabel('Total Sales (£)')
axes[0,1].tick_params(axis='x', rotation=45)

# Distribution of order values
axes[1,0].hist(df_clean['TotalAmount'], bins=50, alpha=0.7)
axes[1,0].set_title('Distribution of Order Values')
axes[1,0].set_xlabel('Order Value (£)')
axes[1,0].set_xlim(0, 100)  # Focus on main distribution

# Top products
top_products = df_clean.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
top_products.plot(kind='barh', ax=axes[1,1], title='Top 10 Products by Quantity')
axes[1,1].set_xlabel('Total Quantity Sold')

plt.tight_layout()
plt.show()
```

### RFM Analysis (Recency, Frequency, Monetary)

```python
# Calculate RFM metrics for each customer
# Define analysis date (day after last transaction)
analysis_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days=1)

# Create RFM table
rfm_table = df_clean.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (analysis_date - x.max()).days,  # Recency
    'InvoiceNo': 'count',  # Frequency  
    'TotalAmount': 'sum'   # Monetary
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
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

rfm_table['Recency'].hist(bins=50, ax=axes[0], alpha=0.7)
axes[0].set_title('Recency Distribution')
axes[0].set_xlabel('Days since last purchase')

rfm_table['Frequency'].hist(bins=50, ax=axes[1], alpha=0.7)
axes[1].set_title('Frequency Distribution') 
axes[1].set_xlabel('Number of purchases')

rfm_table['Monetary'].hist(bins=50, ax=axes[2], alpha=0.7)
axes[2].set_title('Monetary Distribution')
axes[2].set_xlabel('Total amount spent (£)')

plt.tight_layout()
plt.show()

# Correlation matrix
plt.figure(figsize=(8, 6))
correlation_matrix = rfm_table.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('RFM Correlation Matrix')
plt.show()
```

### Customer Segmentation with K-Means

```python
# Prepare data for clustering
# Log transform monetary values to handle skewness
rfm_log = rfm_table.copy()
rfm_log['Monetary'] = np.log1p(rfm_log['Monetary'])

# Standardize the features
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)

# Find optimal number of clusters using elbow method and silhouette score
inertias = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(rfm_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(rfm_scaled, kmeans.labels_))

# Plot elbow curve and silhouette scores
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.plot(k_range, inertias, 'bo-')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method')
ax1.grid(True)

ax2.plot(k_range, silhouette_scores, 'ro-')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Analysis')
ax2.grid(True)

plt.tight_layout()
plt.show()

# Choose optimal k (let's use 4 based on the plots)
optimal_k = 4
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(rfm_scaled)

# Add cluster labels to RFM table
rfm_table['Cluster'] = cluster_labels

print(f"Silhouette Score for k={optimal_k}: {silhouette_score(rfm_scaled, cluster_labels):.3f}")
print(f"Calinski-Harabasz Score: {calinski_harabasz_score(rfm_scaled, cluster_labels):.3f}")
```

### Analyze and Visualize Customer Segments

```python
# Analyze cluster characteristics
cluster_summary = rfm_table.groupby('Cluster').agg({
    'Recency': ['mean', 'median'],
    'Frequency': ['mean', 'median'], 
    'Monetary': ['mean', 'median'],
    'CustomerID': 'count'
}).round(2)

cluster_summary.columns = ['Recency_Mean', 'Recency_Median', 'Frequency_Mean', 
                          'Frequency_Median', 'Monetary_Mean', 'Monetary_Median', 'Count']

print("Cluster Summary:")
print(cluster_summary)

# Create meaningful cluster names based on characteristics
cluster_names = {
    0: "Need Attention", 
    1: "Loyal Customers",
    2: "Big Spenders", 
    3: "New Customers"
}

# You may need to adjust these names based on your actual results
rfm_table['Cluster_Name'] = rfm_table['Cluster'].map(cluster_names)

# Visualize clusters
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 3D scatter plot of RFM values colored by cluster
scatter = axes[0,0].scatter(rfm_table['Recency'], rfm_table['Frequency'], 
                           c=rfm_table['Cluster'], cmap='viridis', alpha=0.6)
axes[0,0].set_xlabel('Recency (days)')
axes[0,0].set_ylabel('Frequency')
axes[0,0].set_title('Customer Segments: Recency vs Frequency')
plt.colorbar(scatter, ax=axes[0,0])

# Frequency vs Monetary
scatter2 = axes[0,1].scatter(rfm_table['Frequency'], rfm_table['Monetary'], 
                            c=rfm_table['Cluster'], cmap='viridis', alpha=0.6)
axes[0,1].set_xlabel('Frequency')
axes[0,1].set_ylabel('Monetary Value (£)')
axes[0,1].set_title('Customer Segments: Frequency vs Monetary')
plt.colorbar(scatter2, ax=axes[0,1])

# Recency vs Monetary  
scatter3 = axes[1,0].scatter(rfm_table['Recency'], rfm_table['Monetary'], 
                            c=rfm_table['Cluster'], cmap='viridis', alpha=0.6)
axes[1,0].set_xlabel('Recency (days)')
axes[1,0].set_ylabel('Monetary Value (£)')
axes[1,0].set_title('Customer Segments: Recency vs Monetary')
plt.colorbar(scatter3, ax=axes[1,0])

# Cluster sizes
cluster_counts = rfm_table['Cluster'].value_counts().sort_index()
axes[1,1].pie(cluster_counts.values, labels=[f'Cluster {i}' for i in cluster_counts.index], 
              autopct='%1.1f%%')
axes[1,1].set_title('Customer Segment Distribution')

plt.tight_layout()
plt.show()

# Detailed cluster analysis
print("\nDetailed Cluster Analysis:")
for cluster in sorted(rfm_table['Cluster'].unique()):
    cluster_data = rfm_table[rfm_table['Cluster'] == cluster]
    print(f"\n--- Cluster {cluster} ({cluster_names.get(cluster, 'Unknown')}) ---")
    print(f"Size: {len(cluster_data)} customers ({len(cluster_data)/len(rfm_table)*100:.1f}%)")
    print(f"Average Recency: {cluster_data['Recency'].mean():.1f} days")
    print(f"Average Frequency: {cluster_data['Frequency'].mean():.1f} purchases")  
    print(f"Average Monetary: £{cluster_data['Monetary'].mean():.2f}")
    print(f"Total Revenue: £{cluster_data['Monetary'].sum():.2f}")
```

### PCA Visualization

```python
# Apply PCA for visualization
pca = PCA(n_components=2)
rfm_pca = pca.fit_transform(rfm_scaled)

# Create PCA plot
plt.figure(figsize=(12, 8))
scatter = plt.scatter(rfm_pca[:, 0], rfm_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
plt.xlabel(f'First Principal Component (explains {pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'Second Principal Component (explains {pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Customer Segments in PCA Space')
plt.colorbar(scatter)

# Add cluster centers in PCA space
centers_pca = pca.transform(kmeans_final.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], marker='x', s=200, 
           linewidths=3, color='red', label='Centroids')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Show PCA components
components_df = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2'], 
    index=['Recency', 'Frequency', 'Monetary']
)

print("PCA Components (Feature Loadings):")
print(components_df)
print(f"\nTotal variance explained: {pca.explained_variance_ratio_.sum():.1%}")
```

---

# Case Study 2: Stock Market Clustering

## Dataset: S&P 500 Stock Data
We'll use the `yfinance` library to get real stock data.

```python
# Install yfinance if not already installed
# pip install yfinance

import yfinance as yf
from datetime import datetime, timedelta

# Get S&P 500 stock symbols (sample of 50 stocks for demonstration)
sp500_symbols = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V',
    'PG', 'UNH', 'HD', 'MA', 'BAC', 'ABBV', 'PFE', 'KO', 'AVGO', 'PEP',
    'TMO', 'COST', 'WMT', 'DIS', 'ABT', 'CRM', 'ACN', 'VZ', 'ADBE', 'NFLX',
    'NKE', 'DHR', 'LIN', 'TXN', 'BMY', 'PM', 'T', 'HON', 'UPS', 'QCOM',
    'LOW', 'AMD', 'SPGI', 'C', 'AMGN', 'CVX', 'NEE', 'IBM', 'BA', 'GS'
]

# Download stock data for the past year
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

print("Downloading stock data...")
stock_data = {}
failed_downloads = []

for symbol in sp500_symbols:
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        if not data.empty:
            stock_data[symbol] = data
        else:
            failed_downloads.append(symbol)
    except Exception as e:
        print(f"Failed to download {symbol}: {e}")
        failed_downloads.append(symbol)

print(f"Successfully downloaded data for {len(stock_data)} stocks")
if failed_downloads:
    print(f"Failed downloads: {failed_downloads}")
```

### Calculate Stock Features for Clustering

```python
# Calculate features for each stock
stock_features = []

for symbol, data in stock_data.items():
    if len(data) < 30:  # Skip stocks with insufficient data
        continue
        
    # Calculate returns
    data['Daily_Return'] = data['Close'].pct_change()
    
    # Calculate features
    features = {
        'Symbol': symbol,
        'Total_Return': (data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1,
        'Volatility': data['Daily_Return'].std(),
        'Average_Volume': data['Volume'].mean(),
        'Price_Range': (data['High'].max() - data['Low'].min()) / data['Close'].mean(),
        'Trend_Slope': np.polyfit(range(len(data)), data['Close'], 1)[0] / data['Close'].mean(),
        'Max_Drawdown': ((data['Close'] / data['Close'].expanding().max()) - 1).min(),
        'Volume_Volatility': data['Volume'].std() / data['Volume'].mean(),
        'Beta': data['Daily_Return'].cov(data['Daily_Return'].mean()) / data['Daily_Return'].var() if data['Daily_Return'].var() > 0 else 0
    }
    
    stock_features.append(features)

# Create DataFrame
stocks_df = pd.DataFrame(stock_features)
print("Stock features shape:", stocks_df.shape)
print("\nFeature statistics:")
print(stocks_df.describe())

# Handle any infinite or NaN values
stocks_df = stocks_df.replace([np.inf, -np.inf], np.nan)
stocks_df = stocks_df.fillna(stocks_df.median())

print("\nFinal dataset shape:", stocks_df.shape)
```

### Stock Market Clustering Analysis

```python
# Prepare features for clustering (exclude Symbol column)
feature_columns = [col for col in stocks_df.columns if col != 'Symbol']
X_stocks = stocks_df[feature_columns].values

# Standardize features
scaler_stocks = StandardScaler()
X_stocks_scaled = scaler_stocks.fit_transform(X_stocks)

# Find optimal number of clusters
inertias_stocks = []
silhouette_scores_stocks = []
k_range_stocks = range(2, 8)

for k in k_range_stocks:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_stocks_scaled)
    inertias_stocks.append(kmeans.inertia_)
    if k > 1:  # Silhouette score requires at least 2 clusters
        silhouette_scores_stocks.append(silhouette_score(X_stocks_scaled, labels))

# Plot evaluation metrics
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.plot(k_range_stocks, inertias_stocks, 'bo-')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method - Stock Clustering')
ax1.grid(True)

ax2.plot(k_range_stocks[1:], silhouette_scores_stocks, 'ro-')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Analysis - Stock Clustering')
ax2.grid(True)

plt.tight_layout()
plt.show()

# Apply final clustering
optimal_k_stocks = 4  # Choose based on the plots above
kmeans_stocks = KMeans(n_clusters=optimal_k_stocks, random_state=42, n_init=10)
stock_clusters = kmeans_stocks.fit_predict(X_stocks_scaled)

# Add cluster labels to dataframe
stocks_df['Cluster'] = stock_clusters

print(f"Stock clustering silhouette score: {silhouette_score(X_stocks_scaled, stock_clusters):.3f}")
```

### Analyze Stock Clusters

```python
# Analyze cluster characteristics
cluster_summary_stocks = stocks_df.groupby('Cluster')[feature_columns].mean().round(4)
print("Stock Cluster Summary:")
print(cluster_summary_stocks)

# Count stocks per cluster
cluster_counts_stocks = stocks_df['Cluster'].value_counts().sort_index()
print(f"\nStocks per cluster:")
for cluster, count in cluster_counts_stocks.items():
    print(f"Cluster {cluster}: {count} stocks")

# Show stocks in each cluster
print("\nStocks by Cluster:")
for cluster in sorted(stocks_df['Cluster'].unique()):
    cluster_stocks = stocks_df[stocks_df['Cluster'] == cluster]['Symbol'].tolist()
    print(f"Cluster {cluster}: {', '.join(cluster_stocks)}")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Total Return vs Volatility
scatter1 = axes[0,0].scatter(stocks_df['Total_Return'], stocks_df['Volatility'], 
                           c=stocks_df['Cluster'], cmap='viridis', alpha=0.7, s=60)
axes[0,0].set_xlabel('Total Return')
axes[0,0].set_ylabel('Volatility')
axes[0,0].set_title('Stock Clusters: Return vs Volatility')
plt.colorbar(scatter1, ax=axes[0,0])

# Volume vs Price Range
scatter2 = axes[0,1].scatter(stocks_df['Average_Volume'], stocks_df['Price_Range'], 
                           c=stocks_df['Cluster'], cmap='viridis', alpha=0.7, s=60)
axes[0,1].set_xlabel('Average Volume')
axes[0,1].set_ylabel('Price Range')
axes[0,1].set_title('Stock Clusters: Volume vs Price Range')
axes[0,1].set_xscale('log')  # Log scale for volume
plt.colorbar(scatter2, ax=axes[0,1])

# Trend vs Max Drawdown
scatter3 = axes[1,0].scatter(stocks_df['Trend_Slope'], stocks_df['Max_Drawdown'], 
                           c=stocks_df['Cluster'], cmap='viridis', alpha=0.7, s=60)
axes[1,0].set_xlabel('Trend Slope')
axes[1,0].set_ylabel('Max Drawdown') 
axes[1,0].set_title('Stock Clusters: Trend vs Max Drawdown')
plt.colorbar(scatter3, ax=axes[1,0])

# Cluster distribution
cluster_counts_stocks.plot(kind='bar', ax=axes[1,1], color='skyblue')
axes[1,1].set_xlabel('Cluster')
axes[1,1].set_ylabel('Number of Stocks')
axes[1,1].set_title('Stock Distribution by Cluster')
axes[1,1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.show()
```

### PCA for Stock Data

```python
# Apply PCA to stock data
pca_stocks = PCA(n_components=2)
stocks_pca = pca_stocks.fit_transform(X_stocks_scaled)

# Create PCA visualization
plt.figure(figsize=(12, 8))
scatter = plt.scatter(stocks_pca[:, 0], stocks_pca[:, 1], c=stock_clusters, cmap='viridis', alpha=0.7, s=60)

# Add stock labels
for i, symbol in enumerate(stocks_df['Symbol']):
    plt.annotate(symbol, (stocks_pca[i, 0], stocks_pca[i, 1]), xytext=(5, 5), 
                textcoords='offset points', fontsize=8, alpha=0.7)

plt.xlabel(f'First Principal Component (explains {pca_stocks.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'Second Principal Component (explains {pca_stocks.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Stock Clusters in PCA Space')
plt.colorbar(scatter)
plt.grid(True, alpha=0.3)
plt.show()

# Show feature loadings
loadings_df = pd.DataFrame(
    pca_stocks.components_.T,
    columns=['PC1', 'PC2'],
    index=feature_columns
)

print("PCA Feature Loadings:")
print(loadings_df.round(3))
print(f"\nTotal variance explained: {pca_stocks.explained_variance_ratio_.sum():.1%}")
```

---

# Case Study 3: News Article Clustering with Text Data

## Dataset: 20 Newsgroups (Sample)
We'll use scikit-learn's built-in newsgroups dataset.

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import re
import string

# Download a subset of categories for faster processing
categories = [
    'alt.atheism',
    'comp.graphics', 
    'comp.os.ms-windows.misc',
    'comp.sys.ibm.pc.hardware',
    'rec.autos',
    'rec.motorcycles', 
    'sci.electronics',
    'sci.med',
    'sci.space',
    'talk.politics.misc'
]

print("Loading 20 newsgroups dataset...")
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, 
                                     shuffle=True, random_state=42, 
                                     remove=('headers', 'footers', 'quotes'))

print(f"Dataset loaded: {len(newsgroups_train.data)} documents")
print(f"Categories: {newsgroups_train.target_names}")
```

### Text Preprocessing and Feature Extraction

```python
def preprocess_text(text):
    """Clean and preprocess text data"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and digits
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# Preprocess all documents
print("Preprocessing text data...")
processed_docs = [preprocess_text(doc) for doc in newsgroups_train.data]

# Create TF-IDF features
print("Creating TF-IDF features...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=1000,  # Limit features for faster processing
    min_df=5,          # Ignore terms that appear in less than 5 documents
    max_df=0.7,        # Ignore terms that appear in more than 70% of documents
    stop_words='english',
    ngram_range=(1, 2)  # Use both unigrams and bigrams
)

tfidf_matrix = tfidf_vectorizer.fit_transform(processed_docs)
feature_names = tfidf_vectorizer.get_feature_names_out()

print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
print(f"Number of features: {len(feature_names)}")
```

### Dimensionality Reduction with TruncatedSVD

```python
# Apply TruncatedSVD (similar to PCA for sparse matrices)
print("Applying dimensionality reduction...")
n_components = 100
svd = TruncatedSVD(n_components=n_components, random_state=42)
tfidf_reduced = svd.fit_transform(tfidf_matrix)

print(f"Reduced dimensions: {tfidf_reduced.shape}")
print(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.3f}")

# Plot explained variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_components + 1), np.cumsum(svd.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Explained Variance by SVD Components')
plt.grid(True)
plt.show()
```

### Text Clustering

```python
# Find optimal number of clusters for text data
inertias_text = []
silhouette_scores_text = []
k_range_text = range(2, 12)

print("Finding optimal number of clusters...")
for k in k_range_text:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=100)
    labels = kmeans.fit_predict(tfidf_reduced)
    inertias_text.append(kmeans.inertia_)
    silhouette_scores_text.append(silhouette_score(tfidf_reduced, labels))

# Plot evaluation metrics
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.plot(k_range_text, inertias_text, 'bo-')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia') 
ax1.set_title('Elbow Method - Text Clustering')
ax1.grid(True)

ax2.plot(k_range_text, silhouette_scores_text, 'ro-')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Analysis - Text Clustering')
ax2.grid(True)

plt.tight_layout()
plt.show()

# Apply final clustering
optimal_k_text = 10  # Should match number of categories
kmeans_text = KMeans(n_clusters=optimal_k_text, random_state=42, n_init=10, max_iter=100)
text_clusters = kmeans_text.fit_predict(tfidf_reduced)

print(f"Text clustering silhouette score: {silhouette_score(tfidf_reduced, text_clusters):.3f}")

# Create DataFrame for analysis
text_df = pd.DataFrame({
    'Document': range(len(processed_docs)),
    'True_Category': newsgroups_train.target,
    'True_Category_Name': [newsgroups_train.target_names[i] for i in newsgroups_train.target],
    'Predicted_Cluster': text_clusters
})

print("\nCluster distribution:")
print(text_df['Predicted_Cluster'].value_counts().sort_index())
```

### Analyze Text Clusters

```python
# Analyze cluster characteristics by finding top terms
def get_top_terms_per_cluster(tfidf_matrix, cluster_labels, feature_names, top_k=10):
    """Get top terms for each cluster"""
    cluster_terms = {}
    
    for cluster_id in np.unique(cluster_labels):
        # Get documents in this cluster
        cluster_docs = tfidf_matrix[cluster_labels == cluster_id]
        
        # Calculate mean TF-IDF scores for this cluster
        mean_scores = np.array(cluster_docs.mean(axis=0)).flatten()
        
        # Get top terms
        top_indices = mean_scores.argsort()[-top_k:][::-1]
        top_terms = [(feature_names[i], mean_scores[i]) for i in top_indices]
        
        cluster_terms[cluster_id] = top_terms
    
    return cluster_terms

# Get top terms for each cluster
cluster_terms = get_top_terms_per_cluster(tfidf_matrix, text_clusters, feature_names)

print("Top terms per cluster:")
for cluster_id, terms in cluster_terms.items():
    print(f"\nCluster {cluster_id}:")
    for term, score in terms:
        print(f"  {term}: {score:.4f}")

# Compare with true categories
print("\nCluster vs True Category Analysis:")
for cluster_id in range(optimal_k_text):
    cluster_data = text_df[text_df['Predicted_Cluster'] == cluster_id]
    print(f"\nCluster {cluster_id} ({len(cluster_data)} documents):")
    category_dist = cluster_data['True_Category_Name'].value_counts()
    for category, count in category_dist.head().items():
        print(f"  {category}: {count} ({count/len(cluster_data)*100:.1f}%)")
```

### Visualize Text Clusters

```python
# Further reduce to 2D for visualization
svd_2d = TruncatedSVD(n_components=2, random_state=42)
tfidf_2d = svd_2d.fit_transform(tfidf_matrix)

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Plot by predicted clusters
scatter1 = ax1.scatter(tfidf_2d[:, 0], tfidf_2d[:, 1], c=text_clusters, cmap='tab10', alpha=0.6)
ax1.set_xlabel(f'First SVD Component (explains {svd_2d.explained_variance_ratio_[0]:.1%} variance)')
ax1.set_ylabel(f'Second SVD Component (explains {svd_2d.explained_variance_ratio_[1]:.1%} variance)')
ax1.set_title('News Articles: Predicted Clusters')
plt.colorbar(scatter1, ax=ax1)

# Plot by true categories
scatter2 = ax2.scatter(tfidf_2d[:, 0], tfidf_2d[:, 1], c=newsgroups_train.target, cmap='tab10', alpha=0.6)
ax2.set_xlabel(f'First SVD Component (explains {svd_2d.explained_variance_ratio_[0]:.1%} variance)')
ax2.set_ylabel(f'Second SVD Component (explains {svd_2d.explained_variance_ratio_[1]:.1%} variance)')
ax2.set_title('News Articles: True Categories')
plt.colorbar(scatter2, ax=ax2)

plt.tight_layout()
plt.show()

# Calculate clustering accuracy (using Hungarian algorithm for best matching)
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

def cluster_accuracy(y_true, y_pred):
    """Calculate clustering accuracy using Hungarian algorithm"""
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Use Hungarian algorithm to find best assignment
    row_ind, col_ind = linear_sum_assignment(-cm)
    
    # Calculate accuracy
    accuracy = cm[row_ind, col_ind].sum() / cm.sum()
    return accuracy, row_ind, col_ind

accuracy, _, _ = cluster_accuracy(newsgroups_train.target, text_clusters)
print(f"\nClustering accuracy: {accuracy:.3f}")

# Show confusion matrix
cm = confusion_matrix(newsgroups_train.target, text_clusters)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f'Cluster {i}' for i in range(optimal_k_text)],
            yticklabels=newsgroups_train.target_names)
plt.title('Confusion Matrix: True Categories vs Predicted Clusters')
plt.xlabel('Predicted Cluster')
plt.ylabel('True Category')
plt.show()
```

---

# Case Study 4: Market Basket Analysis with Retail Data

## Dataset: Online Retail Dataset (Association Rules)
We'll use the same retail dataset but focus on association rule mining.

```python
# Use the previously loaded retail dataset
# If not already loaded, run the data loading code from Case Study 1

print("Preparing data for market basket analysis...")

# Create a basket format dataset
basket_data = df_clean.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)

# Convert to binary format (1 if item was purchased, 0 otherwise)
basket_binary = basket_data.applymap(lambda x: 1 if x > 0 else 0)

print(f"Basket data shape: {basket_binary.shape}")
print(f"Number of transactions: {len(basket_binary)}")
print(f"Number of unique products: {len(basket_binary.columns)}")

# Remove transactions with very few items (likely returns or errors)
basket_binary = basket_binary[basket_binary.sum(axis=1) >= 2]
print(f"After filtering: {len(basket_binary)} transactions")

# Keep only top N products by frequency to make analysis manageable
top_n_products = 50
product_frequency = basket_binary.sum().sort_values(ascending=False).head(top_n_products)
basket_filtered = basket_binary[product_frequency.index]

print(f"Using top {top_n_products} products")
print("Top 10 most frequent products:")
print(product_frequency.head(10))
```

### Association Rule Mining

```python
# Install mlxtend if not already installed
# pip install mlxtend

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

print("Finding frequent itemsets...")

# Find frequent itemsets with minimum support
min_support = 0.01  # Items must appear in at least 1% of transactions
frequent_itemsets = apriori(basket_filtered, min_support=min_support, use_colnames=True)

print(f"Found {len(frequent_itemsets)} frequent itemsets")
print("Top 10 frequent itemsets:")
print(frequent_itemsets.head(10))

# Generate association rules
print("Generating association rules...")
min_confidence = 0.3  # Minimum confidence for rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

print(f"Found {len(rules)} association rules")

# Sort by confidence and lift
rules_sorted = rules.sort_values(['confidence', 'lift'], ascending=False)
print("Top 20 association rules:")
print(rules_sorted[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(20))
```

### Visualize Association Rules

```python
# Create visualizations for association rules
fig, axes = plt.subplots(2, 2, figsize=(20, 15))

# Support vs Confidence
scatter1 = axes[0,0].scatter(rules['support'], rules['confidence'], 
                            c=rules['lift'], cmap='viridis', alpha=0.6, s=50)
axes[0,0].set_xlabel('Support')
axes[0,0].set_ylabel('Confidence')
axes[0,0].set_title('Association Rules: Support vs Confidence (colored by Lift)')
plt.colorbar(scatter1, ax=axes[0,0])

# Support vs Lift
scatter2 = axes[0,1].scatter(rules['support'], rules['lift'], 
                            c=rules['confidence'], cmap='plasma', alpha=0.6, s=50)
axes[0,1].set_xlabel('Support')
axes[0,1].set_ylabel('Lift')
axes[0,1].set_title('Association Rules: Support vs Lift (colored by Confidence)')
plt.colorbar(scatter2, ax=axes[0,1])

# Confidence vs Lift
scatter3 = axes[1,0].scatter(rules['confidence'], rules['lift'], 
                            c=rules['support'], cmap='coolwarm', alpha=0.6, s=50)
axes[1,0].set_xlabel('Confidence')
axes[1,0].set_ylabel('Lift')
axes[1,0].set_title('Association Rules: Confidence vs Lift (colored by Support)')
plt.colorbar(scatter3, ax=axes[1,0])

# Distribution of lift values
axes[1,1].hist(rules['lift'], bins=30, alpha=0.7, color='skyblue')
axes[1,1].set_xlabel('Lift')
axes[1,1].set_ylabel('Frequency')
axes[1,1].set_title('Distribution of Lift Values')
axes[1,1].axvline(x=1, color='red', linestyle='--', label='Lift = 1 (Independence)')
axes[1,1].legend()

plt.tight_layout()
plt.show()

# Show interesting rules (high confidence and lift)
interesting_rules = rules[(rules['confidence'] > 0.6) & (rules['lift'] > 2)]
print(f"\nHighly interesting rules (confidence > 0.6, lift > 2): {len(interesting_rules)}")

if len(interesting_rules) > 0:
    print("Most interesting association rules:")
    for _, rule in interesting_rules.head(10).iterrows():
        antecedent = list(rule['antecedents'])[0] if len(rule['antecedents']) == 1 else list(rule['antecedents'])
        consequent = list(rule['consequents'])[0] if len(rule['consequents']) == 1 else list(rule['consequents'])
        print(f"If customer buys {antecedent}")
        print(f"  → then they will also buy {consequent}")
        print(f"  Support: {rule['support']:.3f}, Confidence: {rule['confidence']:.3f}, Lift: {rule['lift']:.3f}")
        print()
```

### Network Analysis of Product Associations

```python
# Create a network graph of product associations
import networkx as nx

# Create network graph
G = nx.Graph()

# Add edges for strong associations (high confidence and lift)
strong_rules = rules[(rules['confidence'] > 0.5) & (rules['lift'] > 1.5)]

for _, rule in strong_rules.iterrows():
    for antecedent in rule['antecedents']:
        for consequent in rule['consequents']:
            # Add edge with weight based on lift
            if G.has_edge(antecedent, consequent):
                G[antecedent][consequent]['weight'] += rule['lift']
            else:
                G.add_edge(antecedent, consequent, weight=rule['lift'])

print(f"Network has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# Visualize network (for smaller networks)
if G.number_of_nodes() <= 30:  # Only visualize if not too many nodes
    plt.figure(figsize=(15, 10))
    
    # Use spring layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw edges with width based on weight
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    nx.draw_networkx_edges(G, pos, width=[w/max(weights)*3 for w in weights], alpha=0.6)
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue', alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    plt.title('Product Association Network')
    plt.axis('off')
    plt.show()

# Calculate network metrics
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)

print("Top 10 products by degree centrality (most connected):")
sorted_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
for product, centrality in sorted_degree[:10]:
    print(f"  {product}: {centrality:.3f}")

print("\nTop 10 products by betweenness centrality (bridges different product groups):")
sorted_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)
for product, centrality in sorted_betweenness[:10]:
    print(f"  {product}: {centrality:.3f}")
```

---

# Business Insights and Applications

## Customer Segmentation Insights

```python
# Print business insights for customer segmentation
print("=== CUSTOMER SEGMENTATION BUSINESS INSIGHTS ===")
print()

if 'rfm_table' in locals():
    for cluster in sorted(rfm_table['Cluster'].unique()):
        cluster_data = rfm_table[rfm_table['Cluster'] == cluster]
        cluster_size = len(cluster_data)
        cluster_revenue = cluster_data['Monetary'].sum()
        avg_recency = cluster_data['Recency'].mean()
        avg_frequency = cluster_data['Frequency'].mean()
        avg_monetary = cluster_data['Monetary'].mean()
        
        print(f"CLUSTER {cluster}:")
        print(f"  Size: {cluster_size} customers ({cluster_size/len(rfm_table)*100:.1f}% of total)")
        print(f"  Total Revenue: £{cluster_revenue:,.2f} ({cluster_revenue/rfm_table['Monetary'].sum()*100:.1f}% of total)")
        print(f"  Customer Profile:")
        print(f"    - Average days since last purchase: {avg_recency:.0f}")
        print(f"    - Average purchase frequency: {avg_frequency:.1f}")
        print(f"    - Average customer value: £{avg_monetary:,.2f}")
        
        # Business recommendations
        if avg_recency < 30 and avg_frequency > 5:
            print(f"  📈 RECOMMENDATION: VIP/Champions - Reward program, exclusive offers")
        elif avg_recency < 30 and avg_frequency <= 5:
            print(f"  🎯 RECOMMENDATION: New Customers - Welcome series, onboarding")
        elif avg_recency >= 30 and avg_frequency > 5:
            print(f"  ⚠️  RECOMMENDATION: At Risk - Win-back campaigns, special discounts")
        else:
            print(f"  🔄 RECOMMENDATION: Need Attention - Re-engagement campaigns")
        print()

print("=== STOCK CLUSTERING BUSINESS INSIGHTS ===")
print()

if 'stocks_df' in locals():
    for cluster in sorted(stocks_df['Cluster'].unique()):
        cluster_stocks = stocks_df[stocks_df['Cluster'] == cluster]
        avg_return = cluster_stocks['Total_Return'].mean()
        avg_volatility = cluster_stocks['Volatility'].mean()
        
        print(f"STOCK CLUSTER {cluster}:")
        print(f"  Stocks: {', '.join(cluster_stocks['Symbol'].tolist())}")
        print(f"  Profile: Avg Return = {avg_return*100:.1f}%, Avg Volatility = {avg_volatility*100:.1f}%")
        
        # Investment recommendations
        if avg_return > 0.1 and avg_volatility < 0.02:
            print(f"  💎 RECOMMENDATION: High-quality growth stocks - Core portfolio holdings")
        elif avg_return > 0.05 and avg_volatility < 0.03:
            print(f"  📈 RECOMMENDATION: Stable performers - Good for conservative investors")
        elif avg_volatility > 0.04:
            print(f"  ⚡ RECOMMENDATION: High volatility - Suitable for risk-tolerant investors")
        else:
            print(f"  🔍 RECOMMENDATION: Mixed performance - Requires individual analysis")
        print()

print("=== MARKET BASKET ANALYSIS BUSINESS INSIGHTS ===")
print()

if 'rules' in locals() and len(rules) > 0:
    print("KEY ACTIONABLE INSIGHTS:")
    
    # Top cross-selling opportunities
    top_cross_sell = rules.nlargest(5, 'confidence')
    print("🛒 TOP CROSS-SELLING OPPORTUNITIES:")
    for _, rule in top_cross_sell.iterrows():
        antecedent = list(rule['antecedents'])[0] if len(rule['antecedents']) == 1 else str(list(rule['antecedents']))
        consequent = list(rule['consequents'])[0] if len(rule['consequents']) == 1 else str(list(rule['consequents']))
        print(f"  • Customers buying '{antecedent}' have {rule['confidence']*100:.1f}% chance of buying '{consequent}'")
    
    print()
    
    # Highest lift rules (most surprising associations)
    top_lift = rules.nlargest(5, 'lift')
    print("🔍 MOST SURPRISING PRODUCT ASSOCIATIONS:")
    for _, rule in top_lift.iterrows():
        antecedent = list(rule['antecedents'])[0] if len(rule['antecedents']) == 1 else str(list(rule['antecedents']))
        consequent = list(rule['consequents'])[0] if len(rule['consequents']) == 1 else str(list(rule['consequents']))
        print(f"  • '{antecedent}' + '{consequent}' appear together {rule['lift']:.1f}x more than expected")
    
    print()
    print("📋 BUSINESS APPLICATIONS:")
    print("  • Product placement: Place associated items near each other")
    print("  • Bundle offers: Create bundles from high-confidence rules") 
    print("  • Recommendation engine: Suggest items based on current cart")
    print("  • Inventory management: Stock associated items in similar quantities")
```

## Model Deployment and Monitoring

```python
# Save models and preprocessing objects for deployment
import joblib
from datetime import datetime

print("=== MODEL DEPLOYMENT PREPARATION ===")
print()

# Save customer segmentation model
if 'kmeans_final' in locals():
    model_artifacts = {
        'kmeans_model': kmeans_final,
        'scaler': scaler,
        'feature_columns': ['Recency', 'Frequency', 'Monetary'],
        'cluster_names': cluster_names,
        'training_date': datetime.now().strftime('%Y-%m-%d'),
        'model_performance': {
            'silhouette_score': silhouette_score(rfm_scaled, cluster_labels),
            'n_customers': len(rfm_table)
        }
    }
    
    # Save to file
    joblib.dump(model_artifacts, 'customer_segmentation_model.pkl')
    print("✅ Customer segmentation model saved to 'customer_segmentation_model.pkl'")

# Example of how to load and use the model for new predictions
def predict_customer_segment(recency, frequency, monetary, model_path='customer_segmentation_model.pkl'):
    """Predict customer segment for new customer data"""
    
    # Load model artifacts
    artifacts = joblib.load(model_path)
    
    # Prepare data
    customer_data = pd.DataFrame({
        'Recency': [recency],
        'Frequency': [frequency], 
        'Monetary': [np.log1p(monetary)]  # Apply same log transformation
    })
    
    # Scale features
    customer_scaled = artifacts['scaler'].transform(customer_data)
    
    # Predict cluster
    cluster = artifacts['kmeans_model'].predict(customer_scaled)[0]
    cluster_name = artifacts['cluster_names'].get(cluster, f'Cluster {cluster}')
    
    return cluster, cluster_name

# Example usage
print("\nExample prediction for new customer:")
print("Customer: Recency=15 days, Frequency=8 purchases, Monetary=£500")
if 'kmeans_final' in locals():
    cluster, cluster_name = predict_customer_segment(15, 8, 500)
    print(f"Predicted segment: {cluster_name} (Cluster {cluster})")

print("\n=== MONITORING RECOMMENDATIONS ===")
print()
print("📊 KEY METRICS TO MONITOR:")
print("  • Customer Segmentation:")
print("    - Segment distribution changes over time")
print("    - Average RFM values per segment")
print("    - Migration between segments")
print("    - Revenue contribution per segment")
print()
print("  • Stock Clustering:")
print("    - Cluster stability during market changes") 
print("    - Performance of each cluster")
print("    - New stocks assignment to clusters")
print()
print("  • Association Rules:")
print("    - Rule confidence and lift changes")
print("    - New product associations")
print("    - Seasonal pattern changes")
print()
print("🔄 RE-TRAINING SCHEDULE:")
print("  • Customer segments: Monthly (customer behavior changes)")
print("  • Stock clusters: Weekly (market volatility)")
print("  • Association rules: Weekly (inventory and promotions impact)")
print()
print("⚠️  ALERT CONDITIONS:")
print("  • Silhouette score drops below 0.3")
print("  • Cluster sizes become heavily imbalanced")
print("  • Model performance degrades significantly")
```

## Summary and Next Steps

```python
print("="*60)
print("                    TUTORIAL SUMMARY")
print("="*60)
print()
print("🎯 COMPLETED CASE STUDIES:")
print("  1. ✅ Customer Segmentation with RFM Analysis")
print("     - Used real e-commerce transaction data") 
print("     - Applied K-means clustering and PCA")
print("     - Generated actionable business insights")
print()
print("  2. ✅ Stock Market Clustering")
print("     - Downloaded real stock price data")
print("     - Clustered stocks by financial characteristics")  
print("     - Created investment recommendations")
print()
print("  3. ✅ News Article Text Clustering")
print("     - Processed 20 Newsgroups dataset")
print("     - Applied TF-IDF and SVD dimensionality reduction")
print("     - Compared predicted vs actual categories")
print()
print("  4. ✅ Market Basket Analysis")
print("     - Mined association rules from retail data")
print("     - Created product association networks")
print("     - Generated cross-selling recommendations")
print()
print("🛠️  TECHNIQUES LEARNED:")
print("  • K-Means, DBSCAN, Hierarchical Clustering")
print("  • PCA, TruncatedSVD for dimensionality reduction")
print("  • Association rule mining with Apriori algorithm")
print("  • RFM analysis for customer segmentation")
print("  • TF-IDF for text feature extraction")
print("  • Network analysis for relationship discovery")
print()
print("📈 BUSINESS VALUE DELIVERED:")
print("  • Customer targeting and personalization strategies")
print("  • Portfolio diversification recommendations") 
print("  • Content organization and recommendation systems")
print("  • Cross-selling and inventory optimization")
print()
print("🚀 NEXT STEPS:")
print("  1. Apply these techniques to your own datasets")
print("  2. Experiment with different algorithms (DBSCAN, Hierarchical)")
print("  3. Try advanced techniques:")
print("     - UMAP for non-linear dimensionality reduction")
print("     - Topic modeling with LDA")
print("     - Deep learning approaches (autoencoders)")
print("  4. Build automated pipelines for model deployment")
print("  5. Set up monitoring and retraining workflows")
print()
print("📚 ADDITIONAL RESOURCES:")
print("  • Scikit-learn documentation and examples")
print("  • 'Python Machine Learning' by Sebastian Raschka") 
print("  • 'Hands-On Unsupervised Learning' by Ankur Patel")
print("  • Kaggle datasets for more practice")
print()
print("="*60)
print("         🎉 CONGRATULATIONS! 🎉")
print("   You've completed a comprehensive real-world")
print("      unsupervised learning tutorial!")
print("="*60)
```
