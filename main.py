
# ==========================================
# DWBI PROJECT
# Retail Customer Segmentation Using K-Means
# Step 2: Dataset Loading and Exploration
# ==========================================

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("dataset/retail_customers.csv")

# Display first 5 records
print("\n1. FIRST FIVE RECORDS")
print(df.head())

# Display last 5 records
print("\n2. LAST FIVE RECORDS")
print(df.tail())

# Display dataset dimensions
print("\n3. DATASET SHAPE")
print(df.shape)

# Display column names
print("\n4. DATASET ATTRIBUTES")
print(df.columns)

# Display data types and information
print("\n5. DATASET INFORMATION")
print(df.info())

# Display statistical summary
print("\n6. STATISTICAL SUMMARY")
print(df.describe())

# Check missing values
print("\n7. MISSING VALUES")
print(df.isnull().sum())


# ==========================================
# STEP 3: DATA PREPROCESSING
# ==========================================

print("\n========== DATA PREPROCESSING ==========")

# 1. Check missing values
print("\nMissing values before preprocessing:")
print(df.isnull().sum())

# 2. Select relevant numerical features
features = [
    "Annual_Income_kUSD",
    "Spending_Score",
    "Monthly_Store_Visits"
]

X = df[features].copy()

# 3. Check and handle missing values if present
if X.isnull().sum().sum() > 0:
    X = X.fillna(X.median())
    print("\nMissing values handled using median.")

else:
    print("\nNo missing values found.")

# 4. Display selected features
print("\nSelected Features:")
print(X.head())

# 5. Display feature information
print("\nSelected Feature Information:")
print(X.info())

# 6. Final dataset dimensions
print("\nFinal Dataset Shape:")
print(X.shape)

print("\nData preprocessing completed successfully!")




# ==========================================
# STEP 4: FEATURE SCALING
# ==========================================

from sklearn.preprocessing import StandardScaler

print("\n========== FEATURE SCALING ==========")

# Create StandardScaler object
scaler = StandardScaler()

# Apply scaling to selected features
X_scaled = scaler.fit_transform(X)

# Convert scaled data into DataFrame
X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=features
)

# Display original feature values
print("\nOriginal Feature Values:")
print(X.head())

# Display scaled feature values
print("\nScaled Feature Values:")
print(X_scaled_df.head())

# Check mean of scaled features
print("\nMean After Scaling:")
print(X_scaled_df.mean())

# Check standard deviation
print("\nStandard Deviation After Scaling:")
print(X_scaled_df.std(ddof=0))

print("\nFeature scaling completed successfully!")


# ==========================================
# STEP 5: ELBOW METHOD
# ==========================================

from sklearn.cluster import KMeans

print("\n========== ELBOW METHOD ==========")

# Store WCSS values
wcss = []

# Test different numbers of clusters
for k in range(1, 9):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    # Store WCSS (Inertia)
    wcss.append(kmeans.inertia_)

    print(f"K = {k}, WCSS = {kmeans.inertia_:.2f}")


# Plot the Elbow graph
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 9),
    wcss,
    marker='o',
    linestyle='--'
)

plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS (Inertia)")

plt.xticks(range(1, 9))
plt.grid(True)

# Save the graph
plt.savefig("outputs/elbow_method.png", dpi=300)

# Display graph
plt.show()

print("\nElbow Method completed successfully!")





# ==========================================
# STEP 6: K-MEANS CLUSTERING
# ==========================================

print("\n========== K-MEANS CLUSTERING ==========")

# Select optimal number of clusters
optimal_k = 3

# Create K-Means model
kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

# Train the model
kmeans.fit(X_scaled)

# Display model information
print("\nK-Means model trained successfully!")

print("Number of clusters:", kmeans.n_clusters)

print("Number of iterations:", kmeans.n_iter_)

print("Final WCSS:", round(kmeans.inertia_, 2))

print("\nK-Means clustering completed!")






# ==========================================
# STEP 7: CUSTOMER CLUSTER ASSIGNMENT
# ==========================================

print("\n========== CUSTOMER CLUSTER ASSIGNMENT ==========")

# Assign cluster labels to each customer
df["Cluster"] = kmeans.labels_

# Display customer ID and assigned cluster
print("\nCustomer Cluster Assignments:")

print(df[["Customer_ID", "Cluster"]].to_string(index=False))

# Count customers in each cluster
print("\nNumber of Customers in Each Cluster:")

cluster_counts = df["Cluster"].value_counts().sort_index()

print(cluster_counts)

# Display complete segmented dataset
print("\nComplete Customer Segmentation Dataset:")

print(df.to_string(index=False))

# Create outputs folder if it does not exist
import os

os.makedirs("outputs", exist_ok=True)

# Export results to CSV
df.to_csv("outputs/customer_segments.csv", index=False)

print("\nCustomer segmentation results saved successfully!")

print("\nCustomer assignment completed!")



# ==========================================
# STEP 8: CUSTOMER CLUSTER VISUALIZATION
# ==========================================

print("\n========== CUSTOMER CLUSTER VISUALIZATION ==========")

# Create scatter plot
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Annual_Income_kUSD",
    y="Spending_Score",
    hue="Cluster",
    palette="Set1",
    s=120
)

# Calculate cluster centres in original units
centres = scaler.inverse_transform(kmeans.cluster_centers_)

# Display cluster centres
plt.scatter(
    centres[:, 0],
    centres[:, 1],
    marker="X",
    s=250,
    color="black",
    label="Cluster Centres"
)

# Add graph title and labels
plt.title("Retail Customer Segmentation Using K-Means")

plt.xlabel("Annual Income (thousand USD)")

plt.ylabel("Spending Score")

plt.legend(title="Customer Groups")

plt.grid(alpha=0.3)

plt.tight_layout()

# Save scatter plot
plt.savefig(
    "outputs/customer_clusters.png",
    dpi=300
)

# Display graph
plt.show()

print("\nCustomer cluster visualization completed!")






# ==========================================
# STEP 9: CLUSTER CENTRES AND ANALYSIS
# ==========================================

print("\n========== CLUSTER CENTRE ANALYSIS ==========")

# Convert scaled cluster centres into original units
centers_original = scaler.inverse_transform(
    kmeans.cluster_centers_
)

# Create a DataFrame for cluster centres
centers_df = pd.DataFrame(
    centers_original,
    columns=features
)

centers_df.index.name = "Cluster"

# Display cluster centres
print("\nCLUSTER CENTRES:")
print(centers_df.round(2))

# Calculate average characteristics of each cluster
cluster_profile = df.groupby("Cluster")[features].mean()

print("\nCUSTOMER GROUP CHARACTERISTICS:")
print(cluster_profile.round(2))

# Calculate number of customers in each cluster
cluster_size = df["Cluster"].value_counts().sort_index()

print("\nCUSTOMER COUNT IN EACH CLUSTER:")
print(cluster_size)

# Combine cluster size and average characteristics
cluster_summary = cluster_profile.copy()

cluster_summary.insert(
    0,
    "Customer_Count",
    cluster_size
)

print("\nFINAL CUSTOMER SEGMENTATION SUMMARY:")
print(cluster_summary.round(2))

# Save cluster centres
centers_df.to_csv("outputs/cluster_centres.csv")

# Save customer group analysis
cluster_summary.to_csv("outputs/cluster_analysis.csv")

print("\nCluster analysis completed successfully!")
