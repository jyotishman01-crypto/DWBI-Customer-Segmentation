
# ============================================================
# DWBI PROJECT
# RETAIL CUSTOMER SEGMENTATION USING K-MEANS CLUSTERING
# Interactive Dashboard using Streamlit
# ============================================================

# 1. IMPORT LIBRARIES

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Retail Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)


# ============================================================
# 3. LOAD DATASET AND TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model():

    # Locate the main project folder
    base_dir = Path(__file__).resolve().parent.parent

    # Load original dataset
    dataset_path = base_dir / "dataset" / "retail_customers.csv"

    df = pd.read_csv(dataset_path)

    # Select relevant features
    features = [
        "Annual_Income_kUSD",
        "Spending_Score",
        "Monthly_Store_Visits"
    ]

    X = df[features].copy()

    # Feature scaling
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # Train K-Means model
    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    df["Cluster"] = kmeans.fit_predict(X_scaled)

    return df, scaler, kmeans, features


df, scaler, kmeans, features = train_model()


# ============================================================
# 4. CLUSTER DESCRIPTIONS
# ============================================================

cluster_names = {
    0: "Lower-engagement customers",
    1: "Moderately engaged customers",
    2: "Highly engaged customers"
}


# ============================================================
# 5. DASHBOARD HEADER
# ============================================================

st.title("🛍️ Retail Customer Segmentation Dashboard")

st.markdown(
    """
    ### Customer Behaviour Analysis Using K-Means Clustering

    This dashboard uses machine learning to group retail customers
    based on annual income, spending score and monthly store visits.
    """
)

st.divider()


# ============================================================
# 6. KEY PERFORMANCE INDICATORS
# ============================================================

st.subheader("Customer Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Customers",
        len(df)
    )

with col2:
    st.metric(
        "Total Clusters",
        df["Cluster"].nunique()
    )

with col3:
    st.metric(
        "Average Spending Score",
        round(df["Spending_Score"].mean(), 2)
    )


st.divider()


# ============================================================
# 7. NEW CUSTOMER INPUT SECTION
# ============================================================

st.header("New Customer Segmentation")

st.write(
    "Enter the customer's details below to predict the "
    "customer segment using the trained K-Means model."
)

input_col, result_col = st.columns(2)

with input_col:

    st.subheader("Customer Information")

    with st.form("customer_form"):

        income = st.number_input(
            "Annual Income (thousand USD)",
            min_value=0.0,
            max_value=500.0,
            value=60.0,
            step=1.0
        )

        spending = st.number_input(
            "Spending Score",
            min_value=0.0,
            max_value=100.0,
            value=60.0,
            step=1.0
        )

        visits = st.number_input(
            "Monthly Store Visits",
            min_value=0,
            max_value=100,
            value=8,
            step=1
        )

        predict_button = st.form_submit_button(
            "Predict Customer Segment",
            type="primary"
        )


# ============================================================
# 8. CUSTOMER CLUSTER PREDICTION
# ============================================================

with result_col:

    st.subheader("Prediction Result")

    if predict_button:

        # Create new customer DataFrame
        new_customer = pd.DataFrame(
            [[income, spending, visits]],
            columns=features
        )

        # Apply the existing scaler
        new_customer_scaled = scaler.transform(
            new_customer
        )

        # Predict cluster
        predicted_cluster = int(
            kmeans.predict(new_customer_scaled)[0]
        )

        # Display result
        st.success(
            f"Predicted Customer Cluster: {predicted_cluster}"
        )

        st.subheader(
            cluster_names[predicted_cluster]
        )

        # Display entered information
        st.write("Customer Details")

        st.dataframe(
            new_customer,
            use_container_width=True
        )

        # Display cluster profile
        profile = df.groupby("Cluster")[features].mean()

        st.write("Average Characteristics of This Cluster")

        st.dataframe(
            profile.loc[[predicted_cluster]].round(2),
            use_container_width=True
        )

    else:

        st.info(
            "Enter customer details and click Predict "
            "Customer Segment to see the result."
        )


st.divider()


# ============================================================
# 9. CUSTOMER CLUSTER VISUALIZATION
# ============================================================

st.header("Customer Segmentation Visualization")

fig, ax = plt.subplots(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Annual_Income_kUSD",
    y="Spending_Score",
    hue="Cluster",
    palette="Set1",
    s=120,
    ax=ax
)

# Calculate cluster centres in original units
centres = scaler.inverse_transform(
    kmeans.cluster_centers_
)

# Display cluster centres
ax.scatter(
    centres[:, 0],
    centres[:, 1],
    marker="X",
    color="black",
    s=250,
    label="Cluster Centres"
)

ax.set_title(
    "Retail Customer Segmentation Using K-Means",
    fontsize=14
)

ax.set_xlabel("Annual Income (thousand USD)")

ax.set_ylabel("Spending Score")

ax.legend(title="Customer Groups")

ax.grid(alpha=0.3)

st.pyplot(fig)

plt.close(fig)


st.divider()


# ============================================================
# 10. CLUSTER DISTRIBUTION
# ============================================================

st.header("Customer Distribution by Cluster")

cluster_counts = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)

dist_col1, dist_col2 = st.columns(2)

with dist_col1:

    st.write("Customer Count")

    st.bar_chart(cluster_counts)

with dist_col2:

    st.write("Cluster Distribution")

    st.dataframe(
        cluster_counts.rename("Customer_Count"),
        use_container_width=True
    )


st.divider()


# ============================================================
# 11. CLUSTER CHARACTERISTICS
# ============================================================

st.header("Customer Group Analysis")

cluster_summary = (
    df.groupby("Cluster")[features]
    .mean()
    .round(2)
)

cluster_summary.insert(
    0,
    "Customer_Count",
    df.groupby("Cluster").size()
)

st.dataframe(
    cluster_summary,
    use_container_width=True
)


# ============================================================
# 12. COMPLETE CUSTOMER DATASET
# ============================================================

st.header("Complete Customer Dataset")

with st.expander("View all customer records"):

    st.dataframe(
        df,
        use_container_width=True
    )


# ============================================================
# 13. DOWNLOAD CUSTOMER SEGMENTATION RESULTS
# ============================================================

st.subheader("Download Results")

csv_data = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Customer Segmentation CSV",
    data=csv_data,
    file_name="customer_segments.csv",
    mime="text/csv"
)


# ============================================================
# 14. FOOTER
# ============================================================

st.divider()

st.caption(
    "DWBI Academic Project | Retail Customer Segmentation "
    "Using K-Means Clustering | Python and Streamlit"
)

