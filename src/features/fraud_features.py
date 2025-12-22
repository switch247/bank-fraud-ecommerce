import pandas as pd
import numpy as np

def merge_ip_country(fraud_df: pd.DataFrame, ip_country_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge fraud data with IP country data using range-based lookup.
    Assumes ip_country_df has 'lower_bound_ip_address', 'upper_bound_ip_address', 'country'.
    Assumes fraud_df has 'ip_address'.
    """
    # Ensure IPs are numeric
    fraud_df = fraud_df.copy()
    ip_country_df = ip_country_df.copy()
    
    # Sort for merge_asof
    # We preserve the original index to restore order if needed, though usually not critical for EDA
    fraud_df = fraud_df.sort_values("ip_address")
    ip_country_df = ip_country_df.sort_values("lower_bound_ip_address")
    
    # merge_asof on ip_address >= lower_bound_ip_address
    # direction='backward' finds the last row in right where on <= left
    # i.e. lower_bound <= ip_address
    merged = pd.merge_asof(
        fraud_df,
        ip_country_df,
        left_on="ip_address",
        right_on="lower_bound_ip_address",
        direction="backward"
    )
    
    # Filter where ip_address is also <= upper_bound_ip_address
    # If it's not in range, country should be NaN
    mask = (merged["ip_address"] >= merged["lower_bound_ip_address"]) & \
           (merged["ip_address"] <= merged["upper_bound_ip_address"])
           
    merged.loc[~mask, "country"] = np.nan
    
    # Drop the bound columns
    merged = merged.drop(columns=["lower_bound_ip_address", "upper_bound_ip_address"])
    
    return merged

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features: time_since_signup, hour_of_day, day_of_week.
    """
    df = df.copy()
    
    # Convert to datetime if not already
    df["signup_time"] = pd.to_datetime(df["signup_time"])
    df["purchase_time"] = pd.to_datetime(df["purchase_time"])
    
    # Time since signup (in seconds)
    df["time_since_signup"] = (df["purchase_time"] - df["signup_time"]).dt.total_seconds()
    
    # Hour of day and Day of week from purchase time
    df["hour_of_day"] = df["purchase_time"].dt.hour
    df["day_of_week"] = df["purchase_time"].dt.dayofweek
    
    # Month/Year might be useful too
    df["purchase_month"] = df["purchase_time"].dt.month
    
    return df

def add_transaction_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add transaction frequency features per user and device.
    """
    df = df.copy()
    
    # Transaction count per user
    user_counts = df["user_id"].value_counts().rename("user_txn_count")
    df = df.merge(user_counts, left_on="user_id", right_index=True, how="left")
    
    # Transaction count per device
    device_counts = df["device_id"].value_counts().rename("device_txn_count")
    df = df.merge(device_counts, left_on="device_id", right_index=True, how="left")
    
    # IP counts
    ip_counts = df["ip_address"].value_counts().rename("ip_txn_count")
    df = df.merge(ip_counts, left_on="ip_address", right_index=True, how="left")
    
    return df
