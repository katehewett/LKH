import pandas as pd

def rolling_mean_uneven(df, window_size):
    """
    Calculate rolling mean with uneven time spacing.

    Args:
        df (pd.DataFrame): Dataframe with 'timestamp' and 'value' columns.
        window_size (int): Time window in seconds.

    Returns:
        pd.Series: Rolling mean.
    """

    df = df.sort_values('timestamp')
    df['rolling_mean'] = pd.NA

    for i in range(len(df)):
        start_time = df['timestamp'].iloc[i] - pd.Timedelta(seconds=window_size)
        window_data = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= df['timestamp'].iloc[i])]
        df['rolling_mean'].iloc[i] = window_data['value'].mean()

    return df['rolling_mean']

# Example usage:
data = {'timestamp': ['2023-01-01 10:00:00', '2023-01-01 10:01:00', '2023-01-01 10:03:00', '2023-01-01 10:05:00'],
        'value': [1, 2, 3, 4]}
df = pd.DataFrame(data)
df['timestamp'] = pd.to_datetime(df['timestamp'])

df['rolling_mean'] = rolling_mean_uneven(df, window_size=120) # 2-minute window
print(df)