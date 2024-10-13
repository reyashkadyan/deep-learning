
import matplotlib.pyplot as plt

def plot_time_series(df, column_name):
    filtered_df = df[[column_name]].dropna()
    plt.figure(figsize=(15, 5))
    plt.plot(filtered_df.index, filtered_df[column_name])
    plt.title(f'Time series for {column_name}')
    plt.xlabel('Year')
    plt.ylabel(column_name)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()