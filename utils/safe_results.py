import pandas as pd
import pathlib

def safe_raw_results(file_path : str, results : list[dict]):
    df = pd.DataFrame(results)
    csv_path = file_path + "/raw_results.csv"
    df.to_csv(csv_path, index=False)


def safe_by_class_results(file_path : str, results : list[dict]):
    df = pd.DataFrame(results)
    grouped = df.groupby('query_label').mean().reset_index()
    grouped = grouped.rename(columns={
        'ap': 'mean_ap',
        'num_retrieved': 'mean_num_retrieved',
        'relevant': 'mean_relevant'
    })
    csv_path = file_path + "/results_by_class.csv"
    df.to_csv(csv_path, index=False)


def compare_results_by_class(old_csv_path: str, new_csv_path: str, csv_path) -> pd.DataFrame:
    
    old_df = pd.read_csv(old_csv_path)
    new_df = pd.read_csv(new_csv_path)

    # Merge the DataFrames on 'query_label' to align classes
    merged_df = pd.merge(old_df, new_df, on='query_label', suffixes=('_old', '_new'))

    # Calculate differences
    merged_df['delta_mean_ap'] = merged_df['mean_ap_new'] - merged_df['mean_ap_old']
    merged_df['delta_mean_num_retrieved'] = merged_df['mean_num_retrieved_new'] - merged_df['mean_num_retrieved_old']
    merged_df['delta_mean_relevant'] = merged_df['mean_relevant_new'] - merged_df['mean_relevant_old']

    # Select relevant columns for the final DataFrame
    difference_df = merged_df[['query_label', 'delta_mean_ap', 'delta_mean_num_retrieved', 'delta_mean_relevant']]

    difference_df.to_csv(csv_path, index=False)

    return difference_df