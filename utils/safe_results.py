import pandas as pd
import pathlib
import os

def safe_raw_results(file_path: str, results: list[dict]):
    df = pd.DataFrame(results)
    csv_path = os.path.join(file_path, "raw_results.csv")
    df.to_csv(csv_path, index=False)


def safe_by_class_results(file_path: str, results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    grouped = df.groupby('query_label').mean(numeric_only=True).reset_index()
    grouped = grouped.rename(columns={
        'ap': 'mean_ap',
        'relevant': 'mean_relevant'
    })
    csv_path = os.path.join(file_path, "results_by_class.csv")
    grouped.to_csv(csv_path, index=False)
    return grouped

def compare_results_by_class(source: str, target: str) -> pd.DataFrame:

    old_df = pd.read_csv(source)
    new_df = pd.read_csv(target)

    # Merge the DataFrames on 'query_label' to align classes
    merged_df = pd.merge(old_df, new_df, on='query_label',
                         suffixes=('_old', '_new'))

    # Calculate differences
    merged_df['delta_mean_ap'] = merged_df['mean_ap_new'] - merged_df['mean_ap_old']
    merged_df['delta_mean_relevant'] = merged_df['mean_relevant_new'] - merged_df['mean_relevant_old']

    # Select relevant columns for the final DataFrame
    difference_df = merged_df[['query_label', 'delta_mean_ap', 'delta_mean_relevant']]

    # difference_df.to_csv(csv_path, index=False)

    return difference_df

def safe_total_results(file_path: str, map_score, df_by_calsses : pd.DataFrame):
    mean_ap_by_classes = df_by_calsses['mean_ap'].mean()
    total_score_df = pd.DataFrame([{'mAP': map_score, 'mAP by calsses':mean_ap_by_classes}])
    csv_path = csv_path = os.path.join(file_path, "total_score.csv")
    total_score_df.to_csv(csv_path, index=False)
