from operations.data.preprocessing import (
    load_and_clean_columns,
    remove_empty_rows,
    fill_group_means,
    fill_result_columns,
    final_cleaning_and_encoding,
    save_cleaned_data
)

def main():
    input_path = "data/female_athletes_2425_full_stats_with_ranks.csv"
    output_path = "data/female_athletes_cleaned_final.csv"

    column_groups = [
        ['StatShooting_24_25', 'StatShooting_23_24', 'StatShooting_22_23', 'StatShooting_21_22'],
        ['Prone_24_25', 'Prone_23_24', 'Prone_22_23', 'Prone_21_22'],
        ['Standing_24_25', 'Standing_23_24', 'Standing_22_23', 'Standing_21_22'],
        ['Skiing_24_25', 'Skiing_23_24', 'Skiing_22_23', 'Skiing_21_22'],
        ['SkiKMB_24_25', 'SkiKMB_23_24', 'SkiKMB_22_23', 'SkiKMB_21_22']
    ]

    df = load_and_clean_columns(input_path)
    df = remove_empty_rows(df, column_groups)
    df = fill_group_means(df, column_groups)
    df = fill_result_columns(df, last_group_col='SkiKMB_21_22')
    df = final_cleaning_and_encoding(df)
    save_cleaned_data(df, output_path)

if __name__ == "__main__":
    main()
