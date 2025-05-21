from operations.data.preprocessing import (
    load_and_clean_columns,
    remove_empty_rows,
    fill_group_means,
    fill_result_columns,
    final_cleaning_and_encoding,
    save_cleaned_data,
    encode_competition_participation,
    fill_missing_with_personal_average,
    save_to_database
)
from operations.data.updater import update_data
from operations.predict_all_events import train_model

def run_update():
    update_data(output_path="data/female_athletes_2425_full_stats_with_ranks.csv")

def run_preprocessing():
    input_path = "data/female_athletes_2425_full_stats_with_ranks.csv"
    output_path = "data/female_athletes_cleaned_final.csv"
    binary_output = "data/female_athletes_binary_competitions.csv"
    db_path = "data/athletes_data.db"

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
    df = final_cleaning_and_encoding(df)
    df_binary = encode_competition_participation(df, binary_output)
    df = fill_result_columns(df, last_group_col='SkiKMB_21_22')
    df = fill_missing_with_personal_average(df)
    save_cleaned_data(df, output_path)
    save_to_database(df, df_binary, db_path)

def run_training():
    models = {
        "1": "RandomForest",
        "2": "XGBoost",
        "3": "LSTM"
    }
    tasks = {
        "1": "participation",
        "2": "place",
        "3": "both"
    }

    print("\n=== Pasirinkite modelį ===")
    print("1. Random Forest")
    print("2. XGBoost")
    print("3. LSTM")
    model_choice = input("Modelio numeris: ").strip()

    print("\n=== Pasirinkite prognozės tipą ===")
    print("1. Dalyvavimas (participation)")
    print("2. Vieta (place)")
    print("3. Abi dalys")
    task_choice = input("Prognozės tipas: ").strip()

    model_name = models.get(model_choice)
    task_type = tasks.get(task_choice)

    if model_name and task_type:
        train_model(model_name, task_type)
    else:
        print("Netinkamas pasirinkimas.")

def main():
    while True:
        print("\n=== Pagrindinis meniu ===")
        print("1. Atnaujinti duomenis iš IBU API")
        print("2. Apdoroti (išvalyti) duomenis")
        print("3. Treniruoti modelius")
        print("4. Viskas iš eilės")
        print("0. Išeiti")

        choice = input("Pasirinkite veiksmą (0-4): ").strip()

        if choice == "1":
            run_update()
        elif choice == "2":
            run_preprocessing()
        elif choice == "3":
            run_training()
        elif choice == "4":
            run_update()
            run_preprocessing()
            run_training()
        elif choice == "0":
            print("Baigta.")
            break
        else:
            print("Neteisingas pasirinkimas. Bandykite dar kartą.")

if __name__ == "__main__":
    main()
