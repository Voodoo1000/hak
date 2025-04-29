import pandas as pd


def show_all_uuids_and_filter_by_uuid(path_csv):
    # Чтение CSV-файла
    tracks = pd.read_csv(path_csv, low_memory=False)

    # Вывод всех уникальных значений uuid
    unique_uuids = tracks['uuid'].unique()
    print("Все уникальные значения UUID:")
    for uuid in unique_uuids:
        print(uuid)

def filter_by_uuid(csv_file, target_uuid):
    # Чтение CSV-файла
    data = pd.read_csv(csv_file, low_memory=False)

    # Фильтрация строк по заданному uuid
    filtered_data = data[data['uuid'] == target_uuid]

    # Проверка, есть ли результаты
    if filtered_data.empty:
        print(f"Строки с UUID '{target_uuid}' не найдены.")
    else:
        print(f"Строки с UUID '{target_uuid}':")
        print(filtered_data)

    return filtered_data



csv_file_path = "tracks_december.csv"
target_uuid = 9574

# Шаг 1: Вывести все уникальные UUID
show_all_uuids_and_filter_by_uuid(csv_file_path)

# filtered_rows = filter_by_uuid(csv_file_path, target_uuid)
# output_file = 'uuid.csv'
# filtered_rows.to_csv(output_file, index=False)