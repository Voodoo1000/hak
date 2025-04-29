import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Шаг 1: Чтение данных
file_path = 'tracks_december.csv'

# Указываем типы данных для каждого столбца
dtype_spec = {
    'accept_time': str,
    'signal_time': str,
    'clid': str,
    'uuid': 'Int64',  # Nullable integer type
    'vehicle_type': str,
    'route': str,
    'lat': float,
    'lon': float,
    'speed': float,
    'direction': float,
    'thread': str,
    'bind_lat': str,
    'bind_lon': str,
    'fly_time': 'Int64',  # Nullable integer type
    'life_time': 'Int64',  # Nullable integer type
    'd_acc': 'Int64'  # Nullable integer type
}

data = pd.read_csv(file_path, dtype=dtype_spec)

# Шаг 2: Преобразование временных меток
data['accept_time'] = pd.to_datetime(data['accept_time'])
data['signal_time'] = pd.to_datetime(data['signal_time'])

# Шаг 3: Обработка отсутствующих значений
numeric_columns = ['lat', 'lon', 'speed', 'direction', 'fly_time', 'life_time', 'd_acc']
categorical_columns = ['thread', 'bind_lat', 'bind_lon']

# Для числовых столбцов
for col in numeric_columns:
    if data[col].isnull().sum() > 0:
        data[col] = data[col].fillna(data[col].median())  # Заполнение медианой

# Для категориальных столбцов
for col in categorical_columns:
    if data[col].isnull().sum() > 0:
        data[col] = data[col].fillna('Unknown')  # Заполнение строкой

# Шаг 3.5: Удаление выбросов методом межквартильного размаха (IQR)
def remove_outliers_iqr(df, columns):
    mask = pd.Series([True] * len(df))  # Изначально все строки считаем "годными"
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        mask &= (df[col] >= lower_bound) & (df[col] <= upper_bound)  # Накладываем условие
    before_rows = df.shape[0]
    df = df[mask]
    after_rows = df.shape[0]
    print(f"Удалено выбросов: {before_rows - after_rows}")
    return df

data = data[(data['accept_time'].notna()) & (data['signal_time'].notna())]
data = remove_outliers_iqr(data, numeric_columns)

data['speed'] = data['speed'] * 3.6

# Шаг 4: Удаление дубликатов
# data.drop_duplicates(inplace=True)

# Шаг 5: Нормализация числовых данных
# scaler = MinMaxScaler()
# data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

# Шаг 6: Сохранение результата
output_file = 'normalized_tracks_december.csv'
data.to_csv(output_file, index=False)

print(f"Нормализованные данные сохранены в файл: {output_file}")

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
#
# # Шаг 1: Чтение данных
# file_path = 'tracks_march.csv'
#
# # Указываем типы данных для каждого столбца
# dtype_spec = {
#     'signal_time': str,
#     'clid': str,
#     'uuid': 'Int64',  # Nullable integer type
#     'vehicle_type': str,
#     'route': str,  # Маршрут может быть как числом, так и строкой
#     'lat': float,
#     'lon': float,
#     'speed': float,
#     'direction': float
# }
#
# # Чтение данных
# data = pd.read_csv(file_path, dtype=dtype_spec)
#
# # Шаг 2: Преобразование временных меток
# def convert_signal_time(signal_time):
#     try:
#         if pd.isna(signal_time) or signal_time.strip() == '':  # Проверка на пустые значения
#             return None
#         # Если это строка в формате HH:MM
#         return pd.to_datetime(signal_time, format='%H:%M').time()
#     except ValueError:
#         try:
#             # Если это число (например, 3.2407407407407406E-4)
#             return pd.to_datetime(float(signal_time), unit='D', origin='unix').time()
#         except Exception:
#             return None  # Возвращаем None для некорректных значений
#
# data['signal_time'] = data['signal_time'].apply(convert_signal_time)
#
# # Шаг 3: Обработка отсутствующих значений
# numeric_columns = ['lat', 'lon', 'speed', 'direction']
# categorical_columns = ['clid', 'vehicle_type', 'route']  # Добавляем 'route' в категориальные столбцы
#
# # Для числовых столбцов
# for col in numeric_columns:
#     if data[col].isnull().sum() > 0:
#         data[col] = data[col].fillna(data[col].median())  # Заполнение медианой
#
# # Для категориальных столбцов
# for col in categorical_columns:
#     if data[col].isnull().sum() > 0:
#         data[col] = data[col].fillna('Unknown')  # Заполнение строкой
#
# # Шаг 3.5: Удаление выбросов методом межквартильного размаха (IQR)
# def remove_outliers_iqr(df, columns):
#     mask = pd.Series([True] * len(df))  # Изначально все строки считаем "годными"
#     for col in columns:
#         Q1 = df[col].quantile(0.25)
#         Q3 = df[col].quantile(0.75)
#         IQR = Q3 - Q1
#         lower_bound = Q1 - 1.5 * IQR
#         upper_bound = Q3 + 1.5 * IQR
#         mask &= (df[col] >= lower_bound) & (df[col] <= upper_bound)  # Накладываем условие
#     before_rows = df.shape[0]
#     df = df[mask]
#     after_rows = df.shape[0]
#     print(f"Удалено выбросов: {before_rows - after_rows}")
#     return df
#
# data = remove_outliers_iqr(data, numeric_columns)
#
# # Шаг 6: Сохранение результата
# output_file = 'normalized_tracks_march.csv'
# data.to_csv(output_file, index=False)
#
# print(f"Нормализованные данные сохранены в файл: {output_file}")