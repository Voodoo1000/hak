from xlsx2csv import Xlsx2csv
import geopandas as gpd
import shapely.geometry as shp
import matplotlib.pyplot as plt
import os

def Tracks(pathCsv):
    tracks = gpd.pd.read_csv(pathCsv, low_memory=False)

    # Преобразование геометрических данных
    tracks['geometry'] = tracks.apply(
        lambda row: shp.Point(row['lon'], row['lat']), axis=1
    )
    tracks = gpd.GeoDataFrame(tracks, geometry='geometry', crs='EPSG:4326')

    # Меняем систему координат
    tracks = tracks.to_crs(epsg=3857)
    return tracks
# Файл с треками
# track_file_march = 'МАРТ.xlsx'
# track_file_december = 'ДЕКАБРЬ.xlsx'
# # конвертация в csv
# Xlsx2csv(track_file_march, outputencoding="utf-8").convert("tracks_march.csv")
# Xlsx2csv(track_file_december, outputencoding="utf-8").convert("tracks_december.csv")

# Чтение CSV-файла
tracks_march = Tracks("tracks_march.csv")
tracks_december = Tracks("tracks_december.csv")

# Чтение УДС из архива
uds_path = 'Graph_Irkutsk_link'

# Проверка существования ZIP-архива
if not os.path.exists(f'{uds_path}.zip'):
    print(f"ZIP-архив {uds_path}.zip не найден!")
    exit()

try:
    # Чтение Shapefile из ZIP-архива
    uds_path = 'C:/Users/vladb/PycharmProjects/PythonProject1/Graph_Irkutsk_link/Graph_Irkutsk_link.SHP'
    UDS = gpd.read_file(uds_path)
    # UDS = gpd.read_file(f'zip://{uds_path}.zip/{uds_path}.SHP')
    print("SHP-файл успешно загружен!")
except Exception as e:
    print(f"Ошибка при чтении SHP-файла: {e}")
    exit()

# Меняем систему координат для УДС
UDS = UDS.to_crs(epsg=3857)

# Визуализация
base = UDS.plot(edgecolor='yellow')
tracks_march.plot(ax=base, edgecolor='blue')
tracks_december.plot(ax=base, edgecolor='green')
plt.show()