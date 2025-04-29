import pandas as pd
import geopandas as gpd
import shapely.geometry as shp
import matplotlib.pyplot as plt
import os
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points

def snap_points_to_roads(tracks, roads):
    snapped_points = []

    for idx, point in tracks.iterrows():
        # Найдем ближайшую дорогу
        nearest_road = roads.geometry.iloc[
            roads.distance(point.geometry).idxmin()
        ]

        # Найдем ближайшую точку на этой дороге
        nearest_point_on_road = nearest_points(point.geometry, nearest_road)[1]

        # Добавляем точку в список
        snapped_points.append({
            'uuid': point['uuid'],
            'geometry': nearest_point_on_road
        })

    # Создаем новый GeoDataFrame
    snapped_gdf = gpd.GeoDataFrame(snapped_points, geometry='geometry', crs=tracks.crs)
    return snapped_gdf

def get_tracks(path_csv):
    tracks = pd.read_csv(path_csv, low_memory=False)

    tracks['geometry'] = tracks.apply(
        lambda row: Point(row['lon'], row['lat']), axis=1
    )
    tracks = gpd.GeoDataFrame(tracks, geometry='geometry', crs='EPSG:4326')

    tracks = tracks.to_crs(epsg=3857)
    return tracks

def filter_by_uuid(tracks, uuid_filter):

    filtered_tracks = tracks[tracks['uuid'].isin(uuid_filter)]
    if filtered_tracks.empty:
        print(f"Точки с UUID {uuid_filter} не найдены.")
    return filtered_tracks

tracks_december = get_tracks("normalized_tracks_march.csv")

uds_path = 'Graph_Irkutsk_link'

try:
    uds_path = 'Graph_Irkutsk_link/Graph_Irkutsk_link.SHP'
    UDS = gpd.read_file(uds_path)
    print("SHP-файл успешно загружен!")
except Exception as e:
    print(f"Ошибка при чтении SHP-файла: {e}")
    exit()


UDS = UDS.to_crs(epsg=3857)

needed_uuid = [29660]
routes_december = filter_by_uuid(tracks_december, uuid_filter=needed_uuid)

snapped_tracks = snap_points_to_roads(tracks_december, UDS)

# Визуализация
fig, ax = plt.subplots(figsize=(12, 8))
base = UDS.plot(ax=ax, edgecolor='yellow', facecolor='none', linewidth=0.5)
snapped_tracks.plot(ax=base, color='green', label='Routes December')
plt.legend()
plt.show()
