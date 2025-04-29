import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.widgets import TextBox, Button
import numpy as np
import datetime
import traceback
import os

# === Загрузка дорожной сети ===
# (Функция load_road_network остается без изменений)
def load_road_network(path):
    """Загружает дорожную сеть из SHP-файла и перепроецирует в EPSG:3857."""
    if not os.path.exists(path):
        print(f"Ошибка: Файл дорожной сети не найден: {path}")
        exit()
    try:
        roads = gpd.read_file(path)
        print("Дорожная сеть загружена!")
        if roads.crs is None:
            print("Внимание: У SHP-файла отсутствует система координат. Попытка установить WGS84 (EPSG:4326).")
            roads.set_crs(epsg=4326, inplace=True)
        return roads.to_crs(epsg=3857)
    except Exception as e:
        print(f"Ошибка при чтении SHP-файла или установке/трансформации CRS: {e}")
        exit()

# === Загрузка и обработка данных треков (с исправлениями) ===
def load_and_process_tracks(filepath, time_col, time_format, lat_col='lat', lon_col='lon', speed_col='speed', target_crs=None): # Added time_format
    """Загружает CSV, обрабатывает время, координаты, скорость и создает GeoDataFrame."""
    if not os.path.exists(filepath):
        print(f"Предупреждение: Файл треков не найден: {filepath}. Пропуск.")
        return None

    print(f"Загрузка треков из {filepath}...")
    try:
        # --- ИСПРАВЛЕНИЕ 1: low_memory=False ---
        tracks_raw = pd.read_csv(filepath, low_memory=False)
        # --------------------------------------

        required_cols = [time_col, lat_col, lon_col, speed_col]
        missing_cols = [col for col in required_cols if col not in tracks_raw.columns]
        if missing_cols:
            print(f"Ошибка в {filepath}: Отсутствуют необходимые столбцы: {', '.join(missing_cols)}")
            return None

        # Преобразования с проверками
        tracks_raw['speed_num'] = pd.to_numeric(tracks_raw[speed_col], errors='coerce')

        # --- ИСПРАВЛЕНИЕ 2: Указание формата времени ---
        if time_format:
             print(f"  Используется формат времени: '{time_format}'")
             tracks_raw['datetime_col'] = pd.to_datetime(tracks_raw[time_col], format=time_format, errors='coerce')
        else:
             # Оставляем возможность автоопределения, если формат не задан
             print("  Попытка автоопределения формата времени...")
             tracks_raw['datetime_col'] = pd.to_datetime(tracks_raw[time_col], errors='coerce')
        # --------------------------------------------

        tracks_raw[lat_col] = pd.to_numeric(tracks_raw[lat_col], errors='coerce')
        tracks_raw[lon_col] = pd.to_numeric(tracks_raw[lon_col], errors='coerce')

        initial_rows = len(tracks_raw)
        tracks = tracks_raw.dropna(subset=['datetime_col', lat_col, lon_col, 'speed_num'])
        dropped_rows = initial_rows - len(tracks)
        if dropped_rows > 0:
             print(f"В {filepath} удалено {dropped_rows} строк из-за неверного формата/отсутствия времени, координат или скорости.")

        if tracks.empty:
            print(f"В {filepath} не осталось валидных данных после обработки.")
            return None

        geometry = gpd.points_from_xy(tracks[lon_col], tracks[lat_col])
        tracks_gdf = gpd.GeoDataFrame(tracks, geometry=geometry, crs='EPSG:4326')

        if target_crs:
            tracks_gdf = tracks_gdf.to_crs(target_crs)

        print(f"Треки из {filepath} загружены и подготовлены.")
        return tracks_gdf

    except Exception as e:
        print(f"Ошибка при загрузке или обработке треков из {filepath}: {e}")
        traceback.print_exc()
        return None

# === Создание графа (опционально) ===
# (Функция create_graph_from_roads остается без изменений)
def create_graph_from_roads(gdf):
    """Создает граф NetworkX из геометрии дорожной сети."""
    G = nx.Graph()
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom and geom.geom_type == 'LineString':
            coords = list(geom.coords)
            for i in range(len(coords) - 1):
                start = coords[i]
                end = coords[i + 1]
                if not (np.isnan(start[0]) or np.isnan(start[1]) or \
                        np.isnan(end[0]) or np.isnan(end[1])):
                    dx = start[0] - end[0]
                    dy = start[1] - end[1]
                    dist = (dx ** 2 + dy ** 2) ** 0.5
                    if dist > 0:
                        G.add_edge(start, end, weight=dist)
                        G.add_node(start, pos=start)
                        G.add_node(end, pos=end)
    return G

# --- Глобальные переменные для видимости ---
december_stops_visible = True
march_stops_visible = True
# -------------------------------------------

# --- Глобальные переменные для данных и плотов ---
december_zero_speed_points = None
march_zero_speed_points = None
december_stops_plot = None
march_stops_plot = None
# -------------------------------------------------

# === Функции переключения видимости ===
# (toggle_december_visibility и toggle_march_visibility без изменений)
def toggle_december_visibility(event):
    """Переключает видимость декабрьских остановок."""
    global december_stops_visible
    december_stops_visible = not december_stops_visible
    update_plots_visibility() # Обновляем все

def toggle_march_visibility(event):
    """Переключает видимость мартовских остановок."""
    global march_stops_visible
    march_stops_visible = not march_stops_visible
    update_plots_visibility() # Обновляем все


# === ОСНОВНАЯ Функция обновления отображения ===
# (update_plots_visibility без изменений)
def update_plots_visibility(event=None):
    """Обновляет видимость меток ОБОИХ наборов данных и применяет фильтр времени суток."""
    global december_stops_plot, march_stops_plot, \
           december_zero_speed_points, march_zero_speed_points, ax, \
           start_time_box, end_time_box, \
           december_stops_visible, march_stops_visible

    start_time_str = start_time_box.text
    end_time_str = end_time_box.text

    try:
        # --- Парсинг ВРЕМЕНИ СУТОК ---
        try:
            start_time = pd.to_datetime(start_time_str, format='%H:%M:%S', errors='raise').time()
        except (ValueError, TypeError):
            # print(f"Неверный формат начального времени '{start_time_str}'. Используется 00:00:00.")
            start_time = datetime.time(0, 0, 0)
            start_time_box.set_val("00:00:00")

        try:
            end_time = pd.to_datetime(end_time_str, format='%H:%M:%S', errors='raise').time()
        except (ValueError, TypeError):
            # print(f"Неверный формат конечного времени '{end_time_str}'. Используется 23:59:59.")
            end_time = datetime.time(23, 59, 59)
            end_time_box.set_val("23:59:59")

        # --- Функция для фильтрации и обновления одного набора данных ---
        def filter_and_update_single_dataset(zero_speed_points, plot_object, is_visible_flag, color, marker, label):
            # Обработка случая отсутствия данных для этого набора
            if zero_speed_points is None or zero_speed_points.empty:
                if plot_object: # Если плот был, но данных нет, скрываем
                    plot_object.set_visible(False)
                    plot_object.set_offsets(np.empty((0, 2)))
                return None # Возвращаем None, т.к. плота нет или он скрыт

            # Фильтрация по времени
            times_in_data = zero_speed_points['datetime_col'].dt.time
            if start_time <= end_time:
                time_condition = (times_in_data >= start_time) & (times_in_data <= end_time)
            else: # Пересечение полуночи
                time_condition = (times_in_data >= start_time) | (times_in_data <= end_time)
            filtered_stops = zero_speed_points[time_condition]

            # Обновление или создание графика
            current_plot = plot_object
            if current_plot: # Если график уже существует
                if not filtered_stops.empty:
                    coords = np.array([(point.x, point.y) for point in filtered_stops.geometry])
                    current_plot.set_offsets(coords)
                    current_plot.set_visible(is_visible_flag) # Видимость по флагу
                else: # Если после фильтрации точек нет
                    current_plot.set_offsets(np.empty((0, 2)))
                    current_plot.set_visible(False) # Скрываем
            # Если плота не было, но появились отфильтрованные точки
            elif not filtered_stops.empty:
                # Создаем новый график
                current_plot = ax.scatter(filtered_stops.geometry.x, filtered_stops.geometry.y,
                                        color=color, marker=marker, s=60, label=label, zorder=5)
                current_plot.set_visible(is_visible_flag) # Видимость по флагу
            # В противном случае (плота не было и точек нет) current_plot остается None

            # Возвращаем обновленный или новый объект графика (или None, если он не создан/скрыт)
            # Важно проверять get_visible(), т.к. он может быть скрыт флагом is_visible_flag
            return current_plot if current_plot and current_plot.get_visible() else None
        # ---------------------------------------------------------------

        # --- Применяем фильтрацию и обновление к каждому набору данных ---
        global december_stops_plot, march_stops_plot # Указываем, что будем модифицировать глобальные переменные
        temp_dec_plot = filter_and_update_single_dataset(
            december_zero_speed_points, december_stops_plot, december_stops_visible,
            'red', 'x', 'Остановки (Декабрь)'
        )
        # Обновляем глобальную переменную, только если filter... вернул не None
        if temp_dec_plot is not None:
             december_stops_plot = temp_dec_plot
        # Если вернулся None, значит плот скрыт или данных нет, december_stops_plot
        # может остаться старым (но скрытым) или стать None, если его не было

        temp_mar_plot = filter_and_update_single_dataset(
            march_zero_speed_points, march_stops_plot, march_stops_visible,
            'blue', '+', 'Остановки (Март)' # Другой цвет и маркер
        )
        if temp_mar_plot is not None:
             march_stops_plot = temp_mar_plot


        # --- Обновление легенды ---
        current_legend = ax.get_legend()
        handles = []
        labels = []
        # Собираем существующие элементы, кроме наших остановок
        if current_legend:
            base_handles = current_legend.legend_handles
            base_labels = [text.get_text() for text in current_legend.get_texts()]
            for h, l in zip(base_handles, base_labels):
                if l not in ['Остановки (Декабрь)', 'Остановки (Март)']:
                    handles.append(h)
                    labels.append(l)

        # Добавляем видимые остановки (проверяем существование и видимость plot объекта)
        if december_stops_plot and december_stops_plot.get_visible():
             # Проверяем, что ручка еще не добавлена (на случай многократных вызовов)
             if december_stops_plot not in handles:
                 handles.append(december_stops_plot)
                 labels.append('Остановки (Декабрь)')
        if march_stops_plot and march_stops_plot.get_visible():
             if march_stops_plot not in handles:
                 handles.append(march_stops_plot)
                 labels.append('Остановки (Март)')


        # Перерисовываем легенду только если есть что показывать
        if handles:
            loc = current_legend._get_loc() if current_legend else 'upper left'
            # Используем собранные handles и labels
            ax.legend(handles=handles, labels=labels, loc=loc)
        elif current_legend: # Если ручек не осталось, убираем легенду
             ax.legend_ = None

        plt.draw()

    except Exception as e:
        print(f"Ошибка при фильтрации времени или обновлении графика: {e}")
        traceback.print_exc()


# === Основной блок ===
if __name__ == "__main__":
    # --- Пути к файлам ---
    road_path = "Graph_Irkutsk_link/Graph_Irkutsk_link.SHP"
    december_tracks_path = "normalized_tracks_december.csv"
    march_tracks_path = "normalized_tracks_march.csv"
    # --- Имена ключевых столбцов ---
    time_column = 'signal_time'
    lat_column = 'lat'
    lon_column = 'lon'
    speed_column = 'speed'
    # --- Форматы времени для каждого файла ---
    december_time_format = '%Y-%m-%d %H:%M:%S' # Примерный формат для декабря
    march_time_format = '%H:%M:%S'            # Формат только времени для марта
    # -------------------------------------------------------------

    # Загрузка дорожной сети
    roads_gdf = load_road_network(road_path)
    if roads_gdf is None: exit()

    # Загрузка и обработка треков с указанием формата времени
    december_tracks_gdf = load_and_process_tracks(
        december_tracks_path, time_column, december_time_format, # Передаем формат
        lat_column, lon_column, speed_column, roads_gdf.crs
    )
    march_tracks_gdf = load_and_process_tracks(
        march_tracks_path, time_column, march_time_format,       # Передаем другой формат
        lat_column, lon_column, speed_column, roads_gdf.crs
    )

    # Извлечение точек с нулевой скоростью
    if december_tracks_gdf is not None:
        december_zero_speed_points = december_tracks_gdf[december_tracks_gdf['speed_num'] == 0].copy()
        if december_zero_speed_points.empty:
            print("В данных за Декабрь нет точек с нулевой скоростью.")
    else:
         december_zero_speed_points = None

    if march_tracks_gdf is not None:
        march_zero_speed_points = march_tracks_gdf[march_tracks_gdf['speed_num'] == 0].copy()
        if march_zero_speed_points.empty:
            print("В данных за Март нет точек с нулевой скоростью.")
    else:
         march_zero_speed_points = None

    # === ВИЗУАЛИЗАЦИЯ ===
    fig, ax = plt.subplots(figsize=(15, 10))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.25)

    # Рисуем дороги
    roads_gdf.plot(ax=ax, edgecolor='gray', linewidth=0.5, label='Дорожная сеть', zorder=1)

    # --- Расположение виджетов (2 столбца) ---
    col1_left = 0.10  # Левый столбец для времени/фильтра
    col2_left = 0.45  # Правый столбец для кнопок вкл/выкл
    textbox_width = 0.30
    button_width = 0.25
    widget_height = 0.04
    vertical_gap = 0.015

    # --- Столбец 1: Время и Фильтр ---
    start_time_bottom = 0.15  # Начинаем повыше
    ax_start_time = plt.axes([col1_left, start_time_bottom, textbox_width, widget_height])
    end_time_bottom = start_time_bottom - widget_height - vertical_gap
    ax_end_time = plt.axes([col1_left, end_time_bottom, textbox_width, widget_height])
    filter_button_bottom = end_time_bottom - widget_height - vertical_gap
    ax_filter_button = plt.axes(
        [col1_left, filter_button_bottom, button_width, widget_height])  # Кнопка фильтра шириной как другие

    # --- Столбец 2: Кнопки Вкл/Выкл (одна над другой) ---
    # Верхняя кнопка (Декабрь) - выровняем ее верх с верхом кнопки Фильтр для симметрии
    dec_toggle_button_bottom = start_time_bottom
    ax_dec_toggle_button = plt.axes([col2_left, dec_toggle_button_bottom, button_width, widget_height])

    # Нижняя кнопка (Март) - под кнопкой Декабрь
    mar_toggle_button_bottom = dec_toggle_button_bottom - widget_height - vertical_gap
    ax_mar_toggle_button = plt.axes([col2_left, mar_toggle_button_bottom, button_width, widget_height])
    # -------------------------------------------------------------

    # --- Создаем виджеты ---
    start_time_initial = "00:00:00"
    end_time_initial = "23:59:59"
    start_time_box = TextBox(ax_start_time, "Начало (ЧЧ:ММ:СС):", initial=start_time_initial)
    end_time_box = TextBox(ax_end_time, "Конец (ЧЧ:ММ:СС):", initial=end_time_initial)

    filter_button = Button(ax_filter_button, 'Фильтр времени')
    filter_button.on_clicked(update_plots_visibility)

    dec_toggle_button = Button(ax_dec_toggle_button, 'Показать/Скрыть Декабрь')
    dec_toggle_button.on_clicked(toggle_december_visibility)

    mar_toggle_button = Button(ax_mar_toggle_button, 'Показать/Скрыть Март')
    mar_toggle_button.on_clicked(toggle_march_visibility)
    # -------------------------

    # --- Первоначальная отрисовка и легенда ---
    update_plots_visibility()
    if december_zero_speed_points is None and march_zero_speed_points is None:
         print("Нет данных об остановках ни за один период.")
    elif (december_zero_speed_points is not None and december_zero_speed_points.empty and
          (march_zero_speed_points is None or march_zero_speed_points.empty)):
         print("Нет точек с нулевой скоростью для отображения в загруженных данных.")
    elif (march_zero_speed_points is not None and march_zero_speed_points.empty and
          (december_zero_speed_points is None or december_zero_speed_points.empty)):
          print("Нет точек с нулевой скоростью для отображения в загруженных данных.")


    ax.set_title("Остановки транспорта по дорожной сети Иркутска")
    ax.axis('off')
    plt.show()