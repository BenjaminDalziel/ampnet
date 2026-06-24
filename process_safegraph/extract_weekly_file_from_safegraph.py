from json_columns_parsing import *
import os, sys
import pandas as pd
import polars as pl
import pyarrow
import numpy as np
import json
import time
from datetime import date, datetime

target_folder = r"\\depot.engr.oregonstate.edu\mime_u1\dalziel\Safe Graph Data\Weekly Patterns\Digital_Twins_Analysis\2019\2019_weekly"

data_2019 = '//depot.engr.oregonstate.edu/mime_u1/dalziel/Safe Graph Data/Weekly Patterns/2019_Weekly_Patterns'
data_2020 = '//depot.engr.oregonstate.edu/mime_u1/dalziel/Safe Graph Data/Weekly Patterns/2020_Weekly_Patterns'
cities = ['Portland']


if (__name__) == '__main__':
    
    start = time.time()
    # iterate through the files in 2019 data

    for index, week in enumerate(os.listdir(data_2019)):
        print("opening: " + os.path.join((data_2019),week))
        if (week.endswith('.csv.gz')):
            # read each week's csv
            week_start = datetime.strptime(week[:10], "%Y-%m-%d").date()
            curr_week = pl.scan_csv(os.path.join((data_2019),week))
            # choose these columns to keep
            curr_week = curr_week.select(['safegraph_place_id', 
                                        'region',
                                        'city',
                                        'raw_visit_counts',
                                        'raw_visitor_counts',
                                        'visits_by_day',
                                        'visits_by_each_hour',
                                        'poi_cbg',
                                        'visitor_home_cbgs',
                                        'visitor_daytime_cbgs',
                                        'distance_from_home',
                                        'median_dwell',
                                        'bucketed_dwell_times',
                                        'date_range_start'])
            curr_week = curr_week.filter(pl.col('region') == 'OR')
            curr_week = curr_week.drop('region').collect()
            # generate csv name in the format dateOfWeekStart.csv
            csv_name = f'{week_start}.csv'
            # cities = list(curr_week.unique(subset = 'city')['city'])
            # for each city of interest, process the related data and write_csv to a folder under the city name
            for city in cities:
                city_csv = curr_week.filter(pl.col('city') == city)
                city_csv = read_visits_by_each_hour(city_csv,'visits_by_each_hour')
                # city_csv = read_visits_by_each_day(city_csv,'visits_by_day')
                city_csv = read_home_cbg_column(city_csv,'visitor_home_cbgs')
                city_csv = read_daytime_cbg_column(city_csv,'visitor_daytime_cbgs')
                city_csv = read_bucketed_dwell_times(city_csv)
                city_csv = city_csv.with_columns(pl.col("distance_from_home").cast(pl.Int64, strict = False).fill_null(np.nan).alias("distance_from_home"))
                city_csv = city_csv.with_columns(pl.col('date_range_start').str.to_datetime().dt.replace_time_zone("America/Ensenada").alias('date_range_start'))
                city_csv = city_csv.with_columns((pl.col('date_range_start').dt.year()).alias('date_range_start_year'))
                city_csv = city_csv.with_columns((pl.col('date_range_start').dt.month()).alias('date_range_start_month'))
                city_csv = city_csv.with_columns((pl.col('date_range_start').dt.day()).alias('date_range_start_day'))
                file_location = target_folder + "/" + city
                city_csv = city_csv.drop('city')
                # city_csv.write_csv(os.path.join(file_location,csv_name))
    print (time.time()-start)