import json
import os
import sys
import time
from datetime import date, datetime

import numpy as np
import pandas as pd
import polars as pl
import pyarrow

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

city = 'Portland'

data_2019 = "../../novus/matchingnemo/scratch/safegraph_data/Weekly Patterns/2019_Weekly_Patterns/"
cols_to_read = [
    "safegraph_place_id",
    "location_name",
    "street_address",
    "city",
    "region",
    "postal_code",
    "iso_country_code",
    "date_range_start",
    "raw_visit_counts",
    "raw_visitor_counts",
    "visits_by_day",
    "visits_by_each_hour",
    "poi_cbg",
    "visitor_home_cbgs",
    "visitor_daytime_cbgs",
    "visitor_country_of_origin",
    "distance_from_home",
    "median_dwell",
    "bucketed_dwell_times",
]

core_places_data_2019 = "../../novus/matchingnemo/scratch/safegraph_data/Core Places Data/CoreRecords-CORE_POI-2019_03-2020-03-25"
city = "Portland"
gdf = gpd.read_file(r"../../novus/matchingnemo/scratch/safegraph_data/Weekly Patterns/Digital_Twins_Analysis/temporary_stash_very_heavy/entire_or_structures_clip.gpkg")
folder_to_save = r"../../novus/matchingnemo/scratch/ampnet_data/Portland"


gdf_subset = gdf[
    [
        "BUILD_ID",
        "OCC_CLS",
        "PRIM_OCC",
        "SQMETERS",
        "SQFEET",
        "CENSUSCODE",
        "UUID",
        "geometry",
    ]
]
gdf_nonresidential = gdf_subset[gdf_subset["OCC_CLS"] != "Residential"]


places_data = pd.read_csv(
    r"../../novus/matchingnemo/scratch/safegraph_data/Weekly Patterns/Digital_Twins_Analysis/temporary_stash_very_heavy/2019_Portland.csv"
)
places_centroid = gpd.GeoDataFrame(
    places_data,
    geometry=gpd.points_from_xy(places_data.longitude, places_data.latitude),
    crs="EPSG:4326",
)

matches = gpd.sjoin(
    gdf_nonresidential, places_centroid, predicate="contains", how="left"
)

schema_overrides = {"date_range_start": pl.Datetime, "distance_from_home": pl.Int64}

for file in os.listdir(data_2019):
    
    read = (
        pl.scan_csv(os.path.join(data_2019, file), schema_overrides=schema_overrides)
        .select(cols_to_read)
        .with_columns(
            [
                pl.col("visits_by_each_hour").str.json_decode(pl.List(pl.Int64)),
                pl.col("visits_by_day").str.json_decode(pl.List(pl.Int64)),
            ]
        )
    )
    
    
    cols_to_select = [
        "safegraph_place_id",
        "city",
        "region",
        "date_range_start",
        "visits_by_each_hour",
    ]
    
    
    read = (
        read.filter(
            pl.col("iso_country_code") == "US",
            pl.col("city") == "Portland",
            pl.col("region") == "OR",
        )
        .select(cols_to_select)
        .with_columns(t=(pl.int_ranges(0, pl.col("visits_by_each_hour").list.len())))
        .explode(["t", "visits_by_each_hour"])
        .rename({"visits_by_each_hour": "visits"})
    )

    read = read.collect()
    read_pd = read.to_pandas()

    week = min(read['date_range_start'].dt.week().to_numpy())
    
    matches["safegraph_place_id"] = matches["safegraph_place_id"].astype(str)
    read_pd["safegraph_place_id"] = read_pd["safegraph_place_id"].astype(str)
    outfile = matches.merge(read_pd, on="safegraph_place_id", how="left")

    cols_in_csv = [
        "BUILD_ID",
        "OCC_CLS",
        "PRIM_OCC",
        "SQMETERS",
        "SQFEET",
        "CENSUSCODE",
        "UUID",
        "safegraph_place_id",
        "date_range_start",
        "t",
        "visits"
    ]

    outfile = outfile.loc[:,cols_in_csv]

    outfile.to_parquet(os.path.join(folder_to_save, f"w{week}.parquet"), engine="pyarrow", index=False)


