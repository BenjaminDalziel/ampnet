import geopandas as gpd
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import collections
import numpy as np
from shapely.geometry import Point

# call: python3 augment.py "//path_to_Safe Graph Data_folder" city
# example python3 augment.py "//depot.engr.oregonstate.edu/mime_u1/dalziel/Safe Graph Data" Portland

# TODO: take this path as a command line argument as well
# gdf = gpd.read_file(r"\\stak.engr.oregonstate.edu\Users\len2\research\fema_data\raw\or_structures_clip.gpkg")
# gdf_subset = gdf[['BUILD_ID', 'OCC_CLS', 'PRIM_OCC', 'SQMETERS', 'SQFEET', 'CENSUSCODE', 'UUID', 'geometry']]

def find_pois_contained_in_fema_geometry(fema_data, safegraph_place_data):
    # given a fema dataset and a safegraph dataset, loop through the sg pois and see what fema build id it's within, then write the poi id to fema dataset as match
    places_city = safegraph_place_data[safegraph_place_data['city']==city]
    sg_match = ['' for _ in range(len(fema_data))]
    for poi in places_city.itertuples(index= False):
        poi_point = Point(getattr(poi, 'longitude'), getattr(poi, 'latitude'))
        poi_id = getattr(poi, 'safegraph_place_id')
        fema_data['contains'] = fema_data['geometry'].contains(poi_point)
        match_indices = fema_data[fema_data['contains']==True].index
        if match_indices.size == 0:
            continue
        match_ind = np.random.choice(match_indices)
        sg_match[match_ind] = poi_id
    fema_data['sg_match'] = sg_match

    return fema_data

def main():
    args = sys.argv[1:]
    dta_dir_prefix = args[0]
    city = args[1]
    places_data = pd.read_csv(os.path.join(fr"{dta_dir_prefix}","Weekly Patterns", "Digital_Twins_Analysis", "Places", "2019.csv"))

    gdf = gpd.read_file(r"\\stak.engr.oregonstate.edu\Users\len2\research\fema_data\raw\or_structures_clip.gpkg")
    gdf_subset = gdf[['BUILD_ID', 'OCC_CLS', 'PRIM_OCC', 'SQMETERS', 'SQFEET', 'CENSUSCODE', 'UUID', 'geometry']]

    week_start = [x[:10] for x in os.listdir(os.path.join(dta_dir_prefix, "Weekly Patterns", "Digital_Twins_Analysis", "2019", "2019_weekly", city))]
    gdf_nonresidential = gdf_subset[gdf_subset['OCC_CLS'] != 'Residential']

    gdf_nonresidential = find_pois_contained_in_fema_geometry(gdf_nonresidential.reset_index(), places_data)

    gdf_subset = pd.merge(gdf_subset, gdf_nonresidential[['BUILD_ID', 'sg_match']], on='BUILD_ID', how='left')

    for week_start_date in week_start:
        visitation_file = pd.read_csv(os.path.join(fr"{dta_dir_prefix}","Weekly Patterns", "Digital_Twins_Analysis", "2019", "2019_weekly",f"{city}", f"{week_start_date}.csv"))
        visitation_file = visitation_file[['safegraph_place_id', 'median_dwell'] + [f'visits_h{x}' for x in range(168)]]
        safegraph_file = visitation_file.merge(places_data, on='safegraph_place_id', how='inner')

        fema_file = gdf_subset.merge(gdf['BUILD_ID'], how='right',on='BUILD_ID')
        augmented_file = fema_file.merge(safegraph_file, how='left', left_on = 'sg_match', right_on = 'safegraph_place_id')

        augmented_file.to_csv(os.path.join(fr"{dta_dir_prefix}","Weekly Patterns", "Digital_Twins_Analysis", "2019", "fema_augmented", f"{city}", f"{week_start_date}.csv"))

if __name__ == "__main__":
    main()