import os
import pandas as pd
# import pyarrow
# import json
# import time

target_folder = r"../../novus/matchingnemo/scratch/ampnet_data/safegraph_POIs"
core_places_data_2019 = "../../novus/matchingnemo/scratch/safegraph_data/Core Places Data/CoreRecords-CORE_POI-2019_03-2020-03-25"
# cities = ["Portland"]

parquets = []
if (__name__) == '__main__':
    # iterate through the files in 2019 data
    first_iteration = True
    for file in os.listdir(core_places_data_2019):
        if (file.endswith('.csv.gz')):
            curr_file = pd.read_csv(os.path.join((core_places_data_2019),file)).drop('phone_number', axis = 1)

            # curr_file = curr_file[curr_file["city"].isin(cities)]
            parquets.append(curr_file)

    final = pd.concat(parquets, ignore_index = True)
    final.to_parquet(
        os.path.join(target_folder, "2019.parquet"))
