import os
import pandas as pd
# import pyarrow
# import json
# import time

target_folder = r"\\depot.engr.oregonstate.edu\mime_u1\dalziel\Safe Graph Data\Weekly Patterns\Digital_Twins_Analysis\temporary_stash_very_heavy"
core_places_data_2019 = "//depot.engr.oregonstate.edu/mime_u1/dalziel/Safe Graph Data/Core Places Data/CoreRecords-CORE_POI-2019_03-2020-03-25"
cities = ["Portland"]

if (__name__) == '__main__':
    # iterate through the files in 2019 data
    first_iteration = True
    for file in os.listdir(core_places_data_2019):
        if (file.endswith('.csv.gz')):
            curr_file = pd.read_csv(os.path.join((core_places_data_2019),file)).drop('phone_number', axis = 1)
            curr_file = curr_file[curr_file["city"].isin(cities)]
            if first_iteration:
                curr_file.to_csv(os.path.join(target_folder,"2019.csv"),mode="a",index=False)
                first_iteration = False
            else:
                curr_file.to_csv(os.path.join(target_folder,"2019.csv"),mode="a",index=False,header=False)

