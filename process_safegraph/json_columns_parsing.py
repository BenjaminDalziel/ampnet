import polars as pl
import json
import time
import ast
import pyarrow
import pandas as pd


#! Heavily commented because the thought process is very messy
def read_visits_by_each_hour(polars_df, column):
    """
    Input:
        polars_df : a polars dataframe
        column: name of the column with hourly visit (hourly visit data must be a string representation of a list)
    Output: 
        the original polars dataframe with its hourly visit column exploded out
    """
    # This is a funnily unoptimized function but we get there when we get there 
    # create a sub-dataframe with only the column of interest
    subfile = polars_df.select(column)
    # evaluate all the list strings in the column
    subfile = subfile.with_columns(pl.col(column).map_elements(eval,return_dtype=list[int]))
    # I think I'll just hardcode this tbh no function can do the formatting for me
    # Explode the lists into length_of_list columns
    subfile = subfile.with_row_index().explode(column).with_columns(on=pl.format('visits_h{}', pl.int_range(0,168).over('index'))).pivot(on="on",index=['index'])
    # drop the index column (byproduct of above line of code)
    subfile = subfile.drop(["index"])
    # concatenate horizontally with the original dataframe
    polars_df = pl.concat([polars_df,subfile],how='horizontal')
    polars_df = polars_df.drop(column)
    return polars_df

def read_visits_by_each_day(polars_df, column):
    """
    Input:
        polars_df : a polars dataframe
        column: name of the column with daily visit (daily visit data must be a string representation of a list)
    Output: 
        the original polars dataframe with its daily visit column exploded out
    """
    subfile = polars_df.select(column)
    subfile = subfile.with_columns(pl.col(column).map_elements(eval,return_dtype=list[int]))
    # Only difference between this and the above function: the "visits_d" formatting instead of "visits_h". Somehow I genuinely cannot make it work using fstrings. We get there when we get there
    subfile = subfile.with_row_index().explode(column).with_columns(on=pl.format('visits_d{}', pl.int_range(0,7).over('index')+1)).pivot(on="on",index=['index'])
    subfile = subfile.drop(["index"])
    polars_df = pl.concat([polars_df,subfile],how='horizontal')
    polars_df = polars_df.drop(column)
    return polars_df

def read_home_cbg_column(polars_df, home_cbg_column_name):
    """
    Input:
        polars_df : a polars dataframe
        home_cbg_column_name: name of the column with semi-properly-formatted JSON string of visitors' home cbgs
    Output: 
        the original polars dataframe with its visitors' daytime cbg column exploded out with column names in the format h_cbg:{cbg provided as key in the json dictionary}   
    """
    subfile = polars_df.select(home_cbg_column_name)
    subfile = polars_df.to_pandas()
    subfile[home_cbg_column_name] = subfile[home_cbg_column_name].apply(json.loads)
    subfile = pd.json_normalize(subfile[home_cbg_column_name]).fillna(0).map(int)
    subfile = subfile.rename(columns = lambda cbg: f'h_cbg:{cbg}')
    subfile = pl.from_pandas(subfile)
    polars_df = pl.concat([polars_df,subfile],how='horizontal')
    polars_df = polars_df.drop(home_cbg_column_name)
    return polars_df

def read_daytime_cbg_column(polars_df, daytime_cbg_column_name):
    """Parse the safegraph visitors_daytime_cbgs column 

    Args:
        polars_df (polars dataframe): A polars.DataFrame object
        daytime_cbg_column_name (string): name of the column with visitors' daytime cbgs

    Returns:
        polars dataframe: same dataframe passed in except with the bucketed_dwell_times column expanded out into {"d_cbg:{cbg}": {visit_count}}
    """
    subfile = polars_df.select(daytime_cbg_column_name)
    subfile = polars_df.to_pandas()
    subfile[daytime_cbg_column_name] = subfile[daytime_cbg_column_name].apply(json.loads)
    subfile = pd.json_normalize(subfile[daytime_cbg_column_name]).fillna(0).map(int)
    subfile = subfile.rename(columns = lambda cbg: f'd_cbg:{cbg}')
    subfile = pl.from_pandas(subfile)
    polars_df = pl.concat([polars_df,subfile],how='horizontal')
    polars_df = polars_df.drop(daytime_cbg_column_name)
    return polars_df

def read_bucketed_dwell_times(polars_df):
    """Parse the safegraph bucketed_dwell_times column

    Args:
        polars_df (polars dataframe): A polars.DataFrame object

    Returns:
        polars dataframe: same dataframe passed in except with the bucketed_dwell_times column expanded out into "<5", "5-20", "21-60", "61-240", ">241" with corresponding bucketed dwell counts (int)
    """
    subfile = polars_df.select("bucketed_dwell_times")
    subfile = polars_df.to_pandas()
    subfile["bucketed_dwell_times"] = subfile["bucketed_dwell_times"].apply(json.loads)
    subfile = pd.json_normalize(subfile["bucketed_dwell_times"]).fillna(0).map(int)
    subfile = pl.from_pandas(subfile)
    polars_df = pl.concat([polars_df,subfile],how='horizontal')
    polars_df = polars_df.drop("bucketed_dwell_times")
    return polars_df

if __name__ == "__main__": 
    pass