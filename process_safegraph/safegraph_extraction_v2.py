import os, sys
import pandas as pd
import polars as pl
import pyarrow
import numpy as np
import json
import time
from datetime import date, datetime 


pl.scan_csv(os.path.join(r"//depot.engr.oregonstate.edu/mime_u1/dalziel/Safe Graph Data/Weekly Patterns/2019_Weekly_Patterns/","*.csv.gz")).sink_parquet(r"//depot.engr.oregonstate.edu/mime_u1/dalziel/Safe Graph Data/Weekly Patterns/Digital_Twins_Analysis/test.parquet")

