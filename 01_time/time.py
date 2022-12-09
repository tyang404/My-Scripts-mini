# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 09:42:34 2022

@author: yangt
"""
# time tackle
from datetime import datetime
import time
from pytz import timezone

import numpy as np
import pandas as pd

# %% now time

# readable
now_utc = datetime.now(timezone('UTC'))
'datetime.datetime(2022, 10, 11, 14, 45, 22, 481000)'
'2022-10-11 14:45:22.481000'

# unix epoch time
now_ts = time.time()
'1665499835.2203832'

# %% covert now time to US/Central time zone
now_myzone = now_utc.astimezone(timezone('US/Central'))
'2022-10-11 09:45:22.481970-05:00'


# %% time format
format_1 = "%Y-%m-%d %H:%M:%S %Z%z"
format_2 = "%m%d%H%M"
time_stamp = now_myzone.strftime(format_2)
'10110945'

# %% convert unix epoch time to readable
timestamp = 1545730073
time_unix_to_read = pd.to_datetime(timestamp, unit='s', utc=True).tz_convert('US/Pacific')

# %% time stamp
format_1 = "%Y-%m-%d %H:%M:%S %Z%z"
format_2 = "%m%d%H%M"
# US/Central time zone
now_utc = datetime.now(timezone('UTC'))
now_myzone = now_utc.astimezone(timezone('US/Central'))
time_stamp = now_myzone.strftime(format_2)
print(time_stamp)
