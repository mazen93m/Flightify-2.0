# Data Cleaning Tool

`dataCleaner.py` is a program that prepares, manipulates, and merges FAA, weather, and holiday data in preparation for the predictive module. 
The tool utilizes a host of user defined functions that work together to prepare the data for predictive modeling. 

## How it works
1.	`dataCleaner.py` imports `airports.py`, which is a:
    - Python module that houses a dictionary (`airports_dict`), which includes the airport code as the key, a list of csv file names
      (one representing the FAA dataset and the other representing the NOAA dataset), and the airport’s name as displayed in the `NAME` column
      of the NOAA weather dataset. 
    - Module includes the `airports_dict` dictionary that lists and organizes the lookup data for each airport.
2.	`dataCleaner.py` unpacks the airports’ FAA and NOAA datasets by extracting the pandas dataframes and saves each airport’s FAA and NOAA
    datasets as list elements in a list of lists (`dataFrameLst`). 
3.	The tool then performs several data cleaning steps including data manipulation, data type conversion, variable creation etc. 
4.	Each airport’s datasets are extracted from the `dataFrameLst`, merged together on the `Date` column, and saved as an element of a list (`merged_Lst`). 
5.	Once the dataframes are merged, the tool adds the Location ID (`LOC`) to each dataset. It achieves this by matching the 3rd element of the
    `airports_dict` at the appropriate index to the `NAME` column of each dataset and grabbing the key element as the value for `LOC`.
6.  The tool accounts for any null values that may be present in any of the individual datasets' columns (typically one or more weather variables.
7.	Finally, the tool subsets the applicable columns for each cleaned dataframe and saves each dataframe as a value to a
    python dictionary (`datasets`) with the appropiate `LOC` as the key.


## How to use it
The `datasets` dictionary from `dataCleaner.py` can be used in a python file by importing the module and  asigning the `airports_dict` from the `airports.py` module to a local variable.

```python
import dataCleaner as dc
import pandas as pd

data = dc.datasets
data['ANC'].head()
```
```
         Date  LOC      STATION  \
0  2017-01-01  ANC  USW00026451   
1  2017-01-02  ANC  USW00026451   
2  2017-01-03  ANC  USW00026451   
3  2017-01-04  ANC  USW00026451   
4  2017-01-05  ANC  USW00026451   

                                                NAME  LATITUDE  LONGITUDE  \
0  ANCHORAGE TED STEVENS INTERNATIONAL AIRPORT, A...  61.16916 -150.02771   
1  ANCHORAGE TED STEVENS INTERNATIONAL AIRPORT, A...  61.16916 -150.02771   
2  ANCHORAGE TED STEVENS INTERNATIONAL AIRPORT, A...  61.16916 -150.02771   
3  ANCHORAGE TED STEVENS INTERNATIONAL AIRPORT, A...  61.16916 -150.02771   
4  ANCHORAGE TED STEVENS INTERNATIONAL AIRPORT, A...  61.16916 -150.02771   

   VFR  IFR  AWND  PRCP  PRCP_SQRT  SNOW  TMIN  TMAX  isAHoliday  
0   67  255  2.01   0.0        0.0   0.0     5    17         1.0  
1    5  406  2.01   0.0        0.0   0.0     4    15         0.0  
2    4  398  1.79   0.0        0.0   0.0     3    14         0.0  
3   32  529  2.24   0.0        0.0   0.0     3    17         0.0  
4  100  487  4.70   0.0        0.0   0.0     1    25         0.0  
```

