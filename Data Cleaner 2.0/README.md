# Intructions on Opening datasets (zipped .pkl file)

1. Unzip file
2. Run the following code
3. Make sure the pickle package is installed

```python
import pickle

# Setting any output to display all columns and rows 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# #Read in model dictionary from disk

file_to_read = open("datasets.pkl", "rb")
#
datasets = pickle.load(file_to_read)

gfk = datasets['GFK']

print(gfk.head())
```
```
  Date  LOC      STATION                                      NAME  \
0  2017-01-01  GFK  USW00014916  GRAND FORKS INTERNATIONAL AIRPORT, ND US   
1  2017-01-02  GFK  USW00014916  GRAND FORKS INTERNATIONAL AIRPORT, ND US   
2  2017-01-03  GFK  USW00014916  GRAND FORKS INTERNATIONAL AIRPORT, ND US   
3  2017-01-04  GFK  USW00014916  GRAND FORKS INTERNATIONAL AIRPORT, ND US   
4  2017-01-05  GFK  USW00014916  GRAND FORKS INTERNATIONAL AIRPORT, ND US   

   LATITUDE  LONGITUDE  VFR  IFR   AWND  PRCP  PRCP_SQRT  SNOW  TMIN  TMAX  \
0  47.94281  -97.18294   10   13   8.50  0.00   0.000000   0.0    16    23   
1  47.94281  -97.18294    0   16  16.11  0.20   0.447214   4.1     7    17   
2  47.94281  -97.18294    3   17  21.25  0.07   0.264575   0.7   -10     7   
3  47.94281  -97.18294   28   64   4.92  0.00   0.000000   0.0   -22    -2   
4  47.94281  -97.18294  128  112   7.38  0.00   0.000000   0.0   -24    -6   

   isAHoliday  SNOW_SQRT  
0         0.0   0.000000  
1         1.0   2.024846  
2         0.0   0.836660  
3         0.0   0.000000  
4         0.0   0.000000  


```
