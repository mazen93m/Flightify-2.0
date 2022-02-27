# FAA Dataset Cleansing Tool

1. This data cleansing tool is desgined to handle any OPSNET : Tower Operations : Standard Report saved as a csv file. 
2. The tool accounts for unnecessary leading headers and footer text, data reformatting, column labelling, column creation, 
and datatype conversion. 
3. This file currently looks at the fairbanks.csv file, but it can and should be modified in the pd.read_csv() line to take in any such csv file 
from the OPSNET Tower Ops Report builder
