# Import packages
import pandas as pd
from tabulate import tabulate

# Import dataframes
neighborhoodAtlasDF = pd.read_csv('https://ahistorage.blob.core.windows.net/507/Neighborhood_atlas.csv')
SPARCS = pd.read_json('https://health.data.ny.gov/resource/82xm-y6g8.json')

# Column checking
SPARCS.columns 
neighborhoodAtlasDF.columns

# Column cleaning
SPARCS.columns = SPARCS.columns.str.replace('[^A-Za-z0-9]+', '_')
neighborhoodAtlasDF.columns = neighborhoodAtlasDF.columns.str.replace('[^A-Za-z0-9]+', '_')
SPARCS.columns = SPARCS.columns.str.lower()
neighborhoodAtlasDF.columns = neighborhoodAtlasDF.columns.str.lower()

# Column type checking
SPARCS.dtypes
neighborhoodAtlasDF.dtypes

# Smaller dataframes
SPARCSsmall = SPARCS[['hospital_county', 'facility_name',  'age_group', 'gender',  'race', 'zip_code_3_digits']]
print(SPARCSsmall.sample(10).to_markdown())
SPARCSsmall.shape

neighborhoodAtlasDFsmall = neighborhoodAtlasDF[['type', 'zipid', 'adi_natrank']]
print(neighborhoodAtlasDFsmall.sample(10).to_markdown())
neighborhoodAtlasDFsmall.shape

# Merge dataframes
mergedDF = SPARCSsmall.merge(neighborhoodAtlasDFsmall, how='left', left_on='zip_code_3_digits', right_on='zipid')
mergedDF = pd.merge(SPARCSsmall, neighborhoodAtlasDFsmall, how='left', left_on='zip_code_3_digits', right_on='zipid')
mergedDF.shape

# Merged df to csv
mergedDF.to_csv('combined_data.csv')