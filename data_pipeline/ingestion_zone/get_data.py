import hypercadaster_ES as hc

# Download data for Barcelona municipality
hc.download("./data_pipeline/ingestion_zone/data", cadaster_codes=["08279"],elevation_layer=False)

# Merge all data into a unified GeoDataFrame
gdf = hc.merge("./data_pipeline/ingestion_zone/data", cadaster_codes=["08279"], neighborhood_layer=False,elevations_layer=False)

# Save results
gdf.to_pickle("./daºta_pipeline/ingestion_zone/terrassa.pkl", compression="gzip")

