import osmnx as ox
import pandas as pd
place = "Terrassa, Spain"
G = ox.graph_from_place(place, network_type="drive")
gdf_edges = ox.graph_to_gdfs(G, nodes=False)
if 'width' in gdf_edges.columns:
    gdf_edges['width'] = pd.to_numeric(gdf_edges['width'], errors='coerce')
else:
    gdf_edges['width'] = None
default_widths = {
    'motorway': 25,
    'primary': 15,
    'secondary': 12,
    'tertiary': 10,
    'residential': 7,
    'service': 5
}
def get_highway_type(hw):
    if isinstance(hw, list):
        return hw[0]
    return hw

gdf_edges['highway_type'] = gdf_edges['highway'].apply(get_highway_type)
gdf_edges['estimated_width'] = gdf_edges.apply(
    lambda row: row['width'] if pd.notnull(row['width']) else default_widths.get(row['highway_type'], 6),
    axis=1
)
streets_df = gdf_edges[['name', 'highway_type', 'width', 'estimated_width']].copy()
streets_df = streets_df.dropna(subset=['name'])  # Keep only streets with names
streets_df = streets_df.drop_duplicates(subset=['name'])
print(streets_df.head())
streets_df.to_csv("data_pipeline/trusted_zone/preprocessed_data/terrassa_street_widths.csv", index=False)
