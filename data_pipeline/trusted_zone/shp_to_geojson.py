import geopandas as gpd
import os

def convert_shp_to_geojson(input_shp_path, output_geojson_path, crs="EPSG:25831"):
    """
    Converts a shapefile to GeoJSON and sets CRS if missing.

    Parameters:
        input_shp_path (str): Path to the input shapefile (.shp).
        output_geojson_path (str): Path to the output GeoJSON file.
        crs (str): Coordinate Reference System to assign if missing.
    """
    gdf = gpd.read_file(input_shp_path)

    if gdf.crs is None:
        gdf.set_crs(crs, inplace=True)
    os.makedirs(os.path.dirname(output_geojson_path), exist_ok=True)

    gdf.to_file(output_geojson_path, driver="GeoJSON")

