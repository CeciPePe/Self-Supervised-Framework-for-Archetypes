import subprocess
import os

def convert_gml_to_shapefile(input_gml_path, output_shapefile_path):
    """
    Converts a GML file to an ESRI Shapefile using ogr2ogr.

    Parameters:
        input_gml_path (str): Path to the input GML file.
        output_shapefile_path (str): Path to the output Shapefile (.shp).
    """
    os.makedirs(os.path.dirname(output_shapefile_path), exist_ok=True)

    subprocess.run([
        "ogr2ogr",
        "-f", "ESRI Shapefile",
        output_shapefile_path,
        input_gml_path
    ], check=True)

