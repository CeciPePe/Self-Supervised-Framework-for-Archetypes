# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:21:52 2025

@author: cperez
"""
import os
import geopandas as gpd
import matplotlib.pyplot as plt

def export_geojson_features_as_images(geojson_file, output_folder):
    """
    Export each feature in a GeoJSON file as an individual PNG image.
    Parameters:
        geojson_file (str): Path to the input GeoJSON file.
        output_folder (str): Directory where PNG images will be saved.
    """
    gdf = gpd.read_file(geojson_file)
    os.makedirs(output_folder, exist_ok=True)

    for idx, row in gdf.iterrows():
        fig, ax = plt.subplots()
        gdf.iloc[[idx]].plot(ax=ax, edgecolor="black", facecolor="lightblue")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        ref = str(row.get("reference", f"feature_{idx + 1}")).replace("/", "_").replace("\\", "_")
        output_path = os.path.join(output_folder, f"{ref}.png")
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close()

    print(f"Exported {len(gdf)} images to {output_folder}")

# Example usage:
# export_geojson_features_as_images(
#     "data_pipeline/trusted_zone/preprocessed_data/cleaned_catastro.geojson",
#     "/Users/ceciliaperez/Documents/UPC- MD/TFM/Data/data_root"
# )


