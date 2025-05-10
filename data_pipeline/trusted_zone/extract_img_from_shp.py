# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:21:52 2025

@author: cperez
"""
import os
import geopandas as gpd
import matplotlib.pyplot as plt

def export_geojson_features_as_images(geojson_file, output_folder, image_size=256):
    """
    Export each feature in a GeoJSON file as an individual PNG image of fixed pixel size.

    Parameters:
        geojson_file (str): Path to the input GeoJSON file.
        output_folder (str): Directory where PNG images will be saved.
        image_size (int): Size in pixels (width and height) for each output image.
    """
    gdf = gpd.read_file(geojson_file)
    os.makedirs(output_folder, exist_ok=True)

    dpi = 100  # Dots per inch
    inches = image_size / dpi  # Convert desired pixel size to inches

    for idx, row in gdf.iterrows():
        fig, ax = plt.subplots(figsize=(inches, inches), dpi=dpi)
        gdf.iloc[[idx]].plot(ax=ax, edgecolor="black", facecolor="lightblue")

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()

        ref = str(row.get("reference", f"feature_{idx + 1}")).replace("/", "_").replace("\\", "_")
        output_path = os.path.join(output_folder, f"{ref}.png")

        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.close()

    print(f"Exported {len(gdf)} images to {output_folder}")

