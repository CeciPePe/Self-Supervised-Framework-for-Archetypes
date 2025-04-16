import geopandas as gpd
import pandas as pd
import os

def enrich_building_with_floors(building_path, buildingpart_path, output_path, crs="EPSG:25831"):
    """
    Enriches a building shapefile with the maximum number of floors per building
    using data from a building part shapefile.

    Parameters:
        building_path (str): Path to the building shapefile (.shp).
        buildingpart_path (str): Path to the buildingpart shapefile (.shp).
        output_path (str): Path where the updated shapefile will be saved.
        crs (str): Coordinate reference system to enforce if not already present.
    """

    # Read shapefiles
    building_shp = gpd.read_file(building_path)
    buildingpart_shp = gpd.read_file(buildingpart_path)

    #Satndardize crs for all shapefiles
    if building_shp.crs is None:
        building_shp = building_shp.set_crs(crs, allow_override=True)

    if buildingpart_shp.crs is None:
        buildingpart_shp = buildingpart_shp.set_crs(crs, allow_override=True)

    #prepare to standardize the reference
    building_shp['reference'] = building_shp['reference'].astype(str)
    buildingpart_shp['localId'] = buildingpart_shp['localId'].astype(str)
    buildingpart_shp['reference'] = buildingpart_shp['localId'].str[:14]

    floors_by_reference = buildingpart_shp.groupby('reference')['numberOfFl'].max().reset_index()

    #number of floors for shapefile
    shp1_updated = building_shp.merge(floors_by_reference, on='reference', how='left')
    if 'numberOfFl_y' in shp1_updated.columns:
        shp1_updated['numberOfFl'] = shp1_updated['numberOfFl_y']
        shp1_updated.drop(columns=['numberOfFl_y'], inplace=True)
    elif 'numberOfFl' in shp1_updated.columns:
        # already correctly named
        pass
    else:
        raise KeyError("Column 'numberOfFl' not found after merge.")


    # Save the updated shapefile
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    shp1_updated.to_file(output_path)
    print(f"Shapefile successfully written to {output_path}")

