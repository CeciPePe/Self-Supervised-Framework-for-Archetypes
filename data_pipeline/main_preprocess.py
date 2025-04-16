from ingestion_zone.get_inspiredata import *
from trusted_zone.gml_to_shp import convert_gml_to_shapefile
from trusted_zone.shp_to_geojson import convert_shp_to_geojson
from trusted_zone.preprocess_catastrodata import enrich_building_with_floors
from trusted_zone.extract_img_from_shp import export_geojson_features_as_images

def run_preprocess():
    """
    Runs all preprocessing steps for the trusted zone data.
    """
    #get INSPIRE data
    download(data_to_download=atom_urls['buildings'], provincia=8, municipio=279)

    convert_gml_to_shapefile(
        "data_pipeline/landing_zone/raw_data/A.ES.SDGC.BU.08279.buildingpart.gml",
        "data_pipeline/trusted_zone/preprocessed_data/A.ES.SDGC.BU.08279.buildingpart.shp"
    )

    convert_gml_to_shapefile(
        "data_pipeline/landing_zone/raw_data/A.ES.SDGC.BU.08279.building.gml",
        "data_pipeline/trusted_zone/preprocessed_data/A.ES.SDGC.BU.08279.building.shp"
    )

    # Enrich building with floors
    enrich_building_with_floors(
        building_path="data_pipeline/trusted_zone/preprocessed_data/A.ES.SDGC.BU.08279.building.shp",
        buildingpart_path="data_pipeline/trusted_zone/preprocessed_data/A.ES.SDGC.BU.08279.buildingpart.shp",
        output_path="data_pipeline/trusted_zone/preprocessed_data/cleaned_catastro.shp"
    )

    # Convert shapefile to GeoJSON
    convert_shp_to_geojson(
        "data_pipeline/trusted_zone/preprocessed_data/cleaned_catastro.shp",
        "data_pipeline/trusted_zone/preprocessed_data/cleaned_catastro.geojson"
    )

    # Extract images from GeoJSON
    export_geojson_features_as_images(
        "data_pipeline/trusted_zone/preprocessed_data/cleaned_catastro.geojson",
        "data/data_root_1"
    )

if __name__ == "__main__":
    run_preprocess()    