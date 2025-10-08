#!/bin/bash

# MARL Building Energy Estimation - Automated Dataset Runner
echo "Starting automated MARL training for all datasets..."
echo "Started at: $(date)"
echo "Keeping Mac awake with caffeinate"
echo ""

# Keep Mac awake and run datasets sequentially
caffeinate -i bash -c '
    echo "Running: cultural dataset..."
    python main.py --dataset cultural --dataset-m Cultural
    echo "✅ Cultural completed!"
    
    echo "Running: commercial dataset..."
    python main.py --dataset commercial --dataset-m Commercial  
    echo "✅ Commercial completed!"
    
    echo "Running: entertainment_venues dataset..."
    python main.py --dataset entertainment_venues --dataset-m Entertainment_venues
    echo "✅ Entertainment venues completed!"
    
    echo "Running: industrial dataset..."
    python main.py --dataset industrial --dataset-m Industrial
    echo "✅ Industrial completed!"
    
    echo "Running: leisure_and_hospitality dataset..."
    python main.py --dataset leisure_and_hospitality --dataset-m Leisure_and_hospitality
    echo "✅ Leisure and hospitality completed!"
    
    echo "Running: offices dataset..."
    python main.py --dataset offices --dataset-m Offices
    echo "✅ Offices completed!"
    
    echo "Running: residential dataset..."
    python main.py --dataset residential --dataset-m Residential
    echo "✅ Residential completed!"
    
    echo "Running: singular_building dataset..."
    python main.py --dataset singular_building --dataset-m Singular_building
    echo "✅ Singular building completed!"
    
    echo "Running: sports_facilities dataset..."
    python main.py --dataset sports_facilities --dataset-m Sports_facilities
    echo "✅ Sports facilities completed!"
    
    echo "Running: urbanization_land dataset..."
    python main.py --dataset urbanization_land --dataset-m Urbanization_land
    echo "✅ Urbanization land completed!"
    
    echo "Running: warehouse_parking dataset..."
    python main.py --dataset warehouse_parking --dataset-m Warehouse_parking
    echo "✅ Warehouse parking completed!"
'

echo "ALL DATASETS COMPLETED!"
echo "Finished at: $(date)"
echo "Check ./model_accuracy_results.txt for all results"