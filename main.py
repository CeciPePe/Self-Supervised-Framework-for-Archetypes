#!/usr/bin/env python3
"""
Main pipeline script for MARL Building Energy Estimation
Runs the complete pipeline: VQAE training -> MARL training -> Latent clustering
"""

import os
import sys
import subprocess
import json
import glob
import argparse
from pathlib import Path

# Add project root to path
sys.path.append('.')

def find_best_checkpoint(checkpoint_dir, pattern="*-error-*.json", dataset_name=None):
    """Find the best checkpoint based on lowest error value, optionally filtered by dataset"""
    error_files = glob.glob(os.path.join(checkpoint_dir, pattern))
    
    if not error_files:
        raise FileNotFoundError(f"No checkpoint files found matching pattern {pattern}")
    
    best_error = float('inf')
    best_checkpoint = None
    
    for error_file in error_files:
        try:
            with open(error_file, 'r') as f:
                error_data = json.load(f)
                
                # Try to get the last value from test_recon_error or train_recon_error arrays
                error_value = float('inf')
                if 'test_recon_error' in error_data and error_data['test_recon_error']:
                    error_value = float(error_data['test_recon_error'][-1])  # Last epoch value
                elif 'train_recon_error' in error_data and error_data['train_recon_error']:
                    error_value = float(error_data['train_recon_error'][-1])  # Last epoch value
                else:
                    # Fallback: try to parse error from filename
                    import re
                    match = re.search(r'error-([0-9.]+)\.json', error_file)
                    if match:
                        error_value = float(match.group(1))
                
                if error_value < best_error:
                    best_error = error_value
                    best_checkpoint = error_file
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Warning: Could not parse {error_file}: {e}")
            continue
    
    if best_checkpoint is None:
        raise ValueError("No valid checkpoint files found")
    
    print(f"Best checkpoint found: {best_checkpoint} with error: {best_error}")
    return best_checkpoint, best_error

def run_vqae_training(dataset_name, dataset_name_m):
    """Run VQAE training (train.py)"""
    print("=" * 60)
    print("STEP 1: Running VQAE Training")
    print("=" * 60)
    
    # Modify train.py to use the specified dataset
    train_script = "train.py"
    
    # Read the current train.py
    with open(train_script, 'r') as f:
        content = f.read()
    
    # Replace dataset variables - comprehensive replacement
    print(f"Replacing dataset variables: {dataset_name} -> {dataset_name_m}")
    
    # Define all possible hardcoded dataset names
    hardcoded_datasets = ['cultural', 'residential', 'commercial', 'industrial', 'entertainment_venues', 'healthcare_and_charity', 'offices', 'leisure_and_hospitality', 'sports_facilities', 'warehouse_parking', 'urbanization_land', 'singular_building']
    hardcoded_datasets_m = ['Cultural', 'Residential', 'Commercial', 'Industrial', 'Entertainment_venues', 'Healthcare_and_charity', 'Offices', 'Leisure_and_hospitality', 'Sports_facilities', 'Warehouse_parking', 'Urbanization_land', 'Singular_building']
    
    # Replace all possible hardcoded dataset names
    for old_dataset in hardcoded_datasets:
        content = content.replace(f"dataset_name = '{old_dataset}'", f"dataset_name = '{dataset_name}'")
    
    for old_dataset_m in hardcoded_datasets_m:
        content = content.replace(f"dataset_name_m ='{old_dataset_m}'", f"dataset_name_m ='{dataset_name_m}'")
        content = content.replace(f"dataset_name_m = '{old_dataset_m}'", f"dataset_name_m = '{dataset_name_m}'")
    
    # Also need to modify FloorPlanLoader.py temporarily
    floorplan_loader_path = "data/FloorPlanLoader.py"
    with open(floorplan_loader_path, 'r') as f:
        loader_content = f.read()
    
    # Backup original and create modified version
    backup_path = "data/FloorPlanLoader_backup.py"
    with open(backup_path, 'w') as f:
        f.write(loader_content)
    
    # Modify FloorPlanLoader - comprehensive replacement
    for old_dataset in hardcoded_datasets:
        loader_content = loader_content.replace(f"dataset_name = '{old_dataset}'", f"dataset_name = '{dataset_name}'")
    
    for old_dataset_m in hardcoded_datasets_m:
        loader_content = loader_content.replace(f"dataset_name_m ='{old_dataset_m}'", f"dataset_name_m ='{dataset_name_m}'")
        loader_content = loader_content.replace(f"dataset_name_m = '{old_dataset_m}'", f"dataset_name_m = '{dataset_name_m}'")
    
    # Fix the hardcoded data_config default value
    if dataset_name == 'residential':
        correct_data_config = f'data/data_config_1/residential/{dataset_name_m}/'
    else:
        correct_data_config = f'data/data_config_1/tertiary/{dataset_name_m}/'
    
    # Replace the hardcoded data_config default value
    loader_content = loader_content.replace(
        f"data_config=f'data/data_config_1/tertiary/Commercial/',",
        f"data_config='{correct_data_config}',"
    )
    
    with open(floorplan_loader_path, 'w') as f:
        f.write(loader_content)
    
    print(f"Temporarily modified FloorPlanLoader.py for dataset: {dataset_name_m}")
    
    # Debug: Show what dataset variables are in the content after replacement
    import re
    dataset_matches = re.findall(r"dataset_name = '[^']*'", content)
    dataset_m_matches = re.findall(r"dataset_name_m = ?'[^']*'", content)
    print(f"VQAE Dataset variables after replacement: {dataset_matches}")
    print(f"VQAE Dataset_m variables after replacement: {dataset_m_matches}")
    
    # Fix the hardcoded residential path in train.py
    if dataset_name == 'residential':
        data_config_path = f'./data/data_config_1/residential/{dataset_name_m}/'
    else:
        data_config_path = f'./data/data_config_1/tertiary/{dataset_name_m}/'
    
    # Replace the specific hardcoded line in train.py - handle both residential and tertiary
    old_line_residential = "data_config=f'./data/data_config_1/residential/{dataset_name_m}/'"
    old_line_tertiary = "data_config=f'./data/data_config_1/tertiary/{dataset_name_m}/'"
    new_line = f"data_config='{data_config_path}'"
    content = content.replace(old_line_residential, new_line)
    content = content.replace(old_line_tertiary, new_line)
    
    # Also handle variations without the f-string
    content = content.replace(
        "./data/data_config_1/residential/", 
        f"./data/data_config_1/{'residential' if dataset_name == 'residential' else 'tertiary'}/"
    )
    
    # Write modified content to temporary file
    temp_train = "temp_train.py"
    with open(temp_train, 'w') as f:
        f.write(content)
    
    print(f"Created temporary file: {temp_train}")
    print("Key lines in temp file:")
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'dataset_name' in line:
            print(f"  Line {i+1}: {line}")
    
    try:
        # Run the training with real-time output
        print("Starting VQAE training...")
        process = subprocess.Popen([sys.executable, temp_train], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT,
                                 universal_newlines=True,
                                 cwd='.')
        
        # Capture all output for parsing
        full_output = ""
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                full_output += output
        
        return_code = process.poll()
        if return_code != 0:
            print(f"VQAE training failed with return code {return_code}")
            return False
        
        print("VQAE training completed successfully!")
        
        # Parse the training output to find the best epoch and error
        # Look for patterns like "Best epoch:26" and "Validation Loss: 6.029980329796672"
        best_epoch = None
        best_error = None
        
        # Use the captured output
        output = full_output
        
        # Find the best epoch
        import re
        epoch_match = re.search(r'Best epoch:(\d+)', output)
        if epoch_match:
            best_epoch = int(epoch_match.group(1))
        
        # Find the validation loss that corresponds to the best epoch
        # Look for the pattern: "✅ Best model updated at epoch X with val_loss=Y.YYYY"
        if best_epoch is not None:
            best_epoch_pattern = rf'✅ Best model updated at epoch {best_epoch} with val_loss=([\d.]+)'
            best_epoch_match = re.search(best_epoch_pattern, output)
            if best_epoch_match:
                best_error = float(best_epoch_match.group(1))
            else:
                # Fallback: look for the last validation loss before "Best epoch"
                val_loss_matches = re.findall(r'Validation Loss: ([\d.]+)', output)
                if val_loss_matches:
                    best_error = float(val_loss_matches[-1])
        
        if best_epoch is not None and best_error is not None:
            checkpoint_filename = f"./checkpoint/{best_epoch}-error-{best_error}.json"
            if os.path.exists(checkpoint_filename):
                print(f"Found best checkpoint: {checkpoint_filename}")
                return True, checkpoint_filename
            else:
                print(f"Warning: Expected checkpoint {checkpoint_filename} not found")
        
        # Fallback: Find the most recently created checkpoint
        checkpoint_files = glob.glob("./checkpoint/*-error-*.json")
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
            print(f"Fallback: Using most recent checkpoint: {latest_checkpoint}")
            return True, latest_checkpoint
        else:
            print("Warning: No checkpoint files found")
            return True, None
        
    finally:
        # Restore original FloorPlanLoader.py
        backup_path = "data/FloorPlanLoader_backup.py"
        floorplan_loader_path = "data/FloorPlanLoader.py"
        if os.path.exists(backup_path):
            with open(backup_path, 'r') as f:
                original_content = f.read()
            with open(floorplan_loader_path, 'w') as f:
                f.write(original_content)
            os.remove(backup_path)
            print("Restored original FloorPlanLoader.py")
        
        # Keep temporary file for debugging
        print(f"Temporary file kept for inspection: {temp_train}")
        # if os.path.exists(temp_train):
        #     os.remove(temp_train)

def run_marl_training(dataset_name, dataset_name_m, best_checkpoint_path):
    """Run MARL training using the best VQAE checkpoint"""
    print("=" * 60)
    print("STEP 2: Running MARL Training")
    print("=" * 60)
    
    # Convert notebook to Python script and run it
    notebook_path = "notebooks/train_marl.ipynb"
    
    # Use nbconvert to convert notebook to Python
    convert_cmd = [
        "jupyter", "nbconvert", 
        "--to", "python", 
        "--output", "temp_train_marl",
        "--output-dir", ".",  # Output to current directory
        notebook_path
    ]
    
    try:
        print(f"Converting notebook {notebook_path} to Python...")
        result = subprocess.run(convert_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Failed to convert notebook: {result.stderr}")
            return False
        
        # Read the converted Python file
        python_file = "temp_train_marl.py"
        
        # Check if file exists
        if not os.path.exists(python_file):
            print(f"Error: Converted file {python_file} not found")
            return False
        with open(python_file, 'r') as f:
            content = f.read()
        
        # Modify dataset variables
        print(f"Replacing dataset variables in MARL script: {dataset_name} -> {dataset_name_m}")
        
        # Define all possible hardcoded dataset names
        hardcoded_datasets = ['cultural', 'residential', 'commercial', 'industrial', 'entertainment_venues', 'healthcare_and_charity', 'offices', 'leisure_and_hospitality', 'sports_facilities', 'warehouse_parking', 'urbanization_land', 'singular_building']
        hardcoded_datasets_m = ['Cultural', 'Residential', 'Commercial', 'Industrial', 'Entertainment_venues', 'Healthcare_and_charity', 'Offices', 'Leisure_and_hospitality', 'Sports_facilities', 'Warehouse_parking', 'Urbanization_land', 'Singular_building']
        
        # Replace all possible hardcoded dataset names
        for old_dataset in hardcoded_datasets:
            content = content.replace(f"dataset_name = '{old_dataset}'", f"dataset_name = '{dataset_name}'")
        
        for old_dataset_m in hardcoded_datasets_m:
            content = content.replace(f"dataset_name_m ='{old_dataset_m}'", f"dataset_name_m ='{dataset_name_m}'")
            content = content.replace(f"dataset_name_m = '{old_dataset_m}'", f"dataset_name_m = '{dataset_name_m}'")
        
        # Also need to modify FloorPlanLoader.py temporarily for MARL training
        floorplan_loader_path = "data/FloorPlanLoader.py"
        with open(floorplan_loader_path, 'r') as f:
            loader_content = f.read()
        
        # Backup original and create modified version
        backup_path = "data/FloorPlanLoader_backup_marl.py"
        with open(backup_path, 'w') as f:
            f.write(loader_content)
        
        # Modify FloorPlanLoader - comprehensive replacement
        loader_content = loader_content.replace("dataset_name = 'residential'", f"dataset_name = '{dataset_name}'")
        loader_content = loader_content.replace("dataset_name_m ='Residential'", f"dataset_name_m ='{dataset_name_m}'")
        
        # Also fix the hardcoded path in the __init__ method
        if dataset_name == 'residential':
            loader_data_config = f'data/data_config_1/residential/{dataset_name_m}/'
        else:
            loader_data_config = f'data/data_config_1/tertiary/{dataset_name_m}/'
        
        loader_content = loader_content.replace(
            f"data_config=f'data/data_config_1/residential/{{dataset_name_m}}/'", 
            f"data_config=f'{loader_data_config}'"
        )
        loader_content = loader_content.replace(
            f"data_config=f'data/data_config_1/tertiary/{{dataset_name_m}}/'", 
            f"data_config=f'{loader_data_config}'"
        )
        
        # Handle all possible hardcoded dataset names in FloorPlanLoader
        for old_dataset in hardcoded_datasets:
            loader_content = loader_content.replace(f"dataset_name = '{old_dataset}'", f"dataset_name = '{dataset_name}'")
        for old_dataset_m in hardcoded_datasets_m:
            loader_content = loader_content.replace(f"dataset_name_m ='{old_dataset_m}'", f"dataset_name_m ='{dataset_name_m}'")
        
        with open(floorplan_loader_path, 'w') as f:
            f.write(loader_content)
        
        print(f"Temporarily modified FloorPlanLoader.py for MARL training: {dataset_name_m}")
        
        # Fix data paths - convert relative paths to absolute paths from current directory
        if dataset_name == 'residential':
            data_config_path = f'./data/data_config_1/residential/{dataset_name_m}/'
            data_root_path = f'./data/data_br/{dataset_name}/'
        else:
            data_config_path = f'./data/data_config_1/tertiary/{dataset_name_m}/'
            data_root_path = f'./data/data_br/{dataset_name}/'
        
        # Replace relative paths with absolute paths from current directory
        content = content.replace(
            f"../data/data_br/{dataset_name}/", 
            data_root_path
        )
        content = content.replace(
            f"../data/data_config_1/tertiary/{dataset_name_m}/", 
            data_config_path
        )
        content = content.replace(
            f"../data/data_config_1/residential/{dataset_name_m}/", 
            data_config_path
        )
        
        # Also handle generic path patterns and hardcoded dataset names
        content = content.replace(
            "../data/data_br/", 
            "./data/data_br/"
        )
        content = content.replace(
            "../data/data_config_1/", 
            "./data/data_config_1/"
        )
        
        # Fix checkpoint loading paths
        content = content.replace(
            "../checkpoint/", 
            "./checkpoint/"
        )
        content = content.replace(
            "../notebooks/", 
            "./notebooks/"
        )
        content = content.replace(
            "../results/", 
            "./results/"
        )
        
        # Create necessary directories for MARL training
        if not os.path.exists("./results/recon_img"):
            os.makedirs("./results/recon_img", exist_ok=True)
        if not os.path.exists(f"./results/recon_img/{dataset_name}"):
            os.makedirs(f"./results/recon_img/{dataset_name}", exist_ok=True)
        if not os.path.exists(f"./notebooks/{dataset_name}"):
            os.makedirs(f"./notebooks/{dataset_name}", exist_ok=True)
        print(f"Created output directories for MARL training: {dataset_name}")
        
        # Disable automatic image display/opening in MARL training
        content = content.replace("plt.show()", "# plt.show()  # Disabled for pipeline")
        content = content.replace("Image.open(", "# Image.open(")
        content = content.replace(".show()", "# .show()  # Disabled for pipeline")
        
        # Set matplotlib to non-interactive backend
        if "import matplotlib.pyplot as plt" in content:
            content = content.replace("import matplotlib.pyplot as plt", 
                                    "import matplotlib\nmatplotlib.use('Agg')  # Non-interactive backend\nimport matplotlib.pyplot as plt")
        elif "from matplotlib import pyplot as plt" in content:
            content = content.replace("from matplotlib import pyplot as plt", 
                                    "import matplotlib\nmatplotlib.use('Agg')  # Non-interactive backend\nfrom matplotlib import pyplot as plt")
        
        # Replace any hardcoded dataset names in the notebook (comprehensive replacement)
        hardcoded_datasets = ['cultural', 'residential', 'commercial', 'industrial', 'entertainment_venues', 'healthcare_and_charity', 'offices', 'leisure_and_hospitality', 'sports_facilities', 'warehouse_parking', 'urbanization_land', 'singular_building']
        hardcoded_datasets_m = ['Cultural', 'Residential', 'Commercial', 'Industrial', 'Entertainment_venues', 'Healthcare_and_charity', 'Offices', 'Leisure_and_hospitality', 'Sports_facilities', 'Warehouse_parking', 'Urbanization_land', 'Singular_building']
        
        for old_dataset in hardcoded_datasets:
            content = content.replace(
                f"dataset_name = '{old_dataset}'", 
                f"dataset_name = '{dataset_name}'"
            )
        
        for old_dataset_m in hardcoded_datasets_m:
            content = content.replace(
                f"dataset_name_m = '{old_dataset_m}'", 
                f"dataset_name_m = '{dataset_name_m}'"
            )
            content = content.replace(
                f"dataset_name_m ='{old_dataset_m}'", 
                f"dataset_name_m ='{dataset_name_m}'"
            )
        
        # Also fix hardcoded paths in FloorPlanDataset calls - be more comprehensive
        # Replace any f-string patterns with the correct path
        import re
        content = re.sub(
            r"data_config=f?['\"]\.\.?/data/data_config_1/[^/]+/\{?dataset_name_m\}?/?['\"]", 
            f"data_config='{data_config_path}'", 
            content
        )
        
        # Also handle specific patterns we know exist
        patterns_to_replace = [
            "data_config=f'../data/data_config_1/residential/{dataset_name_m}/'",
            "data_config=f'../data/data_config_1/tertiary/{dataset_name_m}/'",
            "data_config=f'./data/data_config_1/residential/{dataset_name_m}/'",
            "data_config=f'./data/data_config_1/tertiary/{dataset_name_m}/'",
            f"data_config=f'../data/data_config_1/residential/{dataset_name_m}/'",
            f"data_config=f'../data/data_config_1/tertiary/{dataset_name_m}/'",
            f"data_config=f'./data/data_config_1/residential/{dataset_name_m}/'",
            f"data_config=f'./data/data_config_1/tertiary/{dataset_name_m}/'",
        ]
        
        for pattern in patterns_to_replace:
            content = content.replace(pattern, f"data_config='{data_config_path}'")
        
        # CRITICAL FIX: Replace the hardcoded residential path with literal {dataset_name_m}
        content = content.replace(
            "data_config=f'../data/data_config_1/residential/{dataset_name_m}/'",
            f"data_config='{data_config_path}'"
        )
        
        # Additional fix: Replace the pattern that appears in the actual temp file
        content = content.replace(
            f"data_config=f'../data/data_config_1/residential/{dataset_name_m}/'",
            f"data_config='{data_config_path}'"
        )
        
        # CRITICAL FIX: Replace the exact pattern from the notebook
        # The notebook has: data_config=f'../data/data_config_1/residential/{dataset_name_m}/'
        content = content.replace(
            "data_config=f'../data/data_config_1/residential/{dataset_name_m}/'",
            f"data_config='{data_config_path}'"
        )
        
        # Also replace any hardcoded dataset_name_m values in paths
        for old_dataset_m in hardcoded_datasets_m:
            content = content.replace(
                f"data_config=f'../data/data_config_1/residential/{old_dataset_m}/'",
                f"data_config='{data_config_path}'"
            )
            content = content.replace(
                f"data_config=f'../data/data_config_1/tertiary/{old_dataset_m}/'",
                f"data_config='{data_config_path}'"
            )
            content = content.replace(
                f"data_config=f'./data/data_config_1/residential/{old_dataset_m}/'",
                f"data_config='{data_config_path}'"
            )
            content = content.replace(
                f"data_config=f'./data/data_config_1/tertiary/{old_dataset_m}/'",
                f"data_config='{data_config_path}'"
            )
        
        # Direct replacement for the specific FloorPlanDataset line we know exists
        old_floorplan_line = f"floor = FloorPlanDataset(multi_scale=True, root=f'../data/data_br/{{dataset_name}}/', data_config=f'../data/data_config_1/residential/{{dataset_name_m}}/', preprocess=True)"
        new_floorplan_line = f"floor = FloorPlanDataset(multi_scale=True, root='{data_root_path}', data_config='{data_config_path}', preprocess=True)"
        content = content.replace(old_floorplan_line, new_floorplan_line)
        
        # Also try without the f-string formatting
        old_floorplan_line2 = f"floor = FloorPlanDataset(multi_scale=True, root=f'../data/data_br/{dataset_name}/', data_config=f'../data/data_config_1/residential/{dataset_name_m}/', preprocess=True)"
        content = content.replace(old_floorplan_line2, new_floorplan_line)
        
        # Handle all possible combinations with hardcoded values
        for old_ds in hardcoded_datasets:
            for old_ds_m in hardcoded_datasets_m:
                old_line = f"floor = FloorPlanDataset(multi_scale=True, root=f'../data/data_br/{old_ds}/', data_config=f'../data/data_config_1/residential/{old_ds_m}/', preprocess=True)"
                content = content.replace(old_line, new_floorplan_line)
                old_line = f"floor = FloorPlanDataset(multi_scale=True, root=f'../data/data_br/{old_ds}/', data_config=f'../data/data_config_1/tertiary/{old_ds_m}/', preprocess=True)"
                content = content.replace(old_line, new_floorplan_line)
        
        print(f"Fixed paths: data_root={data_root_path}, data_config={data_config_path}")
        print(f"Expected metadata file: {data_config_path}meta_trsa_{dataset_name_m}.csv")
        
        # Debug: Show what dataset variables are in the content after replacement
        dataset_matches = re.findall(r"dataset_name = '[^']*'", content)
        dataset_m_matches = re.findall(r"dataset_name_m = ?'[^']*'", content)
        floorplan_matches = re.findall(r"FloorPlanDataset\([^)]*data_config=[^,)]*[,)]", content)
        print(f"Dataset variables after replacement: {dataset_matches}")
        print(f"Dataset_m variables after replacement: {dataset_m_matches}")
        print(f"FloorPlanDataset calls after replacement: {floorplan_matches}")
        
        # CRITICAL FIX: Replace the hardcoded checkpoint path with the actual best checkpoint
        # The notebook has: with open("../checkpoint/55-error-6.475618576863781.json") as json_file:
        # We need to replace this with the actual checkpoint that was just trained
        import re
        # Replace any pattern like "../checkpoint/XX-error-X.XXXXX.json"
        checkpoint_pattern = r'\.\./checkpoint/\d+-error-[\d.]+\.json'
        checkpoint_replacement = best_checkpoint_path.replace('./', '../')
        content = re.sub(checkpoint_pattern, checkpoint_replacement, content)
        
        # Also handle the case without the ../ prefix
        checkpoint_pattern2 = r'\./checkpoint/\d+-error-[\d.]+\.json'
        content = re.sub(checkpoint_pattern2, best_checkpoint_path, content)
        
        print(f"Replaced hardcoded checkpoint path with: {best_checkpoint_path}")
        
        # Write modified content
        with open(python_file, 'w') as f:
            f.write(content)
        
        print(f"Modified MARL script written to {python_file}")
        
        # Run the MARL training with real-time output
        print("Starting MARL training...")
        process = subprocess.Popen([sys.executable, python_file], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT,
                                 universal_newlines=True,
                                 cwd='.')
        
        # Print output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        return_code = process.poll()
        if return_code != 0:
            print(f"MARL training failed with return code {return_code}")
            return False
        
        print("MARL training completed successfully!")
        return True
        
    finally:
        # Restore original FloorPlanLoader.py
        backup_path = "data/FloorPlanLoader_backup_marl.py"
        floorplan_loader_path = "data/FloorPlanLoader.py"
        if os.path.exists(backup_path):
            with open(backup_path, 'r') as f:
                original_content = f.read()
            with open(floorplan_loader_path, 'w') as f:
                f.write(original_content)
            os.remove(backup_path)
            print("Restored original FloorPlanLoader.py after MARL training")
        
        # Clean up temporary files
        for temp_file in ["temp_train_marl.py", "temp_train_marl.ipynb"]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"Cleaned up {temp_file}")

def run_latent_clustering(dataset_name, dataset_name_m):
    """Run latent clustering analysis"""
    print("=" * 60)
    print("STEP 3: Running Latent Clustering")
    print("=" * 60)
    
    # Convert notebook to Python script and run it
    notebook_path = "notebooks/latent_clustering.ipynb"
    
    # Use nbconvert to convert notebook to Python
    convert_cmd = [
        "jupyter", "nbconvert", 
        "--to", "python", 
        "--output", "temp_latent_clustering",
        "--output-dir", ".",  # Output to current directory
        notebook_path
    ]
    
    try:
        print(f"Converting notebook {notebook_path} to Python...")
        result = subprocess.run(convert_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Failed to convert notebook: {result.stderr}")
            return False
        
        # Read the converted Python file
        python_file = "temp_latent_clustering.py"
        
        # Check if file exists
        if not os.path.exists(python_file):
            print(f"Error: Converted file {python_file} not found")
            return False
            
        with open(python_file, 'r') as f:
            content = f.read()
        
        # Modify dataset variables - comprehensive replacement
        print(f"Replacing dataset variables in latent clustering script: {dataset_name} -> {dataset_name_m}")
        
        # Define all possible hardcoded dataset names
        hardcoded_datasets = ['cultural', 'residential', 'commercial', 'industrial', 'entertainment_venues', 'healthcare_and_charity', 'offices', 'leisure_and_hospitality', 'sports_facilities', 'warehouse_parking', 'urbanization_land', 'singular_building']
        hardcoded_datasets_m = ['Cultural', 'Residential', 'Commercial', 'Industrial', 'Entertainment_venues', 'Healthcare_and_charity', 'Offices', 'Leisure_and_hospitality', 'Sports_facilities', 'Warehouse_parking', 'Urbanization_land', 'Singular_building']
        
        # Replace all possible hardcoded dataset names
        for old_dataset in hardcoded_datasets:
            content = content.replace(f"dataset_name = '{old_dataset}'", f"dataset_name = '{dataset_name}'")
        
        for old_dataset_m in hardcoded_datasets_m:
            content = content.replace(f"dataset_name_m ='{old_dataset_m}'", f"dataset_name_m ='{dataset_name_m}'")
            content = content.replace(f"dataset_name_m = '{old_dataset_m}'", f"dataset_name_m = '{dataset_name_m}'")
            content = content.replace(f"curr_data = 'BR_{old_dataset_m}'", f"curr_data = 'BR_{dataset_name_m}'")
        
        # Fix any remaining hardcoded dataset references in paths and filenames
        for old_dataset in hardcoded_datasets:
            content = content.replace(f"/{old_dataset}/", f"/{dataset_name}/")
            content = content.replace(f"_{old_dataset}_", f"_{dataset_name}_")
            content = content.replace(f"BR_{old_dataset.capitalize()}", f"BR_{dataset_name_m}")
            content = content.replace(f"'{old_dataset}/'", f"'{dataset_name}/'")
            content = content.replace(f'"{old_dataset}/"', f'"{dataset_name}/"')
        
        for old_dataset_m in hardcoded_datasets_m:
            content = content.replace(f"BR_{old_dataset_m}", f"BR_{dataset_name_m}")
            content = content.replace(f"/{old_dataset_m}/", f"/{dataset_name_m}/")
            content = content.replace(f"'{old_dataset_m}/'", f"'{dataset_name_m}/'")
            content = content.replace(f'"{old_dataset_m}/"', f'"{dataset_name_m}/"')
        
        # Fix specific patterns that might be causing issues
        content = content.replace("new_dir = dataset_name", f"new_dir = '{dataset_name}'")
        content = content.replace("new_dir = 'commercial'", f"new_dir = '{dataset_name}'")
        content = content.replace("new_dir = 'residential'", f"new_dir = '{dataset_name}'")
        content = content.replace("new_dir = 'cultural'", f"new_dir = '{dataset_name}'")
        
        # Fix data config paths
        if dataset_name == 'residential':
            data_config_path = f'./data/data_config_1/residential/{dataset_name_m}/'
        else:
            data_config_path = f'./data/data_config_1/tertiary/{dataset_name_m}/'
        
        content = content.replace(
            "./data/data_config_1/residential/", 
            f"./data/data_config_1/{'residential' if dataset_name == 'residential' else 'tertiary'}/"
        )
        content = content.replace(
            "data/data_config_1/tertiary/", 
            f"data/data_config_1/{'residential' if dataset_name == 'residential' else 'tertiary'}/"
        )
        
        # Fix specific problematic patterns
        content = content.replace(
            f"./data/data_config_1/{dataset_name}/", 
            data_config_path
        )
        content = content.replace(
            f"data/data_config_1/{dataset_name}/", 
            data_config_path
        )
        
        # Fix any remaining hardcoded dataset_name_m in paths
        for old_dataset_m in hardcoded_datasets_m:
            content = content.replace(
                f"./data/data_config_1/{'residential' if dataset_name == 'residential' else 'tertiary'}/{old_dataset_m}/",
                data_config_path
            )
            content = content.replace(
                f"data/data_config_1/{'residential' if dataset_name == 'residential' else 'tertiary'}/{old_dataset_m}/",
                data_config_path
            )
        
        # Fix the specific pattern that creates duplicate Cultural/Cultural
        content = content.replace(
            f"data_config=f'./data/data_config_1/tertiary/Cultural/{{dataset_name_m}}/'",
            f"data_config=f'{data_config_path}'"
        )
        content = content.replace(
            f"data_config=f'./data/data_config_1/residential/Residential/{{dataset_name_m}}/'",
            f"data_config=f'{data_config_path}'"
        )
        
        # Also fix the multi-line pattern with line continuation
        content = content.replace(
            f"data_config=f'./data/data_config_1/tertiary/Cultural/{{dataset_name_m}}/', preprocess=True)",
            f"data_config=f'{data_config_path}', preprocess=True)"
        )
        content = content.replace(
            f"data_config=f'./data/data_config_1/residential/Residential/{{dataset_name_m}}/', preprocess=True)",
            f"data_config=f'{data_config_path}', preprocess=True)"
        )
        
        # Fix relative paths in latent clustering - be more comprehensive
        content = content.replace("../data/", "./data/")
        content = content.replace("../checkpoint/", "./checkpoint/")
        content = content.replace("../notebooks/", "./notebooks/")
        content = content.replace("../results/", "./results/")
        content = content.replace("../data_pipeline/", "./data_pipeline/")
        
        # Fix double path separators and other path issues
        content = content.replace(".././", "./")
        content = content.replace("././", "./")
        content = content.replace("//", "/")
        
        # Fix any remaining hardcoded dataset combinations in paths
        for old_dataset in hardcoded_datasets:
            for old_dataset_m in hardcoded_datasets_m:
                # Fix patterns like /tertiary/Cultural/Commercial/ -> /tertiary/Cultural/
                content = content.replace(f"/{old_dataset_m}/{old_dataset_m}/", f"/{old_dataset_m}/")
                content = content.replace(f"/{old_dataset}/{old_dataset_m}/", f"/{old_dataset}/")
                content = content.replace(f"/{old_dataset_m}/{old_dataset}/", f"/{old_dataset_m}/")
                
                # Fix filename patterns
                content = content.replace(f"meta_trsa_{old_dataset_m}.csv", f"meta_trsa_{dataset_name_m}.csv")
                content = content.replace(f"height_trsa_{old_dataset_m}.csv", f"height_trsa_{dataset_name_m}.csv")
        
        # Handle specific geojson file path
        content = content.replace(
            "../data_pipeline/trusted_zone/preprocessed_data/cleaned_catastro.geojson",
            "./data_pipeline/trusted_zone/preprocessed_data/cleaned_catastro.geojson"
        )
        
        # Create necessary directories
        if not os.path.exists("./results/recon_img"):
            os.makedirs("./results/recon_img", exist_ok=True)
        if not os.path.exists(f"./results/recon_img/{dataset_name}"):
            os.makedirs(f"./results/recon_img/{dataset_name}", exist_ok=True)
        if not os.path.exists(f"./notebooks/{dataset_name}"):
            os.makedirs(f"./notebooks/{dataset_name}", exist_ok=True)
        print(f"Created output directories for {dataset_name}")
        
        # Also need to modify FloorPlanLoader.py temporarily for latent clustering
        floorplan_loader_path = "data/FloorPlanLoader.py"
        with open(floorplan_loader_path, 'r') as f:
            loader_content = f.read()
        
        # Backup original and create modified version
        backup_path = "data/FloorPlanLoader_backup_clustering.py"
        with open(backup_path, 'w') as f:
            f.write(loader_content)
        
        # Modify FloorPlanLoader - comprehensive replacement
        for old_dataset in hardcoded_datasets:
            loader_content = loader_content.replace(f"dataset_name = '{old_dataset}'", f"dataset_name = '{dataset_name}'")
        
        for old_dataset_m in hardcoded_datasets_m:
            loader_content = loader_content.replace(f"dataset_name_m ='{old_dataset_m}'", f"dataset_name_m ='{dataset_name_m}'")
            loader_content = loader_content.replace(f"dataset_name_m = '{old_dataset_m}'", f"dataset_name_m = '{dataset_name_m}'")
        
        # Also replace any hardcoded values in f-strings for meta and height files
        for old_dataset_m in hardcoded_datasets_m:
            loader_content = loader_content.replace(f"f'meta_trsa_{old_dataset_m}.csv'", f"f'meta_trsa_{dataset_name_m}.csv'")
            loader_content = loader_content.replace(f"f'height_trsa_{old_dataset_m}.csv'", f"f'height_trsa_{dataset_name_m}.csv'")
        
        # Also fix the hardcoded path in the __init__ method
        if dataset_name == 'residential':
            loader_data_config = f'data/data_config_1/residential/{dataset_name_m}/'
        else:
            loader_data_config = f'data/data_config_1/tertiary/{dataset_name_m}/'
        
        # Fix the data_config path more precisely to avoid duplicates
        loader_content = loader_content.replace(
            f"data_config=f'data/data_config_1/residential/{{dataset_name_m}}/'", 
            f"data_config=f'{loader_data_config}'"
        )
        loader_content = loader_content.replace(
            f"data_config=f'data/data_config_1/tertiary/{{dataset_name_m}}/'", 
            f"data_config=f'{loader_data_config}'"
        )
        
        # Also fix the hardcoded default value in the __init__ method signature
        loader_content = loader_content.replace(
            f"data_config=f'data/data_config_1/residential/{{dataset_name_m}}/',", 
            f"data_config=f'{loader_data_config}',"
        )
        
        # Also fix any existing hardcoded paths that might cause duplicates
        for old_dataset_m in hardcoded_datasets_m:
            loader_content = loader_content.replace(
                f"data/data_config_1/residential/{old_dataset_m}/", 
                loader_data_config
            )
            loader_content = loader_content.replace(
                f"data/data_config_1/tertiary/{old_dataset_m}/", 
                loader_data_config
            )
            loader_content = loader_content.replace(
                f"data/data_config_1/{dataset_name}/{old_dataset_m}/", 
                loader_data_config
            )
        
        with open(floorplan_loader_path, 'w') as f:
            f.write(loader_content)
        
        # Debug: Check what the data_config path looks like after modification
        debug_lines = loader_content.split('\n')
        for i, line in enumerate(debug_lines):
            if 'data_config=' in line:
                print(f"DEBUG FloorPlanLoader line {i+1}: {line.strip()}")
            if 'dataset_name_m' in line and '=' in line:
                print(f"DEBUG FloorPlanLoader line {i+1}: {line.strip()}")
        
        print(f"Temporarily modified FloorPlanLoader.py for latent clustering: {dataset_name_m}")
        
        # Disable automatic image display/opening
        content = content.replace("plt.show()", "# plt.show()  # Disabled for pipeline")
        content = content.replace("Image.open(", "# Image.open(")
        content = content.replace(".show()", "# .show()  # Disabled for pipeline")
        
        # Set matplotlib to non-interactive backend
        if "import matplotlib.pyplot as plt" in content:
            content = content.replace("import matplotlib.pyplot as plt", 
                                    "import matplotlib\nmatplotlib.use('Agg')  # Non-interactive backend\nimport matplotlib.pyplot as plt")
        elif "from matplotlib import pyplot as plt" in content:
            content = content.replace("from matplotlib import pyplot as plt", 
                                    "import matplotlib\nmatplotlib.use('Agg')  # Non-interactive backend\nfrom matplotlib import pyplot as plt")
        
        # Debug: Show what dataset variables are in the content after replacement
        import re
        dataset_matches = re.findall(r"dataset_name = '[^']*'", content)
        dataset_m_matches = re.findall(r"dataset_name_m = ?'[^']*'", content)
        curr_data_matches = re.findall(r"curr_data = '[^']*'", content)
        print(f"Latent clustering dataset variables after replacement: {dataset_matches}")
        print(f"Latent clustering dataset_m variables after replacement: {dataset_m_matches}")
        print(f"Latent clustering curr_data variables after replacement: {curr_data_matches}")
        
        # Write modified content
        with open(python_file, 'w') as f:
            f.write(content)
        
        # Direct fix for the duplicate dataset/dataset pattern in the temp file
        with open(python_file, 'r') as f:
            temp_content = f.read()
        
        # Replace any pattern that has {dataset_name_m}/{dataset_name_m}/ with just {dataset_name_m}/
        import re
        # Generic pattern to catch any duplicate dataset name in paths
        pattern = rf'\./data/data_config_1/tertiary/{dataset_name_m}/\{{dataset_name_m\}}/'
        replacement = f'{data_config_path}'
        temp_content = re.sub(pattern, replacement, temp_content)
        
        # Also fix specific data_config patterns
        temp_content = temp_content.replace(
            f"data_config=f'./data/data_config_1/tertiary/{dataset_name_m}/{{dataset_name_m}}/'",
            f"data_config=f'{data_config_path}'"
        )
        temp_content = temp_content.replace(
            f"data_config=f'./data/data_config_1/tertiary/{dataset_name_m}/{{dataset_name_m}}/', preprocess=True)",
            f"data_config=f'{data_config_path}', preprocess=True)"
        )
        
        # Fix slice indices in get_zoomed_img function
        temp_content = temp_content.replace(
            "x1 = center_x - half_pixel\n    x2 = center_x + half_pixel\n    y1 = center_y - half_pixel\n    y2 = center_y + half_pixel",
            "x1 = int(center_x - half_pixel)\n    x2 = int(center_x + half_pixel)\n    y1 = int(center_y - half_pixel)\n    y2 = int(center_y + half_pixel)"
        )
        
        # Fix hardcoded metadata paths that create duplicate dataset names
        # Replace any pattern that has {dataset_name_m}/{dataset_name_m}/ with just {dataset_name_m}/
        metadata_pattern = rf'\./data/data_config_1/tertiary/{dataset_name_m}/\{{dataset_name_m\}}/'
        temp_content = re.sub(metadata_pattern, f'{data_config_path}', temp_content)
        
        # Also fix the specific metadata lines we know about
        temp_content = temp_content.replace(
            f'f"./data/data_config_1/tertiary/{dataset_name_m}/{{dataset_name_m}}/meta_trsa_{dataset_name_m}.csv"',
            f'f"{data_config_path}meta_trsa_{dataset_name_m}.csv"'
        )
        temp_content = temp_content.replace(
            f"f'./data/data_config_1/tertiary/{dataset_name_m}/{{dataset_name_m}}/meta_trsa_{dataset_name_m}.csv'",
            f"f'{data_config_path}meta_trsa_{dataset_name_m}.csv'"
        )
        
        # Write the fixed content back
        with open(python_file, 'w') as f:
            f.write(temp_content)
        
        print(f"Direct fix applied to {python_file}")
        print(f"Slice indices fix applied to {python_file}")
        
        # Run the clustering analysis with real-time output
        print("Starting latent clustering...")
        process = subprocess.Popen([sys.executable, python_file], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT,
                                 universal_newlines=True,
                                 cwd='.')
        
        # Print output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        return_code = process.poll()
        if return_code != 0:
            print(f"Latent clustering failed with return code {return_code}")
            return False
        
        print("Latent clustering completed successfully!")
        return True
        
    finally:
        # Restore original FloorPlanLoader.py
        backup_path = "data/FloorPlanLoader_backup_clustering.py"
        floorplan_loader_path = "data/FloorPlanLoader.py"
        if os.path.exists(backup_path):
            with open(backup_path, 'r') as f:
                original_content = f.read()
            with open(floorplan_loader_path, 'w') as f:
                f.write(original_content)
            os.remove(backup_path)
            print("Restored original FloorPlanLoader.py after latent clustering")
        
        # Clean up temporary files (keep for debugging)
        for temp_file in ["temp_latent_clustering.ipynb"]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"Cleaned up {temp_file}")
        
        if os.path.exists("temp_latent_clustering.py"):
            print("Temporary file kept for inspection: temp_latent_clustering.py")

def main():
    parser = argparse.ArgumentParser(description='Run complete MARL Building Energy Estimation pipeline')
    parser.add_argument('--dataset', type=str, default='residential', 
                       help='Dataset name (default: residential)')
    parser.add_argument('--dataset-m', type=str, default='Residential',
                       help='Dataset name for metadata (default: Residential)')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoint',
                       help='Directory containing checkpoints (default: ./checkpoint)')
    parser.add_argument('--skip-vqae', action='store_true',
                       help='Skip VQAE training step')
    parser.add_argument('--skip-marl', action='store_true',
                       help='Skip MARL training step')
    parser.add_argument('--skip-clustering', action='store_true',
                       help='Skip latent clustering step')
    
    args = parser.parse_args()
    
    # Set dataset variables
    dataset_name = args.dataset
    dataset_name_m = args.dataset_m
    curr_data = f"BR_{dataset_name_m}"
    
    print("=" * 80)
    print("MARL BUILDING ENERGY ESTIMATION PIPELINE")
    print("=" * 80)
    print(f"Dataset: {dataset_name}")
    print(f"Dataset (metadata): {dataset_name_m}")
    print(f"Current data: {curr_data}")
    print("=" * 80)
    
    try:
        # Step 1: VQAE Training
        latest_checkpoint = None
        if not args.skip_vqae:
            result = run_vqae_training(dataset_name, dataset_name_m)
            if isinstance(result, tuple):
                success, latest_checkpoint = result
            else:
                success = result
                latest_checkpoint = None
            
            if not success:
                print("Pipeline failed at VQAE training step")
                return 1
        else:
            print("Skipping VQAE training step")
        
        # Step 2: Use the checkpoint from VQAE training or find best checkpoint
        if not args.skip_marl:
            try:
                if latest_checkpoint:
                    best_checkpoint = latest_checkpoint
                    # Parse error from filename
                    import re
                    match = re.search(r'error-([0-9.]+)\.json', latest_checkpoint)
                    best_error = float(match.group(1)) if match else 0.0
                    print(f"Using checkpoint from VQAE training: {best_checkpoint} (error: {best_error})")
                else:
                    best_checkpoint, best_error = find_best_checkpoint(args.checkpoint_dir)
                    print(f"Using best checkpoint: {best_checkpoint} (error: {best_error})")
                
                success = run_marl_training(dataset_name, dataset_name_m, best_checkpoint)
                if not success:
                    print("Pipeline failed at MARL training step")
                    return 1
            except (FileNotFoundError, ValueError) as e:
                print(f"Error finding checkpoint: {e}")
                return 1
        else:
            print("Skipping MARL training step")
        
        # Step 3: Latent Clustering
        if not args.skip_clustering:
            success = run_latent_clustering(dataset_name, dataset_name_m)
            if not success:
                print("Pipeline failed at latent clustering step")
                return 1
        else:
            print("Skipping latent clustering step")
        
        print("=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Results saved in:")
        print(f"  - Checkpoints: {args.checkpoint_dir}")
        print(f"  - Clustering results: ./results/recon_img/{dataset_name}/")
        print(f"  - Latent representations: ./data/data_root/marl_latent_{curr_data}.pt")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
