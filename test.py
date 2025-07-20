from data.FloorPlanLoader import FloorPlanDataset

dataset = FloorPlanDataset(multi_scale=True, root='data/data_root_1/data02/', data_config='data/data_config/', preprocess=False)

print("Dataset length:", len(dataset))
sample = dataset[0]
print("Sample keys:", sample.keys())
for key in sample:
    print(f"{key}: {type(sample[key])}, shape: {getattr(sample[key], 'shape', 'N/A')}")
