from data.FloorPlanLoader import FloorPlanDataset

dataset = FloorPlanDataset(multi_scale=True, root='data/data_root_1/data02/', data_config='data/data_config/', preprocess=False)

print(len(dataset))
sample = dataset[0]
print(sample['image_tensor'].shape)