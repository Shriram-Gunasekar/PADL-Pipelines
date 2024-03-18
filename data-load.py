source_dataset = SourceDataset(source_data_path, transform=transformations)
target_dataset = TargetDataset(target_data_path, transform=transformations)

source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)
