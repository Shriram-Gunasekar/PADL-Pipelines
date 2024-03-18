transformations = Compose([
    Transformations.Rescale(256),
    Transformations.RandomCrop(224),
    Transformations.ToTensor(),
    Transformations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
