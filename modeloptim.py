segmentation_model = SegmentationModel(num_classes)
discriminator_model = DiscriminatorModel()

segmentation_optimizer = optim.Adam(segmentation_model.parameters(), lr=learning_rate)
discriminator_optimizer = optim.Adam(discriminator_model.parameters(), lr=learning_rate)
