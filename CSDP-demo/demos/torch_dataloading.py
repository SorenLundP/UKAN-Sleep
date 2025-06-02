from csdp_pipeline.factories.dataloader_factory import Dataloader_Factory
from csdp_pipeline.pipeline_elements.models import Split
from csdp_pipeline.pipeline_elements.samplers import Random_Sampler, Determ_sampler, SamplerConfiguration
from ml_architectures.usleep.usleep import USleep
import torch
import os

def main():
    # Path to your current directory
    batch_size = 64

hdf5_file_path = './data/demo.hdf5'  # Path to the HDF5 file
    base_hdf5_path = os.path.dirname(hdf5_file_path) # Get the directory containing the HDF5 file
    split_file_path = os.getcwd() # Where to save the generated dataloading split file
    sleep_epochs_per_sample = 35 # Number of sleep epochs per sample
    num_batches = 10 # Number of batches per training epoch
    training_iterations = batch_size*num_batches # The resulting number of sampling iterations in the data
    dataloader_workers = 1 # Number of dataloader workers. Set to number of CPU cores available.

    # Create a random subject-based split in the data. Can be dumped to a file, which can later be used with Split.File(....)
    split = Split.random(base_hdf5_path=base_hdf5_path,
                        split_name="demo",
                        split_percentages=(0.4,0.3,0.3))
    split.dump_file(path=split_file_path)

    # Reloading the file, not necessary for the same run though.
    split = Split.file(f"{split_file_path}/demo.json")

    # Define the samplers and build the dataloaders
    train_sampler = Random_Sampler(split,
                                   num_epochs=sleep_epochs_per_sample,
                                   num_iterations=training_iterations)

    val_sampler = Determ_sampler(split,
                                split_type="val")

    test_sampler = Determ_sampler(split,
                                split_type="test")

    samplers = SamplerConfiguration(train_sampler,
                                    val_sampler,
                                    test_sampler)

    factory = Dataloader_Factory(training_batch_size=batch_size,
                                    samplers=samplers)

    train_loader = factory.training_loader(num_workers=dataloader_workers)
    val_loader = factory.validation_loader(num_workers=dataloader_workers)
    test_loader = factory.testing_loader(num_workers=dataloader_workers)

    # Load the data
    iterator = iter(train_loader)
    batch = next(iterator)

    eeg_signals = batch["eeg"]
    eog_signals = batch["eog"]

    labels = batch["labels"]
    meta_data = batch["tag"]

    # Feed the sampled data through U-Sleep
    batched_data = torch.cat([eeg_signals, eog_signals], dim=1)

    usleep_instance = USleep()

    with torch.no_grad():
        output = usleep_instance(batched_data.float())
        class_predictions = torch.argmax(output, dim=1)
        print(class_predictions)

if __name__=="__main__":
    main()