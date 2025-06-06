from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from csdp_pipeline.pipeline_elements.pipeline import PipelineDataset, PipelineConfiguration
from csdp_pipeline.pipeline_elements.samplers import Random_Sampler, SamplerConfiguration, Determ_sampler

class IDataloader_Factory(ABC):
    @abstractmethod
    def training_loader(self, num_workers):
        pass

    @abstractmethod
    def validation_loader(self, num_workers):
        pass

    @abstractmethod
    def testing_loader(self, num_workers):
        pass

class Dataloader_Factory(IDataloader_Factory):

    def __init__(
        self,
        training_batch_size: int,
        samplers: SamplerConfiguration,
        preprocessing_pipes: PipelineConfiguration = PipelineConfiguration(),
    ):
        """_summary_
        Args:
            gradient_steps (int): How many gradient steps per epoch. Gradient_steps * batch_size determines how many samples are drawn each epoch
            batch_size (int): The batch_size. Gradient_steps * batch_size determines how many samples are drawn each epoch
            hdf5_base_path (str): Absolute path to the root containing HDF5 datasets
            trainsets (list[str]): List of names of datasets used for training. For example, if the dataset "abc.hdf5" is in the hdf5_base_path, add "abc" to the list
            valsets (list[str]): List of names of datasets used for validation. For example, if the dataset "abc.hdf5" is in the hdf5_base_path, add "abc" to the list
            testsets (list[str]): List of names of datasets used for testing. For example, if the dataset "abc.hdf5" is in the hdf5_base_path, add "abc" to the list
            data_split_path (str, optional): Specifies a path to a split json file. For examples, look in the folder "csdp_pipeline/splits" of this repository. Defaults to None, which means all subjects are used for both training, validation and test.
            create_random_split (bool, optional): If set to True, a random split json file will be created and used for data-loading. Defaults to False.
        """

        self.pipe_configuration = preprocessing_pipes
        self.samplers = samplers
        self.training_batch_size = training_batch_size

    def training_loader(self,
                        num_workers=1) -> DataLoader:
        """_summary_

        Args:
            num_workers (int, optional): Number of workers in the PyTorch dataloader. Defaults to 1.

        Returns:
            DataLoader: The training dataloader. When drawing samples from this dataloader, the data will be served with 4 values - (eeg_data, eog_data, labels, tags).
        """

        dataset = PipelineDataset(self.samplers.get_sampler_by_stage("train"),
                                  self.pipe_configuration.get_pipe_by_stage("train"))

        trainloader = DataLoader(
            dataset,
            batch_size=self.training_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        return trainloader

    def validation_loader(self,
                          num_workers=1) -> DataLoader:
        """_summary_

        Args:
            num_workers (int, optional): Number of workers in the PyTorch dataloader. Defaults to 1.

        Returns:
            DataLoader: The validation dataloader. When drawing samples from this dataloader, the data will be served with 4 values - (eeg_data, eog_data, labels, tags).
        """

        sampler = self.samplers.get_sampler_by_stage("val")

        dataset = PipelineDataset(sampler,
                                  self.pipe_configuration.get_pipe_by_stage("val"))
        
        valloader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=num_workers,
        )
        return valloader

    def testing_loader(self,
                       num_workers=1) -> DataLoader:
        """_summary_

        Args:
            num_workers (int, optional): Number of workers in the PyTorch dataloader. Defaults to 1.

        Returns:
            DataLoader: The testing dataloader. When drawing samples from this dataloader, the data will be served with 4 values - (eeg_data, eog_data, labels, tags).
        """

        sampler = self.samplers.get_sampler_by_stage("test")

        dataset = PipelineDataset(sampler,
                                  self.pipe_configuration.get_pipe_by_stage("test"))

        testloader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=num_workers,
        )

        return testloader

# class DefaultUSleepDataloader(IDataloader_Factory):
#     def __init__(self,
#                  gradient_steps: int,
#                 batch_size: int,
#                 hdf5_base_path: str,
#                 trainsets: list[str],
#                 valsets: list[str],
#                 testsets: list[str],
#                 data_split_path: str,
#                 sub_percentage = 1.0):
        
#         train_sampler = Random_Sampler(
#                 hdf5_base_path,
#                 trainsets,
#                 split_type="train",
#                 num_epochs=35,
#                 split_file_path=data_split_path,
#                 subject_percentage=sub_percentage,
#         )

#         val_sampler = Determ_sampler(
#                 hdf5_base_path,
#                 valsets,
#                 split_type="val",
#                 split_file=data_split_path,
#                 get_all_channels= False,
#             )
        
#         test_sampler = Determ_sampler(
#                 hdf5_base_path,
#                 testsets,
#                 split_type="test",
#                 split_file=data_split_path,
#                 get_all_channels= True,
#             )

#         samplers = SamplerConfiguration(train_sampler,
#                                         val_sampler,
#                                         test_sampler)

#         pipelines = PipelineConfiguration()

#         self.fac = Dataloader_Factory(gradient_steps,
#                                       batch_size,
#                                       samplers,
#                                       pipelines)

#     def training_loader(self, num_workers):
#         return self.fac.training_loader(num_workers)
    
#     def validation_loader(self, num_workers):
#         return self.fac.validation_loader(num_workers)
    
#     def testing_loader(self, num_workers):
#         return self.fac.testing_loader(num_workers)



# class LSeqSleepNet_Dataloader_Factory(IDataloader_Factory):
#     def __init__(
#         self,
#         gradient_steps: int,
#         batch_size: int,
#         num_epochs: int,
#         hdf5_base_path: str,
#         trainsets: list[str],
#         valsets: list[str],
#         testsets: list[str],
#         data_split_path: str = None,
#         create_random_split: bool = False,
#     ):
#         """_summary_
#         Args:
#             gradient_steps (int): How many gradient steps per epoch. Gradient_steps * batch_size determines how many samples are drawn each epoch
#             batch_size (int): The batch_size. Gradient_steps * batch_size determines how many samples are drawn each epoch
#             num_epochs (int): Number of sleep epochs to draw per sample. L-SeqSleepNet default is 200, but SeqSleepNet could use less.
#             hdf5_base_path (str): Absolute path to the root containing HDF5 datasets
#             trainsets (list[str]): List of names of datasets used for training. For example, if the dataset "abc.hdf5" is in the hdf5_base_path, add "abc" to the list
#             valsets (list[str]): List of names of datasets used for validation. For example, if the dataset "abc.hdf5" is in the hdf5_base_path, add "abc" to the list
#             testsets (list[str]): List of names of datasets used for testing. For example, if the dataset "abc.hdf5" is in the hdf5_base_path, add "abc" to the list
#             data_split_path (str, optional): Specifies a path to a split json file. For examples, look in the folder "csdp_pipeline/splits" of this repository. Defaults to None, which means all subjects are used for both training, validation and test.
#             create_random_split (bool, optional): If set to True, a random split json file will be created and used for data-loading. Defaults to False.
#         """
#         super().__init__(data_split_path, hdf5_base_path, create_random_split)

#         self.gradient_steps = gradient_steps
#         self.batch_size = batch_size
#         self.num_epochs = num_epochs

#         self.factory = LSeqSleepNet_Pipeline_Factory(
#             hdf5_base_path=hdf5_base_path,
#             split_path=self.data_split_path,
#             trainsets=trainsets,
#             valsets=valsets,
#             testsets=testsets,
#             num_epochs=num_epochs,
#         )

#     def create_training_loader(self, num_workers=1):
#         pipes = self.factory.create_training_pipeline()
#         dataset = PipelineDataset(pipes, self.gradient_steps * self.batch_size)
#         return DataLoader(
#             dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=num_workers,
#             pin_memory=True,
#         )

#     def create_validation_loader(self, num_workers=1):
#         pipes = self.factory.create_validation_pipeline()
#         dataset = PipelineDataset(pipes, len(pipes[0].records))
#         return DataLoader(
#             dataset, batch_size=1, shuffle=False, num_workers=num_workers
#         )

#     def create_testing_loader(self, num_workers=1):
#         pipes = self.factory.create_test_pipeline()
#         dataset = PipelineDataset(pipes, len(pipes[0].records))
#         return DataLoader(
#             dataset, batch_size=1, shuffle=False, num_workers=num_workers
#         )
