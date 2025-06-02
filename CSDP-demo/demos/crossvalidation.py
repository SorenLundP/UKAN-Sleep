from csdp_training.experiments.cv import CV_Experiment
from csdp_training.lightning_models.usleep import USleep_Lightning
from csdp_pipeline.pipeline_elements.pipeline import PipelineConfiguration
import os

def main():
    # Input where to log
    logging_folder = f"{os.getcwd()}/crossvalidation_logs"

    # Path to the HDF5 files. It must only contain HDF5 files from ERDA and nothing else.
    data_path = "O:/Tech_NTLab/DataSets/testData/sleep_data_set/csdp_demo/hdf5"

    # Names of the HDF5 files in the data directory to use
    datasets = ["homepap.hdf5"]

    batch_size = 64
    learning_rate = 0.0001
    max_training_epohcs = 1
    num_folds = 3
    early_stopping_patience = 1
    batches_per_training_epoch = 2
    num_validation_subjects = 1

    pipeline_configuration = PipelineConfiguration()

    # Load an existing pre-trained model here. You can also load a clean model without specifying a checkpoint.
    net = USleep_Lightning.load_from_checkpoint("usleep_checkpoints/lightning/two_channel/alternative_big_sleep.ckpt")

    cv = CV_Experiment(base_net=net,
                       base_data_path=data_path,
                       datasets=datasets,
                       training_epochs=max_training_epohcs,
                       batch_size=batch_size,
                       num_folds=num_folds,
                       earlystopping_patience=early_stopping_patience,
                       batches_per_epoch=batches_per_training_epoch,
                       logging_folder=logging_folder,
                       num_validation_subjects=num_validation_subjects,
                       pipeline_configuration=pipeline_configuration,
                       split_filepath=None)

    cv.run()

if __name__=="__main__":
    main()