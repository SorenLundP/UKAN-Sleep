from csdp_training.experiments.bids_predictor import BIDS_USleep_Predictor

def main():
    # Define paths to the pretrained weight files
    checkpoint_singlech = "usleep_checkpoints/lightning/single_channel/m1-m2_finetuned.ckpt"
    checkpoint_twoch = "usleep_checkpoints/lightning/two_channel/default_big_sleep.ckpt"

    # Path to dummy data to showcase this demo:
    dataDir='O:/Tech_NTLab/DataSets/testData/sleep_data_set/testingData/derivatives/cleaned_1/'
    dataExtension = ".set"
    tasks = "sleep"
    subjects = ["002"]
    sessions = ["001"]

    predictor = BIDS_USleep_Predictor(data_dir=dataDir, data_extension=dataExtension, data_task=tasks, subjects=subjects, sessions=sessions)

    # Peek the data and obtain avaiable channels
    ch_names = predictor.get_available_channels()
    print(ch_names)

    # Choose which channels to use for building the dataset object
    dataset = predictor.build_dataset(["F3", "C3", "EOGl", "EOGr"])
    
    # Predict on all the data. We tell U-Sleep which of the above channel indexes are EEG and EOG
    two_channel_predictions = predictor.predict_all(checkpoint_twoch, dataset, eeg_indexes=[0,1], eog_indexes=[2,3])
    print(two_channel_predictions)

    # Do the same, but for a single-channel model (only EEG)
    dataset = predictor.build_dataset(["F3", "C3"])
    # No need to specify EEG and EOG
    single_channel_predictions = predictor.predict_all(checkpoint_singlech, dataset)
    print(single_channel_predictions)

if __name__=="__main__":
    main()