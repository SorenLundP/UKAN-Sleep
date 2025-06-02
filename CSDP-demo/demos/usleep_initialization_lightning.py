from csdp_training.lightning_models.usleep import USleep_Lightning
import torch

def main():
    
    # Load a clean lightning model
    net: USleep_Lightning = USleep_Lightning(lr=0.0001, 
                                             batch_size=64,
                                             complexity_factor=1.67, 
                                             depth=12,
                                             include_eog=True)
    
    # Calculate and print total number of parameters
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Since we include EOG, it is a two channel model. We define EEG and EOG dummy data here
    # Shapes are (batchsize, num_channels, num_samples). In other words, 2 EEGs and 2 EOGS, and 20 sleep epochs.
    eeg_channels = torch.rand((1, 2, 30*128*20))
    eog_channels = torch.rand((1, 2, 30*128*20))

    # Feed the data through U-Sleep. The network will perform predictions on all possible combinations of one EEG and one EOG.
    # The resulting output is therefore a dictionary with 4 different keys.
    output = net.majority_vote_prediction(eeg_channels, eog_channels)

    # One possibility is now to sum the votes of each prediction:
    votes = torch.zeros(1, 5, 20)

    for item in output.items():
        pred = item[1]
        votes = torch.add(votes, pred)

    sleep_stages = torch.argmax(votes, axis=1)
    print(sleep_stages)

    # We can also load a pre-trained model:
    net: USleep_Lightning = USleep_Lightning.load_from_checkpoint("usleep_checkpoints/lightning/two_channel/alternative_big_sleep.ckpt")

    # Calculate and print total number of parameters for the loaded model
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Loaded model total parameters: {total_params:,}")
    print(f"Loaded model trainable parameters: {trainable_params:,}")

    # ... And repeat the process
    output = net.majority_vote_prediction(eeg_channels, eog_channels)

    votes = torch.zeros(1, 5, 20)

    for item in output.items():
        pred = item[1]
        votes = torch.add(votes, pred)

    sleep_stages = torch.argmax(votes, axis=1)
    print(sleep_stages)
    

if __name__=="__main__":
    main()