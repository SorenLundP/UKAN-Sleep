from ml_architectures.usleep.usleep import USleep
import torch

def main():
    # Initializaing a clean PyTorch version of U-Sleep
    usleep_instance = USleep(num_channels=2, complexity_factor=0.5)

    # Initializing dummy data - batch size 1, 2 channels and 10 sleep epochs
    dummy_data = torch.rand((1, 2, 30*128*10))

    # Feeding data
    output = usleep_instance(dummy_data)

    # Argmaxing across the confidence axis to gain sleep-stages
    sleep_stages = torch.argmax(output, dim=1)
    print(sleep_stages)

    # Initializing a pretrained PyTorch version of U-Sleep
    import os
    checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "usleep_checkpoints/torch/two_channel/Depth10_CF05.ckpt")
    state_dict = torch.load(checkpoint_path)
    usleep_instance.load_state_dict(state_dict)

    output = usleep_instance(dummy_data)

    # Argmaxing across the confidence axis to gain sleep-stages
    sleep_stages = torch.argmax(output, dim=1)
    print(sleep_stages)
    

if __name__=="__main__":
    main()