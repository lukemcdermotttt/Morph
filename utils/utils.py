import torch
import argparse
import yaml

# Get cpu, gpu or mps device for training/inference.
def identify_device():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    return torch.device(device)



def parse_configuration(verbose: bool = False) -> dict:

    parser = argparse.ArgumentParser(description="PyTorch Training with YAML Configs and Args")

    # Add argument to specify the YAML configuration file
    parser.add_argument('--config', default='configs/default.yaml', type=str, 
        help='path to the configuration YAML file')

    # Command Line Arguments -> Override yaml config options
    parser.add_argument('--learning_rate', type=float, help='Override learning rate')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--device', type=str, help='Override use_cuda flag')

    args = parser.parse_args()

    # Load the YAML configuration
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # Override the YAML settings with command-line arg
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.device is not None: # Explicitly checking against None because False is a valid value
        config['device'] = args.device

    if verbose:
        print('='*20)
        print('Configuration info:')
        print(config)
        print('='*20)


    return config
