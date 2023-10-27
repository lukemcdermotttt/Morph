import optuna
from optuna.trial import TrialState
from utils import utils
from data import image_cls
from models import classifiers
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data


def search(trial, cfg: dict) -> float:

    # Determine cuda, mps, or cpu
    if not cfg['device']:
        device = utils.identify_device()
        print(f"Using {device} device")
    else:
        device = cfg['device']

    # Generate the model.
    model = classifiers.simple_mlp(trial).to(device)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", cfg['training']['optimizer']['options'])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the FashionMNIST dataset.
    train_loader, valid_loader = image_cls.get_fashionmnist(cfg)

    # Training of the model.
    for epoch in range(cfg['training']['num_epochs']):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            
            # Limiting training data for faster epochs.
            if cfg['subsets']['use_subsets']:
                if batch_idx * cfg['training']['batch_size'] >= \
                            cfg['training']['batch_size'] * cfg['subsets']['train']:
                    break

            data, target = data.view(data.size(0), -1).to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                # Limiting validation data.
                if cfg['subsets']['use_subsets']:
                    if batch_idx * cfg['training']['batch_size'] >= \
                                cfg['training']['batch_size'] * cfg['subsets']['test']:
                        break
                data, target = data.view(data.size(0), -1).to(device), target.to(device)
                output = model(data)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / min(len(valid_loader.dataset), cfg['subsets']['test'])

        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy



if __name__ == "__main__":

    cfg = utils.parse_configuration(verbose=True)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: search(trial, cfg), 
                    n_trials=cfg['hpo']['n_trials'], 
                    timeout=cfg['hpo']['timeout'])

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))