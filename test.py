import optuna

def objective(trial):
    # Define your objective function here
    # Use trial.suggest_*() to suggest parameters
    param_x = trial.suggest_float("x", 0, 5)
    param_y = trial.suggest_float("y", 0, 3)
    # Example objectives
    objective1 = (param_x - 2) ** 2
    objective2 = (param_y - 1) ** 2
    return objective1, objective2

def main():
    study = optuna.create_study(directions=['minimize', 'minimize'], 
                                sampler=optuna.multi_objective.samplers.NSGAIIMultiObjectiveSampler())
    study.optimize(objective, n_trials=1000)
    print("Best trial:", study.best_trial)

if __name__ == "__main__":
    main()