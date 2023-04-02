import optuna
import yaml
from train import main
import os
import time
import queue


gpu_queue = queue.Queue()

def objective(trial):
    gpu_idx = gpu_queue.get()
    with open('config/train_config.yaml', 'r') as file:
        args = yaml.safe_load(file)
    args['lr'] = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    args['lr_step'] = trial.suggest_int('lr_step', 5, 35, 2)
    args['lr_decay'] = trial.suggest_float('lr_decay', 0.2, 0.8)
    args['batch_size'] = trial.suggest_int('batch_size', 16, 64, 2)
    args['feat_reg_eff'] = trial.suggest_float('feat_reg_eff', 1e-5, 1e-1, log=True)
    args['btch_momentum'] = trial.suggest_float('btch_momentum', 0.2, 0.8)
    args['epochs'] = 50
    args['online'] = False
    args['visualization'] = False
    args['init_weights'] = None
    args['save_model'] = False
    args['stop_counter'] = 4
    args['valid_freq'] = 3
    args['device'] = f'cuda:{gpu_idx}'
    _, best_val_f1 = main(args)
    gpu_queue.put(gpu_idx)
    return best_val_f1
    
if __name__ == '__main__':
    study = optuna.create_study(storage='sqlite:///results/tuning.db', study_name='tuning1', \
                                load_if_exists=True, direction='maximize')
    n_gpus = 2
    for i in range(n_gpus):
        gpu_queue.put(i)
    study.optimize(objective, n_trials=300, n_jobs=n_gpus)
    # importances
    # fig = optuna.visualization.plot_param_importances(study)
    # fig.show()
    # countor
    # fig = optuna.visualization.plot_contour(study)
    # fig.show()
    # parallel
    # fig = optuna.visualization.plot_parallel_coordinate(study)
    # fig.show()
    # history
    # fig = optuna.visualization.plot_optimization_history(study)
    # fig.show()