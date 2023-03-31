import optuna
import yaml
from train import main

def objective(trial):
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
    _, best_val_f1 = main(args)
    return best_val_f1
    
if __name__ == '__main__':
    study = optuna.create_study(storage='sqlite:///results/tuning_test.db', study_name='test1', \
                                load_if_exists=True, direction='maximize')
    study.optimize(objective, n_trials=100)