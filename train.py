import sys
sys.path.insert(0, '../')
sys.path.insert(0, './utils')
import numpy as np
import torch 
from torch.utils.data import DataLoader
from models.Pointnet import PointNetSeg, feature_transform_reguliarzer, bn_momentum_adjust
from models.Pointnet2 import Pointet2
from torch.utils.data.dataset import random_split
from tqdm import tqdm
import wandb
import yaml
import sys
from utils.humanDBLoader import humanDBLoader
from utils.pcl_utils import plot_frame_annotation_kitti_v2
sys.path.insert(0, 'tests')
from time import time 


# reproducability
random_seed = 0 
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)

def compute_classification_metrics(predicted, ground_truth):
    """
    Computes the precision, recall, and F1 score given the predicted and ground truth labels.
    The function first applies a sigmoid function to the predicted tensor, then converts it to a binary
    tensor based on a threshold of 0.5. It does the same for the ground truth tensor. It then calculates
    the precision, recall, and F1 score for each pair of predicted and ground truth labels, and finally
    returns the sum of these metrics over all batches.

    :param predicted: The tensor of predicted labels.
    :param ground_truth: The tensor of ground truth labels.
    :return: The sum of precisions, recalls, and F1 scores over all batches. 
    """
    predicted = predicted.detach().cpu().numpy() >= 0.5
    ground_truth = ground_truth.detach().cpu().numpy() >= 0.5
    prec_arr, rec_arr, f1_arr = [], [], []
    for (yout, yground) in zip(predicted, ground_truth):
        true_pos = (yout & yground).sum()
        false_neg = ((~yout) & yground).sum()
        false_pos = (yout & (~yground)).sum()
        prec = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 1
        prec_arr.append(prec)
        rec = true_pos / (true_pos + false_neg) if(true_pos + false_neg) > 0 else 1
        rec_arr.append(rec)
        f1_arr.append(2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0)
    return np.sum(prec_arr), np.sum(rec_arr), np.sum(f1_arr)

def update_training_hyperparams(args, epoch):
    """
    This is used during training epochs to update the hyperprameters, like learning rate, batch momentum.

    :param args: the training arguments where the model, the optimizer and hyperparameters are defined
    :param epoch: the current epoch number
    :return: the updated model and optimizer
    """
    lr = max(args['lr'] * (args['lr_decay'] ** (epoch // args['lr_step'])), args['lr_clip'])
    for param_group in args['optimizer'].param_groups:
        param_group['lr'] = lr
    momentum = args['btch_momentum'] + (0.99 - args['btch_momentum']) * min(epoch / args['btch_max_epoch'], 1.0)
    args['model'] = args['model'].apply(lambda x: bn_momentum_adjust(x, momentum))
    return args['model'], args['optimizer']

def process_batch(batch_input, targets, args, eval=False):
    """
    Process a batch of pcl performing inference, calculation of loss and other
    classification metrics (precision, recall, f1). In training mode, it will
    perfrom backpropagation and weight update.

    :param batch_input: the batched input in Tensor format
    :param targets: the ground truth labels
    :param args: the training parameters
    :param eval: bool that when is False will perform backpropagation and weight update
    :return: precision, recall, f1, loss of the batch and the simgoid(model's output)
    """
    batch_input, targets = batch_input.to(args['device']), targets.type(torch.FloatTensor).to(args['device'])
    yout, trans_feat = args['model'](batch_input)       # in case of Pointnet2 the second output is different but we dont use it
    loss = args['loss'](yout.view(-1, batch_input.size(1)), targets.view(-1, batch_input.size(1)))
    if args['model_name'] == 'Pointnet':
        loss += args['feat_reg_eff'] * feature_transform_reguliarzer(trans_feat, args['device'])
    batch_loss = loss.item() * batch_input.size(0)         # scaling loss to batch size (loss reduction: mean)
    yout = torch.sigmoid(yout)
    btc_prec, btc_rec, btc_f1 = compute_classification_metrics(yout.view(-1, batch_input.size(1)), targets.view(-1, batch_input.size(1)))
    if not eval:
        loss.backward()    
        args['optimizer'].step()   # updating weights
    return (btc_prec, btc_rec, btc_f1, batch_loss), yout

def dynamic_class_balancing(loss_f, weight_adapt, val_prec, val_rec, default_val):
    """
    This balance the postive and negative population. It is is dynamic bancing 
    based on the validation precision and recall. When precision is higher than recall
    it adapts the positive weight to higher value to increase recall.

    :param loss_f: the bincary cross entropy loss function
    :param weight_adapt: a free parameter to give more importance to precision (when smaller than one) or recall
    :param val_prec: the validation precision
    :param val_rec: the validation recall
    :param default_val: a default value, used when we have not valid scores from validation
    :return: the new loss function with updated positive weight
    """
    adapt_prob = weight_adapt * val_prec / val_rec if val_rec > 0 else default_val
    loss_f.pos_weight = torch.tensor(adapt_prob).to(loss_f.pos_weight.device)
    return loss_f

def log_training(epoch_log, eval=False):
    """
    This is the main logger for the training procedure. It will print to colcole
    the loss, precision, recall and f1. In case of online monitoring it will send to
    weight and bias too.

    :param epoch_log: a dictionary with epoch number, total precision, recall and f1 and the total 
    examples that evaluated
    :param eval: if true then we are on validation else we are on training
    :return: epoch_log normalized
    """
    epoch_log['loss'], epoch_log['prec'], epoch_log['recall'], epoch_log['f1'] = \
    epoch_log['loss'] / epoch_log['data_evaluated'], \
    epoch_log['prec'] / epoch_log['data_evaluated'],  \
    epoch_log['recall'] / epoch_log['data_evaluated'], \
    epoch_log['f1'] / epoch_log['data_evaluated']
    if not eval:
        print(f"Epoch {epoch_log['epoch']}, loss: {epoch_log['loss']}, Precision: {epoch_log['prec']}, " + \
                f"Recall: {epoch_log['recall']}, F1: {epoch_log['f1']}")
    else:
        print(f"Validation: loss: {epoch_log['loss']}, Precision: {epoch_log['prec']}, " + \
                f"Recall: {epoch_log['recall']}, F1: {epoch_log['f1']}")
    return epoch_log

def log_online(epoch_log, val_log):
    """
    This is used to monitor online in weight and biases

    :param epoch_log: this is the epoch log
    :param val_log: this is used when validation log is available
    :return: None
    """
    online_log = {}
    online_log['train_loss'], online_log['train_prec'], \
    online_log['train_recall'], online_log['train_f1'] = \
    epoch_log['loss'], epoch_log['prec'], \
    epoch_log['recall'], epoch_log['f1']
    if len(val_log.keys()) > 0:
        online_log['valid_loss'], online_log['valid_prec'], \
        online_log['valid_recall'], online_log['valid_f1'] = \
        val_log['loss'], val_log['prec'], \
        val_log['recall'], val_log['f1']
        if len(val_log['imgs']) > 0:
            columns = ['epoch' ,'Image 1', 'Image 2', 'Image3']
            table = wandb.Table(data =[[epoch_log['epoch'], *val_log['imgs']]], columns=columns)
            online_log['table_image'] = table
    wandb.log(online_log)

def train_pointnet(traindata, args, valid_data=None):
    train_loader = DataLoader(traindata, batch_size=None, shuffle=True, pin_memory=True if (args['device'][:4]=='cuda') else False)
    best_val_loss, best_val_f1 = 1e10, 0
    stop_counter =  0

    # train loop
    for epoch in range(args['epoch_start'], args['epochs']):
        args['model'].train()
        epoch_log, valid_log = {'epoch': epoch, 'prec': 0, 'recall': 0, 'f1': 0, 'loss': 0, 'data_evaluated':0}, {}
        args['model'], args['optimizer'] = update_training_hyperparams(args, epoch)

        # batch loop
        for batch_input, targets, _ in tqdm(train_loader, desc=f'Epoch {epoch}: '):
            if batch_input.size(0) < 2: continue # skip batches with single or none items
            args['optimizer'].zero_grad()
            (btc_prec, btc_rec, btc_f1, btc_loss), _ = process_batch(batch_input, targets, args, False)
            epoch_log['prec'], epoch_log['recall'], epoch_log['f1'], epoch_log['loss'] = \
            epoch_log['prec'] + btc_prec, epoch_log['recall'] + btc_rec, epoch_log['f1'] + btc_f1, epoch_log['loss'] + btc_loss
            epoch_log['data_evaluated'] += batch_input.size(0)
        if (epoch_log['data_evaluated'] == 0): continue
        epoch_log = log_training(epoch_log, False)
        
        # validation and model saving
        if((epoch + 1) % args['valid_freq'] == 0) and (valid_data!=None):
            valid_log = validate(valid_data, args)
            args['loss'] = dynamic_class_balancing(args['loss'], args['weight_adapt'], valid_log['prec'], \
                                                   valid_log['recall'], np.log10(9.7796)+1)
            if(valid_log['f1'] > best_val_f1):
                best_val_loss, best_val_f1, stop_counter = valid_log['loss'], valid_log['f1'], 0
            else:   stop_counter += 1
            if args['save_model']:
                torch.save(args['model'].state_dict(), args['save_path'] + f'E{epoch}_{args["session_name"]}.pt')

        if args['online']:  log_online(epoch_log, valid_log)
        
        # stopping creteria
        if(stop_counter == args['stop_counter']): break
    return best_val_loss, best_val_f1

def validate(valid_data, args):
    valid_loader = DataLoader(valid_data, batch_size=None, shuffle=True)
    valid_log = {'prec': 0, 'recall': 0, 'f1': 0, 'loss': 0, 'data_evaluated':0}
    args['model'].eval()
    imgs = []
    for batch_input, targets, _ in tqdm(valid_loader, desc=f'Validation: '):
        batch_input, targets = batch_input.to(args['device']), targets.type(torch.FloatTensor).to(args['device'])
        (btc_prec, btc_rec, btc_f1, btc_loss), yout = process_batch(batch_input, targets, args, True)
        valid_log['prec'], valid_log['recall'], valid_log['f1'], valid_log['loss'] = \
            valid_log['prec'] + btc_prec, valid_log['recall'] + btc_rec, valid_log['f1'] + \
                  btc_f1, valid_log['loss'] + btc_loss
        valid_log['data_evaluated'] += batch_input.size(0)
        if (np.random.rand() < 0.3) and (len(imgs) < 3):        # if none image have beed picked, pick with probability 0.3 the first element in batch
            first_pcl = batch_input[0].detach().cpu().numpy()
            first_annot = yout[0].view(-1).detach().cpu().numpy() > 0.5
            imgs.append(plot_frame_annotation_kitti_v2(first_pcl, first_annot))
    if len(imgs) < 3: imgs += [None] * (3 - len(imgs))
    valid_log = log_training(valid_log, True)
    valid_log['imgs'] = imgs
    return valid_log

def main(args):
    # loading dataset and splitting to train, valid, test
    traindata, validata = humanDBLoader(args['train_data'], batch_size=args['batch_size'], augmentation=args['augmentation']), \
                         humanDBLoader(args['valid_data'], batch_size=args['batch_size'], augmentation=False)
    
    # loading model, optimizer, scheduler, loss func
    if(args['model_name'] == 'Pointnet'):
        args['model'] = PointNetSeg(1, device=args['device']).to(args['device'])
    elif(args['model_name'] == 'Pointnet2'):
        args['model'] = Pointet2(args['p2_range']).to(args['device'])
    args['loss'] = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.Tensor([np.log10(9.7796) + 1])).to(args['device'])
    args['optimizer'] = torch.optim.Adam(args['model'].parameters(), lr=args['lr'], weight_decay=args['l2coef'])

   # loading weights for the model
    if args['init_weights'] != None:
        args['model'].load_state_dict(torch.load(args['init_weights']))
        print('Loaded weights', args['init_weights'])
    
    best_loss, best_acc = train_pointnet(traindata, args, validata)
    return best_loss, best_acc
    
if __name__ == '__main__':
    with open('config/train_config.yaml', 'r') as file:
        args = yaml.safe_load(file)
    print(f"Session started. Parameters: {args}")
    if args['online']:
        wandb.init(
            project="PointNet2",
            name=args['session_name'],
            notes="This is a test",
            tags=["pointnet"],
            config=args,
        )
    main(args)
