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

def get_f1_score(predicted, ground_truth):
    predicted = torch.sigmoid(predicted)
    predicted = predicted.detach().cpu().numpy() > 0.5
    ground_truth = ground_truth.detach().cpu().numpy() > 0.5
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

def train_pointnet(traindata, args, validata=None):
    train_loader = DataLoader(traindata, batch_size=None, shuffle=True, pin_memory=True if (args['device'][:4]=='cuda') else False)
    best_val_loss, best_val_f1 = 1e10, 0
    stop_counter =  0
    # train loop
    for epoch in range(args['epoch_start'], args['epochs']):
        args['model'].train()
        epoch_log, epoch_loss, data_evaluated, imgs = {'prec': 0, 'recall': 0, 'f1': 0}, 0, 0, []
        # Updating the training hyperparams
        lr = max(args['lr'] * (args['lr_decay'] ** (epoch // args['lr_step'])), args['lr_clip'])
        for param_group in args['optimizer'].param_groups:
            param_group['lr'] = lr
        momentum = args['btch_momentum'] + (0.99 - args['btch_momentum']) * min(epoch / args['btch_max_epoch'], 1.0)
        args['model'] = args['model'].apply(lambda x: bn_momentum_adjust(x, momentum))
        # batch loop
        for batch_input, targets, centers in tqdm(train_loader, desc=f'Epoch {epoch}: '):
            args['optimizer'].zero_grad()
            batch_input, targets = batch_input.to(args['device']), targets.type(torch.FloatTensor).to(args['device'])
            if batch_input.size(0) < 2: continue
            yout, trans, trans_feat = args['model'](batch_input)
            # adapt_prob = torch.sigmoid(yout).detach().cpu().numpy().sum() / (batch_input.size(0) * args['input_size'])
            # args['loss'].pos_weight = torch.tensor([(np.log10(9.7796)+1) ** (1 - adapt_prob)]).to(args['device'])
            loss = args['loss'](yout.view(-1, args['input_size']), targets.view(-1, args['input_size'])) + args['feat_reg_eff'] * feature_transform_reguliarzer(trans_feat, args['device'])
            epoch_loss += loss.item() * batch_input.size(0)         # scaling loss to batch size (loss reduction: mean)
            loss.backward()     # gradient calculation
            btc_prec, btc_rec, btc_f1 = get_f1_score(yout.view(-1, args['input_size']), targets.view(-1, args['input_size']))
            epoch_log['prec'], epoch_log['recall'], epoch_log['f1'] = epoch_log['prec'] + btc_prec, epoch_log['recall'] + btc_rec, epoch_log['f1'] + btc_f1
            args['optimizer'].step()   # updating weights
            data_evaluated += batch_input.size(0)
        if (data_evaluated == 0): continue
        epoch_log['Train loss'], epoch_log['prec'], epoch_log['recall'], epoch_log['f1'] = epoch_loss / data_evaluated, \
        epoch_log['prec'] / data_evaluated, epoch_log['recall'] / data_evaluated, epoch_log['f1'] / data_evaluated
        print(f"Epoch {epoch}, loss: {epoch_log['Train loss']}, Precision: {epoch_log['prec']}, " + \
              f"Recall: {epoch_log['recall']}, F1: {epoch_log['f1']}")
        # validation and model saving
        if((epoch + 1) % args['valid_freq'] == 0) and (validata!=None):
            val_loss, val_prec, val_rec, val_f1, imgs = validate(validata, args)
            epoch_log['valid_loss'], epoch_log['valid_prec'], epoch_log['valid_rec'], epoch_log['valid_f1'] = \
                val_loss, val_prec, val_rec, val_f1
            if epoch > 15:
                adapt_prob = val_prec / val_rec if val_rec > 0 else np.log10(9.7796)+1
                args['loss'].pos_weight = torch.tensor(adapt_prob).to(args['device'])
            if(val_f1 > best_val_f1):
                best_val_loss, best_val_f1 = val_loss, val_f1
                stop_counter = 0
            else:
                stop_counter += 1
            if args['save_model']:
                    torch.save(args['model'].state_dict(), args['save_path'] + f'E{epoch}_{args["session_name"]}.pt')
        # Online Monitoring
        if args['online']:
            if len(imgs) > 0:
                columns = ['epoch' ,'Image 1', 'Image 2', 'Image3']
                table = wandb.Table(data =[[epoch, *imgs]], columns=columns)
                epoch_log['table_image'] = table
            wandb.log(epoch_log)
        # stopping creteria
        if(stop_counter == args['stop_counter']):
            break
    return best_val_loss, best_val_f1

def train_pointnet2(traindata, args, validata=None):
    train_loader = DataLoader(traindata, batch_size=None, shuffle=True, pin_memory=True if (args['device'][:4]=='cuda') else False)
    best_val_loss, best_val_f1 = 1e10, 0
    stop_counter =  0
    # train loop
    for epoch in range(args['epoch_start'], args['epochs']):
        args['model'].train()
        epoch_log, epoch_loss, data_evaluated, imgs = {'prec': 0, 'recall': 0, 'f1': 0}, 0, 0, []
        # Updating the training hyperparams
        lr = max(args['lr'] * (args['lr_decay'] ** (epoch // args['lr_step'])), args['lr_clip'])
        for param_group in args['optimizer'].param_groups:
            param_group['lr'] = lr
        momentum = args['btch_momentum'] + (0.99 - args['btch_momentum']) * min(epoch / args['btch_max_epoch'], 1.0)
        args['model'] = args['model'].apply(lambda x: bn_momentum_adjust(x, momentum))
        # batch loop
        for batch_input, targets, centers in tqdm(train_loader, desc=f'Epoch {epoch}: '):
            args['optimizer'].zero_grad()
            batch_input, targets = batch_input.to(args['device']), targets.type(torch.FloatTensor).to(args['device'])
            if batch_input.size(0) < 2: continue
            yout, _ = args['model'](batch_input)
            # adapt_prob = torch.sigmoid(yout).detach().cpu().numpy().sum() / (batch_input.size(0) * args['input_size'])
            # args['loss'].pos_weight = torch.tensor([(np.log10(9.7796)+1) ** (1 - adapt_prob)]).to(args['device'])
            loss = args['loss'](yout.view(-1, args['input_size']), targets.view(-1, args['input_size']))
            epoch_loss += loss.item() * batch_input.size(0)         # scaling loss to batch size (loss reduction: mean)
            loss.backward()     # gradient calculation
            btc_prec, btc_rec, btc_f1 = get_f1_score(yout.view(-1, args['input_size']), targets.view(-1, args['input_size']))
            epoch_log['prec'], epoch_log['recall'], epoch_log['f1'] = epoch_log['prec'] + btc_prec, epoch_log['recall'] + btc_rec, epoch_log['f1'] + btc_f1
            args['optimizer'].step()   # updating weights
            data_evaluated += batch_input.size(0)
        if (data_evaluated == 0): continue
        epoch_log['Train loss'], epoch_log['prec'], epoch_log['recall'], epoch_log['f1'] = epoch_loss / data_evaluated, \
        epoch_log['prec'] / data_evaluated, epoch_log['recall'] / data_evaluated, epoch_log['f1'] / data_evaluated
        print(f"Epoch {epoch}, loss: {epoch_log['Train loss']}, Precision: {epoch_log['prec']}, " + \
              f"Recall: {epoch_log['recall']}, F1: {epoch_log['f1']}")
        # validation and model saving
        if((epoch + 1) % args['valid_freq'] == 0) and (validata!=None):
            val_loss, val_prec, val_rec, val_f1, imgs = validate(validata, args)
            epoch_log['valid_loss'], epoch_log['valid_prec'], epoch_log['valid_rec'], epoch_log['valid_f1'] = \
                val_loss, val_prec, val_rec, val_f1
            adapt_prob = val_prec / val_rec if val_rec > 0 else np.log10(9.7796)+1
            args['loss'].pos_weight = torch.tensor(adapt_prob).to(args['device'])
            if(val_f1 > best_val_f1):
                best_val_loss, best_val_f1 = val_loss, val_f1
                stop_counter = 0
            else:
                stop_counter += 1
            if args['save_model']:
                torch.save(args['model'].state_dict(), args['save_path'] + f'E{epoch}_{args["session_name"]}.pt')
        # Online Monitoring
        if args['online']:
            if len(imgs) > 0:
                columns = ['epoch' ,'Image 1', 'Image 2', 'Image3']
                table = wandb.Table(data =[[epoch, *imgs]], columns=columns)
                epoch_log['table_image'] = table
            wandb.log(epoch_log)
        # stopping creteria
        if(stop_counter == args['stop_counter']):
            break
    return best_val_loss, best_val_f1

def validate(validdata, args, validata=None):
    valid_loader = DataLoader(validdata, batch_size=None, shuffle=True)
    val_loss, prec, recall, f1, data_eval = 0, 0, 0, 0, 0
    args['model'].eval()
    imgs = []
    for batch_input, targets, centers in tqdm(valid_loader, desc=f'Validation: '):
        batch_input, targets = batch_input.to(args['device']), targets.type(torch.FloatTensor).to(args['device'])
        if args['model_name'] == 'Pointnet':
            yout, trans, trans_feat = args['model'](batch_input)
        elif args['model_name'] == 'Pointnet2': 
            yout, _ = args['model'](batch_input)
        loss = args['loss'](yout.view(-1, args['input_size']), targets.view(-1, args['input_size']))
        val_loss += loss.item() * batch_input.size(0)         # scaling loss to batch size (loss reduction: mean)
        scores = get_f1_score(yout.view(-1, args['input_size']), targets.view(-1, args['input_size']))
        prec, recall, f1 = prec + scores[0], recall + scores[1], f1 + scores[2]
        data_eval += batch_input.size(0)
        if args['visualization'] and (np.random.rand() < 0.3) and len(imgs) < 3:
            first_pcl = batch_input[0].detach().cpu().numpy()
            first_annot = yout[0].view(-1).detach().cpu().numpy() > 0.5
            imgs.append(plot_frame_annotation_kitti_v2(first_pcl, first_annot))
    if len(imgs) < 3: imgs += [None] * (3 - len(imgs))
    val_loss, prec, recall, f1 = val_loss / data_eval, prec / data_eval, recall / data_eval, f1 / data_eval
    print(f'Validation loss: {val_loss}, Precision: {prec}, Recall: {recall}, f1: {f1}')
    return val_loss, prec, recall, f1, imgs

def test(model, test_dataset):
    test_loader = DataLoader(test_dataset, batch_size=None, shuffle=True)
    test_acc, data_eval = 0, 0
    model.eval()
    for batch_input, targets, centers in tqdm(test_loader, desc=f'Testing: '):
        batch_input, targets = batch_input.to(args['device']), targets.type(torch.long).to(args['device'])
        yout, _, _  = args['model'](batch_input)
        test_acc +=  get_f1_score(targets.view(-1, args['input_size']), targets.view(-1, args['input_size']))
        data_eval += batch_input.size(0)
    test_acc /= data_eval
    print(f'Testing accuracy: {test_acc}')
    return test_acc

def main(args):
    # loading dataset and splitting to train, valid, test
    traindata, validata, testdata = humanDBLoader(args['train_data'], batch_size=args['batch_size']), \
                         humanDBLoader(args['valid_data'], batch_size=args['batch_size']), \
                        humanDBLoader(args['test_data'], batch_size=args['batch_size'])
    # loading model, optimizer, scheduler, loss func
    if(args['model_name'] == 'Pointnet'):
        model = PointNetSeg(1, device=args['device']).to(args['device'])
        loss = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.Tensor([np.log10(9.7796) + 1])).to(args['device'])
        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['l2coef'])
    elif(args['model_name'] == 'Pointnet2'):
        model = Pointet2().to(args['device'])
        loss = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.Tensor([np.log10(9.7796) + 1])).to(args['device'])
        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['l2coef'])
    # loading weights for the model
    if args['init_weights'] != None:
        model.load_state_dict(torch.load(args['init_weights']))
        print('Loaded weights', args['init_weights'])
    args['model'] = model
    args['loss'] = loss
    args['optimizer'] = optimizer
    if(args['model_name'] == 'Pointnet'):
        best_loss, best_acc = train_pointnet(traindata, args, validata)
    elif(args['model_name'] == 'Pointnet2'):
        best_loss, best_acc = train_pointnet2(traindata, args, validata)
    # testing the model
    # testing_acc = test(model, testdata)
    return best_loss, best_acc
    
if __name__ == '__main__':
    with open('config/train_config.yaml', 'r') as file:
        args = yaml.safe_load(file)
    if args['online']:
        wandb.init(
            project="PointNet2",
            name=args['session_name'],
            notes="This is a test",
            tags=["pointnet"],
            config=args,
        )
    main(args)
