import sys
sys.path.insert(0, '../')
sys.path.insert(0, './utils')
import math
import numpy as np
import torch 
from torch.utils.data import DataLoader
from models.Pointnet import PointNetSeg, feature_transform_reguliarzer, bn_momentum_adjust
from torch.utils.data.dataset import random_split
from tqdm import tqdm
import wandb
import yaml
import sys
from utils import o3d_funcs
from utils.humanDBLoader import humanDBLoader, custom_collate
from utils.o3d_funcs import plot_frame_annotation_kitti_v2
sys.path.insert(0, 'tests')


# reproducability
random_seed = 0 
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)

def train(traindata, args, validata=None):
    train_loader = DataLoader(traindata, batch_size=args['batch_size'], shuffle=True, collate_fn=custom_collate)
    best_val_loss, best_val_acc = 1e10, 0
    stop_counter =  0
    # train loop
    for epoch in range(args['epochs']):
        args['model'].train()
        epoch_log = {}
        epoch_loss = 0
        data_evaluated = 0
        lr = max(args['lr'] * (0.5 ** (epoch // 20)), args['lr_clip'])
        for param_group in args['optimizer'].param_groups:
            param_group['lr'] = lr
        momentum = args['btch_momentum'] + (0.99 - args['btch_momentum']) * min(epoch / (args['epochs']), 1.0)
        args['model'] = args['model'].apply(lambda x: bn_momentum_adjust(x, momentum))
        #classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        for batch_input, targets, centers in tqdm(train_loader, desc=f'Epoch {epoch}: '):
            for btch_idx in range(len(batch_input)):
                args['optimizer'].zero_grad()
                mini_btch_input, mini_btch_target = batch_input[btch_idx].to(args['device']), targets[btch_idx].type(torch.long).to(args['device'])
                yout, trans, trans_feat = args['model'](mini_btch_input)
                # if btch_idx == 0:
                #     first_pcl = batch_input[btch_idx][0].detach().cpu().numpy()
                #     first_annot = np.argmax(yout[btch_idx].detach().cpu().numpy(), axis=1).astype('bool')
                #     plot_frame_annotation_kitti_v2(first_pcl, first_annot)
                #     print(yout.size(), mini_btch_target.size())
                loss = args['loss'](yout.view(-1, 2), mini_btch_target.view(-1)) + args['feat_reg_eff'] * feature_transform_reguliarzer(trans_feat)
                epoch_loss += loss.item()
                loss.backward()
                args['optimizer'].step()
                data_evaluated += mini_btch_input.size(0)
        epoch_log['Train loss'] = epoch_loss / data_evaluated
        print(f"Epoch {epoch} loss: {epoch_log['Train loss']}")
        # validation
        if((epoch + 1) % args['valid_freq'] == 0) and (validata!=None):
            val_loss, val_acc = validate(validata, args)
            if(val_loss < best_val_loss):
                best_val_loss, best_val_acc = val_loss, val_acc
                epoch_log['Validation loss'], epoch_log['Validation accuracy'] = val_loss, val_acc
                if args['save_model']:
                    torch.save(args['model'].state_dict(), args['save_path'] + f'E{epoch}_{args["session_name"]}.pt')
            else:
                stop_counter += 1
        # stopping creteria
        if(stop_counter == args['stop_counter']):
            break
        # Online Monitoring
        if args['online']:
            wandb.log(epoch_log)
    return best_val_loss, best_val_acc

def validate(validdata, args, validata=None):
    train_loader = DataLoader(validdata, batch_size=args['batch_size'], shuffle=True, collate_fn=custom_collate)
    val_loss = 0
    val_acc = 0
    args['model'].eval()
    for batch_input, targets, centers in tqdm(train_loader, desc=f'Validation: '):
        for btch_idx in range(len(batch_input)):
            mini_btch_input, mini_btch_target = batch_input[btch_idx].to(args['device']), targets[btch_idx].to(args['device'])
            yout = args['model'](mini_btch_input).squeeze()
            loss = args['loss'](yout, mini_btch_target)
            val_loss += loss.item() / len(train_loader)
            val_acc += (torch.argmax(yout, dim=-1) == targets).sum() / len(validdata)
    print(f'Validation loss: {val_loss}, accuracy {val_acc}')
    return val_loss, val_acc.cpu().numpy()

def test(model, test_dataset):
    test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=True)
    test_acc = 0
    model.eval()
    for batch_input, targets, centers in tqdm(test_loader, desc=f'Testing: '):
        for btch_idx in range(len(batch_input)):
            mini_btch_input, mini_btch_target = batch_input[btch_idx].to(args['device']), targets[btch_idx].to(args['device'])
            yout = args['model'](mini_btch_input).squeeze()
            test_acc += (torch.argmax(yout, dim=-1) == targets).sum() / len(test_dataset)
    print(f'Testing accuracy: {test_acc}')
    return test_acc.cpu().numpy()

def main(args):
    # loading dataset and splitting to train, valid, test
    dataset = humanDBLoader(args['data_root_path'])
    traindata, validata, testdata = random_split(dataset, [round(1 - args['valid_per'] - args['test_per'], 2), \
         args['valid_per'], args['test_per']])
    # loading model, optimizer, scheduler, loss func
    model = PointNetSeg(2).to(args['device'])
    # loading weights for the model
    if args['init_weights'] != None:
        model.load_state_dict(torch.load(args['init_weights']))
        print('Loaded weights', args['init_weights'])
    loss = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['l2coef'])
    args['model'] = model
    args['loss'] = loss
    args['optimizer'] = optimizer
    
    # training the model
    best_loss, best_acc = train(traindata, args, None)

    # testing the model
    #testing_acc = test(model, testdata)
    return best_loss, best_acc
    
if __name__ == '__main__':
    with open('config/train_config.yaml', 'r') as file:
        args = yaml.safe_load(file)
    if args['online']:
        wandb.init(
            project="PointNetTest",
            name=args['session_name'],
            notes="This is a test",
            tags=["pointnet"],
            config=args,
        )
    main(args)
