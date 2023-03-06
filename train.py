import sys
sys.path.insert(0, '../')
import math
import numpy as np
import torch 
from torch.utils.data import DataLoader
from models.Pointnet import PointNetClass, PointNetSeg
from torch.utils.data.dataset import random_split
from tqdm import tqdm
import wandb
import yaml
import sys
sys.path.insert(0, 'tests')
from utils.MNIST3d import MNIST3D


# reproducability
random_seed = 0 
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)


def train(traindata, args, validata=None):
    train_loader = DataLoader(traindata, batch_size=args['batch_size'], shuffle=True)
    best_val_loss, best_val_acc = 1e10, 0
    stop_counter =  0
    # train loop
    for epoch in range(args['epochs']):
        args['model'].train()
        epoch_log = {}
        epoch_loss = 0
        for batch_input, targets in tqdm(train_loader, desc=f'Epoch {epoch}: '):
            batch_input, targets = batch_input.to(args['device']), targets.to(args['device'])
            yout = args['model'](batch_input)
            loss = args['loss'](yout, targets)
            epoch_loss += loss.item() / len(train_loader)
            loss.backward()
            args['optimizer'].step()
        args['scheduler'].step()
        epoch_log['Train loss'] = epoch_loss
        print(f'Epoch {epoch} loss: {epoch_loss}')
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
    train_loader = DataLoader(validdata, batch_size=args['batch_size'], shuffle=True)
    val_loss = 0
    val_acc = 0
    args['model'].eval()
    for batch_input, targets in tqdm(train_loader, desc=f'Validation: '):
        batch_input, targets = batch_input.to(args['device']), targets.to(args['device'])
        yout = args['model'](batch_input)
        loss = args['loss'](yout, targets)
        val_loss += loss.item() / len(train_loader)
        val_acc += (torch.argmax(yout, dim=-1) == targets).sum() / len(validdata)
    print(f'Validation loss: {val_loss}, accuracy {val_acc}')
    return val_loss, val_acc.cpu().numpy()

def test(model, test_dataset):
    test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=True)
    test_acc = 0
    model.eval()
    for batch_input, targets in tqdm(test_loader, desc=f'Testing: '):
        batch_input, targets = batch_input.to(args['device']), targets.to(args['device'])
        yout = model(batch_input)
        test_acc += (torch.argmax(yout, dim=-1) == targets).sum() / len(test_dataset)
    print(f'Testing accuracy: {test_acc}')
    return test_acc.cpu().numpy()

def main(args):
    # loading dataset and splitting to train, valid, test
    dataset = MNIST3D(256, './tests/data')
    traindata, validata, testdata = random_split(dataset, [round(1 - args['valid_per'] - args['test_per'], 2), \
         args['valid_per'], args['test_per']])
    # loading model, optimizer, scheduler, loss func
    model = PointNetSeg(10).to(args['device'])
    # loading weights for the model
    if args['init_weights'] != None:
        model.load_state_dict(torch.load(args['init_weights']))
        print('Loaded weights', args['init_weights'])
    loss = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['l2coef'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args['gamma'], verbose=False)
    args['model'] = model
    args['loss'] = loss
    args['optimizer'] = optimizer
    args['scheduler'] = scheduler
    
    # training the model
    best_loss, best_acc = train(traindata, args, validata)

    # testing the model
    testing_acc = test(model, testdata)
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
