import sys
sys.path.insert(0, '../')
sys.path.insert(0, './utils')
import numpy as np
import torch 
from torch.utils.data import DataLoader
from models.Pointnet import PointNetSeg, feature_transform_reguliarzer, bn_momentum_adjust
from torch.utils.data.dataset import random_split
from tqdm import tqdm
import wandb
import yaml
import sys
from utils.humanDBLoader import humanDBLoader
from utils.pcl_utils import plot_frame_annotation_kitti_v2
sys.path.insert(0, 'tests')


# reproducability
random_seed = 0 
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)

def get_f1_score(predicted, ground_truth):
    f1_score = 0
    predicted = predicted.detach().cpu().numpy() > 0.5
    ground_truth = np.round(ground_truth.detach().cpu().numpy()).astype('int') == 1
    for (yout, yground) in zip(predicted, ground_truth):
        true_pos = (yout & yground).sum()
        false_neg = ((~yout) & yground).sum()
        false_pos = (yout & (~yground)).sum()
        prec = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        rec = true_pos / (true_pos + false_pos) if(true_pos + false_pos) > 0 else 0
        f1_score += 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    return f1_score

def train(traindata, args, validata=None):
    train_loader = DataLoader(traindata, batch_size=None, shuffle=True, pin_memory=True if (args['device']=='cuda') else False)
    best_val_loss, best_val_acc = 1e10, 0
    stop_counter =  0
    # train loop
    for epoch in range(args['epoch_start'], args['epochs']):
        args['model'].train()
        epoch_log, epoch_loss, data_evaluated, imgs = {}, 0, 0, []
        # Updating the training hyperparams
        lr = max(args['lr'] * (args['lr_decay'] ** (epoch // args['lr_step'])), args['lr_clip'])
        for param_group in args['optimizer'].param_groups:
            param_group['lr'] = lr
        momentum = args['btch_momentum'] + (0.99 - args['btch_momentum']) * min(epoch / args['btch_max_epoch'], 1.0)
        args['model'] = args['model'].apply(lambda x: bn_momentum_adjust(x, momentum))
        for batch_input, targets, centers in tqdm(train_loader, desc=f'Epoch {epoch}: '):
            args['optimizer'].zero_grad()
            batch_input, targets = batch_input.to(args['device']), targets.type(torch.FloatTensor).to(args['device'])
            if batch_input.size(0) < 2: continue
            yout, trans, trans_feat = args['model'](batch_input)
            loss = args['loss'](yout.view(-1, args['input_size']), targets.view(-1, args['input_size'])) + args['feat_reg_eff'] * feature_transform_reguliarzer(trans_feat)
            epoch_loss += loss.item()
            loss.backward()
            args['optimizer'].step()
            data_evaluated += batch_input.size(0)
        if (data_evaluated == 0): continue
        epoch_log['Train loss'] = epoch_loss / data_evaluated
        print(f"Epoch {epoch} loss: {epoch_log['Train loss']}")
        # validation
        if((epoch + 1) % args['valid_freq'] == 0) and (validata!=None):
            val_loss, val_acc, imgs = validate(validata, args)
            epoch_log['Validation loss'], epoch_log['Validation accuracy'] = val_loss, val_acc
            if(val_loss < best_val_loss):
                best_val_loss, best_val_acc = val_loss, val_acc
                if args['save_model']:
                    torch.save(args['model'].state_dict(), args['save_path'] + f'E{epoch}_{args["session_name"]}.pt')
            else:
                stop_counter += 1
        # stopping creteria
        if(stop_counter == args['stop_counter']):
            break
        # Online Monitoring
        if args['online']:
            if len(imgs) > 0:
                columns = ['epoch' ,'Image 1', 'Image 2', 'Image3']
                table = wandb.Table(data =[[epoch, *imgs]], columns=columns)
                epoch_log['table_image'] = table
            wandb.log(epoch_log)
    return best_val_loss, best_val_acc

def validate(validdata, args, validata=None):
    valid_loader = DataLoader(validdata, batch_size=None, shuffle=True)
    val_loss, val_acc, data_eval = 0, 0, 0
    input_size = args['input_size']
    args['model'].eval()
    imgs = []
    for batch_input, targets, centers in tqdm(valid_loader, desc=f'Validation: '):
        batch_input, targets = batch_input.to(args['device']), targets.type(torch.FloatTensor).to(args['device'])
        yout, trans, trans_feat  = args['model'](batch_input)
        loss = args['loss'](yout.view(-1, args['input_size']), targets.view(-1, args['input_size']))
        val_loss += loss.item()
        val_acc += get_f1_score(yout.view(-1, args['input_size']), targets.view(-1, args['input_size']))
        data_eval += batch_input.size(0)
        if args['visualization'] and (np.random.rand() < 0.3) and len(imgs) < 3:
            first_pcl = batch_input[0].detach().cpu().numpy()
            first_annot = yout[0].view(-1).detach().cpu().numpy() > 0.5
            imgs.append(plot_frame_annotation_kitti_v2(first_pcl, first_annot))
    if len(imgs) < 3: imgs += [None] * (3 - len(imgs))
    val_loss, val_acc = val_loss / data_eval, val_acc / data_eval
    print(f'Validation loss: {val_loss}, accuracy {val_acc}')
    return val_loss, val_acc, imgs

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
    dataset = humanDBLoader(args['data_root_path'], batch_size=args['batch_size'])
    traindata, validata, testdata = random_split(dataset, [round(1 - args['valid_per'] - args['test_per'], 2), \
         args['valid_per'], args['test_per']])    
    # loading model, optimizer, scheduler, loss func
    model = PointNetSeg(1).to(args['device'])
    # loading weights for the model
    if args['init_weights'] != None:
        model.load_state_dict(torch.load(args['init_weights']))
        print('Loaded weights', args['init_weights'])
    loss = torch.nn.BCELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['l2coef'])
    args['model'] = model
    args['loss'] = loss
    args['optimizer'] = optimizer
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
            project="PointNet",
            name=args['session_name'],
            notes="This is a test",
            tags=["pointnet"],
            config=args,
        )
    main(args)
