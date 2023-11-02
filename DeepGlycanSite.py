import os
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric.loader as geom_data
import pytorch_lightning   as pl
from   pytorch_lightning.callbacks import ModelCheckpoint
from   pytorch_lightning.callbacks import EarlyStopping
from   pytorch_lightning.callbacks import TQDMProgressBar
from   pytorch_lightning.loggers   import TensorBoardLogger
from   pytorch_lightning.loggers   import TensorBoardLogger
from   pytorch_lightning.loggers   import CSVLogger
from   src.util import LoadFromFile
from   src.util import set_seeds
from   model.visreceptor.model import create_model as cr
from   model.visreceptor.model import Combine
from pytorch_lightning.strategies.ddp import DDPStrategy
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.set_float32_matmul_precision('medium')

parser = argparse.ArgumentParser('Train a GCN Prediction Model')
parser.add_argument('--conf', '-c', type=open, action=LoadFromFile, help='Configuration yaml file')
parser.add_argument('--batch-size',  help='training batch size', type=int, default=6)
parser.add_argument('--seed',type=int,  help='random seed', default=42)
parser.add_argument('--dropout_rate',  help='dropout rate', type=float, default=0.15)
parser.add_argument('--hidden_dim',  help='hidden dimension', type=int, default=256)
parser.add_argument('--residual_layers',  help='residual layers', type=int, default=10)
parser.add_argument('--num_heads_visnet',  help='number of heads', type=int, default=32)
parser.add_argument('--num_heads_Transformer',  help='number of heads', type=int, default=8)
parser.add_argument('--num_encoder_layers',  help='number of layers', type=int, default=3)
parser.add_argument('--num_decoder_layers',  help='number of layers', type=int, default=3)
parser.add_argument('--num_layers_visnet',  help='number of layers', type=int, default=9)
parser.add_argument('--num_rbf',  help='number of rbf', type=int, default=64)
parser.add_argument('--lmax',  help='lmax', type=int, default=2)
parser.add_argument('--trainable_rbf',  help='trainable rbf', action='store_true')
parser.add_argument('--vecnorm_trainable',  help='vecnorm trainable', action='store_true')
parser.add_argument('--lr',  help='learning rate', type=float, default=1e-4)
parser.add_argument('--weight_decay',  help='weight decay', type=float, default=5e-4)
parser.add_argument('--lr_factor',  help='learning rate factor', type=float, default=0.8)
parser.add_argument('--lr_patience',  help='learning rate patience', type=int, default=5)
parser.add_argument('--lr_min',  help='learning rate min', type=float, default=1e-7)
parser.add_argument('--loss_alpha',  help='loss alpha', type=float, default=0.25)
parser.add_argument('--loss_gamma',  help='loss gamma', type=float, default=2)
parser.add_argument('--proj_name',  help='project name', type=str, default='visnet')
parser.add_argument('--output-path', help='path for outputs (default: stdout and without saving)')
parser.add_argument('--train-set', help='path for train set')
parser.add_argument('--val-set', help='path for val set')
parser.add_argument('--results-dir', help='path for results')


class WeightedFocalLoss(torch.nn.Module):
    "Weighted version of Focal Loss"    
    def __init__(self, alpha=.25, gamma=2):
            super(WeightedFocalLoss, self).__init__()        
            self.alpha = torch.tensor([alpha, 1-alpha])
            self.gamma = gamma
            
    def forward(self, inputs, targets):
            device = inputs.device  # Get the device of the input tensors
            self.alpha = self.alpha.to(device) 
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')        
            targets = targets.type(torch.long)        
            at = self.alpha.gather(0, targets.data.view(-1))      
            pt = torch.exp(-BCE_loss)        
            F_loss = at*(1-pt)**self.gamma * BCE_loss        
            return F_loss.mean()

class TrainSite(pl.LightningModule):

    def __init__(self, **model_kwargs):
        super().__init__()
        # Saving hyperparameters

        self.save_hyperparameters()
        # print(model_kwargs)
        self.model = Combine(cr({
    "activation":              'silu',
    "aggr":                    'add',
    "x_dimension":             1309,
    "attn_activation":         'silu',
    "edge_dimension":            9,
    "clip_norm":               10.0,
    "cutoff_lower":            0.0,
    "cutoff_upper":            10.0,
    "distance_influence":      'both',
    "dropout":                 model_kwargs['dropout_rate'],
    "embedding_dimension":     model_kwargs['hidden_dim'],
    "lmax":                    model_kwargs['lmax'],
    "model":                   'ViSNetBlock',
    "neighbor_embedding":      True,
    "num_heads":               model_kwargs['num_heads_visnet'],
    "num_layers":              model_kwargs['num_layers_visnet'],
    "num_rbf":                 model_kwargs['num_rbf'],
    "output_model":            'ScalarKD',
    "power":                   1,
    "rbf_type":                'expnorm',
    "reduce_op":               'add',
    "trainable_rbf":           model_kwargs['trainable_rbf'],
    "vecnorm_trainable":       model_kwargs['vecnorm_trainable'],
    "vecnorm_type":            'max_min',
}), d_model=model_kwargs['hidden_dim'], 
nhead = model_kwargs['num_heads_Transformer'], 
num_encoder_layers=model_kwargs['num_encoder_layers'],
num_decoder_layers=model_kwargs['num_decoder_layers'], 
dropout_rate=model_kwargs['dropout_rate'])
        
        self.loss_fn = WeightedFocalLoss(alpha=model_kwargs['loss_alpha'],gamma=model_kwargs['loss_gamma'])
        self.lr = model_kwargs['lr']
        self.weight_decay = model_kwargs['weight_decay']
        self.lr_factor = model_kwargs['lr_factor']
        self.lr_patience = model_kwargs['lr_patience']
        self.lr_min = model_kwargs['lr_min']
        self.result = []
        self.batch_size = model_kwargs['batch_size']
        self.results_dir = model_kwargs['results_dir']

    def forward(self, data, mode="train"):
        if mode == 'train':
            ligand, target = data
            pi = self.model(target)           
            label = target.label.float()
            pi = pi.squeeze()
            # print(label.shape,'label',pi.shape,'pi')
            loss_fn = self.loss_fn
            mdn = loss_fn(pi, label)
            # wandb.log({"loss": mdn})

            return mdn

        elif mode == 'val':
            ligand, target = data
            label = target.label

            pi = self.model(target)           
            pi = pi.squeeze()
            predictions = [1 if o >= 0.5 else 0 for o in pi]
            label = label.cpu().detach().numpy().squeeze()
            mcc = matthews_corrcoef(label, predictions)
            auc = roc_auc_score(label, predictions)
            # wandb.log({"vali mcc": mcc,"vali auc": auc})

            return mcc

        else:
            ligand, target = data
            label = target.label

            pi = self.model(target)           
            pi = pi.squeeze()
            predictions = [1 if o >= 0.5 else 0 for o in pi]
            label = label.cpu().detach().numpy().squeeze()
            mcc = matthews_corrcoef(label, predictions)

            return mcc

    def configure_optimizers(self):
        # High lr because of small dataset and small model
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(
            optimizer,
            "max",
            factor=self.lr_factor,
            patience=self.lr_patience,
            min_lr=self.lr_min,
        )
        lr_scheduler = {
            "scheduler": scheduler,
            "monitor": "val_mcc",
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]

    def optimizer_step(self, *args, **kwargs):
        optimizer = kwargs["optimizer"] if "optimizer" in kwargs else args[2]
        if self.trainer.global_step < 100:
            lr_scale = min(
                1.0,
                float(self.trainer.global_step + 1)
                / float(100),
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * 0.0001
        super().optimizer_step(*args, **kwargs)
        optimizer.zero_grad()    
    
    def training_step(self, batch, batch_idx):
        loss = self.forward(batch, mode="train")
        self.log("train_loss", loss, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        mcc = self.forward(batch, mode="val")
        self.log("val_mcc", mcc, batch_size=self.batch_size,sync_dist=True)

    def test_step(self, batch, batch_idx):
        mcc = self.forward(batch, mode="val")
        self.log("test_mcc", mcc, batch_size=self.batch_size,sync_dist=True)

def get_pl_trainer(checkpoint, num_epochs):
    """ Create a PyTorch Lightning trainer with the generation callback """

    root_dir = os.path.join(checkpoint, "results")
    os.makedirs(root_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(dirpath=root_dir,
    mode="max", 
    monitor="val_mcc",        
    save_top_k=50,  # -1 to save all
    save_last=True,
    filename="{epoch}-{val_mcc:.4f}")
    ddp_plugin = DDPStrategy(find_unused_parameters=True)
    early_stopping = EarlyStopping("val_mcc", patience=15, mode="max")
    tb_logger = TensorBoardLogger(root_dir, name="tensorbord", version="", default_hp_metric=False)
    csv_logger = CSVLogger(root_dir, name="", version="")
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        callbacks=[TQDMProgressBar(refresh_rate=10), checkpoint_callback, early_stopping],
        accelerator = 'gpu',
        devices = 4,
        strategy=ddp_plugin,
        max_epochs=num_epochs,
        logger=[tb_logger, csv_logger],
        log_every_n_steps=5,
    )
    trainer.logger._default_hp_metric = None
    return trainer

def main():
    global cr_inp

    args = parser.parse_args()
    args = vars(args)
    os.makedirs(args['output_path'], exist_ok=True)
    print(args)
    set_seeds(args['seed'])
    pl.seed_everything(args['seed'])




    dataset0 = torch.load(args['train_set'])
    dataset3 = torch.load(args['val_set'])
    dataset4 = torch.load('20230418/data/all_test.pt')

    
    train_list = dataset0
    vali_list  = dataset3
    test_list  = dataset4

    train_loader = geom_data.DataLoader(train_list, batch_size=args['batch_size'], num_workers=32, shuffle=True)
    vali_loader  = geom_data.DataLoader(vali_list,  batch_size=args['batch_size'], num_workers=32)   
    test_loader  = geom_data.DataLoader(test_list,  batch_size=args['batch_size'], num_workers=32)


    model = TrainSite(seed = args['seed'], dropout_rate = args['dropout_rate'], batch_size=args['batch_size'], results_dir = args['output_path'],
    hidden_dim = args['hidden_dim'], residual_layers = args['residual_layers'],
    num_heads_visnet = args['num_heads_visnet'], num_heads_Transformer = args['num_heads_Transformer'],
    num_encoder_layers = args['num_encoder_layers'], num_decoder_layers = args['num_decoder_layers'],
    num_layers_visnet = args['num_layers_visnet'], num_rbf = args['num_rbf'], lmax = args['lmax'],
    trainable_rbf = args['trainable_rbf'], vecnorm_trainable = args['vecnorm_trainable'],
    lr = args['lr'], weight_decay = args['weight_decay'], lr_factor = args['lr_factor'],
    lr_patience = args['lr_patience'], lr_min = args['lr_min'], loss_alpha = args['loss_alpha'],
    loss_gamma = args['loss_gamma'])

    trainer = get_pl_trainer(args['output_path'], 100)
    trainer.fit(model, train_loader, vali_loader)
    trainer.test(model, dataloaders=test_loader, verbose=True)

    print(r'best val mcc:',   trainer.callback_metrics['val_mcc'].item())
    print(r'best train loss:', trainer.callback_metrics['train_loss'].item())
    print(r'best test mcc:', trainer.callback_metrics['test_mcc'].item())
    print(trainer.checkpoint_callback.best_model_path,'best_path')


if __name__ == '__main__':
    main()
