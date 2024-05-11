import os
import yaml
import torch
import torchvision.utils as vutils

from argparse import ArgumentParser
from torch.nn import functional as F
from torch.fft import fft
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path

from dataset import PLDataModule
from models import *

class DestripeModel(LightningModule):

    def __init__(self, shape, lr=8.31e-5, l_b=3, f_weight=0.001, stripe_weight=1e-4, recurrent_iter=4) -> None:
        super().__init__()
        self.curr_device = None
        self.save_hyperparameters()
        self.learning_rate = lr
        self.f_weight = f_weight
        self.stripe_weight = stripe_weight

        self.model1 = PRN(recurrent_iter=recurrent_iter)
        self.model2 = ExtractStripeSubmodel()

        self.register_buffer("mask", torch.zeros(1, shape[0], shape[1], dtype=torch.bool))
        self.mask[:, :, 0:l_b] = True
        self.mask[:, :, -l_b:] = True

    def forward(self, striped):
        # task 1: denoising
        destriped_predict = self.model1(striped)
        # task 2: extract stripe
        input2 = torch.concat([destriped_predict, striped], dim=1)
        stripes_predict = self.model2(input2)
        
        return destriped_predict, stripes_predict

    def loss_function(self, destriped_predict, stripes_predict, destriped_gt, stripes_gt):
         # loss 1: spatial domain loss
        loss_spatial = F.mse_loss(destriped_predict, destriped_gt)
        # loss 2: frequency domain loss
        freq_gt = torch.abs(fft(destriped_gt, dim=-1)).masked_fill_(self.mask, value=0)
        freq_predict = torch.abs(fft(destriped_predict, dim=-1)).masked_fill_(self.mask, value=0)
        loss_freq = F.mse_loss(freq_gt, freq_predict) * self.f_weight
        # loss 3: stripe reconstruction
        loss_stripe = F.mse_loss(stripes_predict, stripes_gt) * self.stripe_weight

        return loss_spatial, loss_freq, loss_stripe

    def training_step(self, batch, batch_idx):
        # load data
        striped, destriped_gt, stripes_gt = batch
        self.curr_device = striped.device
        
        # model inference
        destriped_predict, stripes_predict = self(striped)
        
        # loss integration
        loss_spatial, loss_freq, loss_stripe = self.loss_function(destriped_predict, stripes_predict, destriped_gt, stripes_gt)
        loss = loss_spatial + loss_freq + loss_stripe

        self.log_dict({"Loss": loss.detach().clone(), 
                        "Spatial Loss": loss_spatial.detach().clone(), 
                        "Freq Loss": loss_freq.detach().clone(),
                        "Stripe Loss": loss_stripe.detach().clone()},
                        sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # load data
        striped, destriped_gt, stripes_gt = batch
        self.curr_device = striped.device
        
        # model inference
        destriped_predict, stripes_predict = self(striped)
        
        # loss integration
        loss_spatial, loss_freq, loss_stripe = self.loss_function(destriped_predict, stripes_predict, destriped_gt, stripes_gt)
        loss = loss_spatial + loss_freq + loss_stripe

        self.log_dict({"Loss (val)": loss.detach().clone(), 
                        "Spatial Loss (val)": loss_spatial.detach().clone(), 
                        "Freq Loss (val)": loss_freq.detach().clone(),
                        "Stripe Loss (val)": loss_stripe.detach().clone()},
                        sync_dist=True)

    def on_validation_end(self) -> None:
        test_striped, _, stripes_gt = next(iter(self.trainer.datamodule.test_dataloader()))
        test_striped = test_striped.to(self.curr_device)

        # model inference
        destriped_predict, stripes_predict = self(test_striped)
        vutils.save_image(destriped_predict.data,
                          os.path.join(self.logger.log_dir, 
                                       "Reconstructions", 
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=4)
        vutils.save_image(test_striped.data,
                          os.path.join(self.logger.log_dir, 
                                       "Origin", 
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=4)
        vutils.save_image(stripes_predict.unsqueeze(-2).repeat([1,1,256,1]).data,
                          os.path.join(self.logger.log_dir, 
                                       "stripes_predict", 
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=4)
        vutils.save_image(stripes_gt.unsqueeze(-2).repeat([1,1,256,1]).data,
                          os.path.join(self.logger.log_dir, 
                                       "stripes_gt", 
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=4)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        

if __name__ == "__main__":

    parser = ArgumentParser()

    # add PROGRAM level args
    parser.add_argument("--max_epochs", type=int, default=3000)
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--auto_lr_find", type=bool, default=True)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=24) 
    parser.add_argument("--recurrent_iter", type=int, default=4)
    
    args = parser.parse_args()

    with open("config.yaml", 'r') as file:
        config = yaml.safe_load(file)

    # For reproducibility
    seed_everything(824, True)
    torch.autograd.set_detect_anomaly(True)
    model = DestripeModel(shape=config['data_params']['shape'], recurrent_iter=args.recurrent_iter)
    
    datamodule = PLDataModule(config['data_params']['data_path'], config['data_params']['shape'], args.batch_size)
    tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                               name=config['model_params']['name'],)
    
    Path(f"{tb_logger.log_dir}/Origin").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/stripes_predict").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/stripes_gt").mkdir(exist_ok=True, parents=True)
    
    trainer = Trainer.from_argparse_args(parser, devices=[0], logger=tb_logger)

    print(f"======= Training {config['model_params']['name']} =======")
    # trainer.tune(model, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)


