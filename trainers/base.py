import logging
from utils import LossRecord
from tqdm import tqdm
import torch
import os
import wandb


class BaseTrainer:
    def __init__(self, model_name, device, epochs, eval_freq=5, patience=-1,
                 verbose=False, wandb_log=False, logger=False, saving_best=True, 
                 saving_checkpoint=False, checkpoint_freq=100, saving_path=None):
        self.model_name = model_name
        self.device = device
        self.epochs = epochs
        self.eval_freq = eval_freq
        self.patience = patience
        self.wandb = wandb_log
        self.verbose = verbose
        self.saving_best = saving_best
        self.saving_checkpoint = saving_checkpoint
        self.checkpoint_freq = checkpoint_freq
        self.saving_path = saving_path
        if verbose:
            self.logger = logging.info if logger else print

    def process(self, model, train_loader, valid_loader, test_loader, optimizer, 
                criterion, regularizer=None, scheduler=None, **kwargs):
        if self.verbose:
            self.logger("Start training")
            self.logger("Train dataset size: {}".format(len(train_loader.dataset)))
            self.logger("Valid dataset size: {}".format(len(valid_loader.dataset)))
            self.logger("Test dataset size: {}".format(len(test_loader.dataset)))

        best_epoch = 0
        best_metrics = None
        counter = 0
        
        for epoch in range(self.epochs):
            train_loss_record = self.train(model, train_loader, optimizer, criterion, scheduler, regularizer)
            if self.verbose:
                self.logger("Epoch {} | {} | lr: {:.6f}".format(epoch, train_loss_record, optimizer.param_groups[0]["lr"]))
            if self.wandb:
                wandb.log(train_loss_record.to_dict())
            
            if self.saving_checkpoint and (epoch + 1) % self.checkpoint_freq == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.cpu().state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss_record': train_loss_record.to_dict(),
                    }, os.path.join(self.saving_path, "checkpoint_{}.pth".format(epoch)))
                model.cuda()
                if self.verbose:
                    self.logger("Epoch {} | save checkpoint in {}".format(epoch, self.saving_path))
                
            if (epoch + 1) % self.eval_freq == 0:
                valid_loss_record = self.evaluate(model, valid_loader, criterion, split="valid")
                if self.verbose:
                    self.logger("Epoch {} | {}".format(epoch, valid_loss_record))
                valid_metrics = valid_loss_record.to_dict()
                
                if self.wandb:
                    wandb.log(valid_loss_record.to_dict())
                
                if not best_metrics or valid_metrics['valid_loss'] < best_metrics['valid_loss']:
                    counter = 0
                    best_epoch = epoch
                    best_metrics = valid_metrics
                    torch.save(model.cpu().state_dict(), os.path.join(self.saving_path, "best_model.pth"))
                    model.cuda()
                    if self.verbose:
                        self.logger("Epoch {} | save best models in {}".format(epoch, self.saving_path))
                elif self.patience != -1:
                    counter += 1
                    if counter >= self.patience:
                        if self.verbose:
                            self.logger("Early stop at epoch {}".format(epoch))
                        break

        self.logger("Optimization Finished!")
        
        # load best model
        if not best_metrics:
            torch.save(model.cpu().state_dict(), os.path.join(self.saving_path, "best_model.pth"))
        else:
            model.load_state_dict(torch.load(os.path.join(self.saving_path, "best_model.pth")))
            self.logger("Load best models at epoch {} from {}".format(best_epoch, self.saving_path))        
        model.cuda()
        
        valid_loss_record = self.evaluate(model, valid_loader, criterion, split="valid")
        self.logger("Valid metrics: {}".format(valid_loss_record))
        test_loss_record = self.evaluate(model, test_loader, criterion, split="test")
        self.logger("Test metrics: {}".format(test_loss_record))
        
        if self.wandb:
            wandb.run.summary["best_epoch"] = best_epoch
            wandb.run.summary.update(test_loss_record.to_dict())

    def train(self, model, train_loader, optimizer, criterion, scheduler=None, **kwargs):
        loss_record = LossRecord(["train_loss"])
        model.cuda()
        model.train()
        with tqdm(total=len(train_loader)) as bar:
            for x, y in train_loader:
                x = x.to('cuda')
                y = y.to('cuda')
                # compute loss
                y_pred = model(x).reshape(y.shape)
                data_loss = criterion(y_pred, y)
                loss = data_loss
                # compute gradient
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # record loss and update progress bar
                loss_record.update({"train_loss": loss.item()})
                bar.update(1)
                bar.set_postfix_str("train loss: {:.4f}".format(loss.item()))
            if scheduler is not None:
                scheduler.step()
        return loss_record
    
    def evaluate(self, model, eval_loader, criterion, split="valid", **kwargs):
        loss_record = LossRecord(["{}_loss".format(split)])
        model.eval()
        with torch.no_grad():
            for x, y in eval_loader:
                x = x.to('cuda')
                y = y.to('cuda')
                # compute loss
                y_pred = model(x).reshape(y.shape)
                data_loss = criterion(y_pred, y)
                loss = data_loss
                loss_record.update({"{}_loss".format(split): loss.item()})
        return loss_record


def base_procedure(args):
    raise NotImplementedError
