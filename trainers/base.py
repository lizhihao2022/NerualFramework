import logging
from utils import LossRecord
from tqdm import tqdm
import torch
import os
import wandb
from time import time
from functools import partial

from datasets import MyDataset


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
    
    def get_initializer(self, name):
        if name is None:
            return None
        
        if name == 'xavier_normal':
            init_ = partial(torch.nn.init.xavier_normal_)
        elif name == 'kaiming_uniform':
            init_ = partial(torch.nn.init.kaiming_uniform_)
        elif name == 'kaiming_normal':
            init_ = partial(torch.nn.init.kaiming_normal_)
        return init_
    
    def build_optimizer(self, model, args, **kwargs):
        if args['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args['lr'],
                weight_decay=args['weight_decay'],
            )
        elif args['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=args['lr'],
                momentum=args['momentum'],
                weight_decay=args['weight_decay'],
            )
        else:
            raise NotImplementedError("Optimizer {} not implemented".format(args['optimizer']))
        return optimizer
    
    def build_scheduler(self, optimizer, args, **kwargs):
        if args['scheduler'] == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=args['milestones'],
                gamma=args['gamma'],
            )
        elif args['scheduler'] == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=args['lr'],
                div_factor=args['div_factor'],
                final_div_factor=args['final_div_factor'],
                pct_start=args['pct_start'],
                steps_per_epoch=args['steps_per_epoch'],
                epochs=args['epochs'],
            )
        elif args['scheduler'] == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=args['step_size'],
                gamma=args['gamma'],
            )
        else:
            raise NotImplementedError("Scheduler {} not implemented".format(args['scheduler']))
        return scheduler
    
    def build_model(self, **kwargs):
        raise NotImplementedError

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
        with tqdm(total=self.epochs) as bar:
            for epoch in range(self.epochs):
                train_loss_record = self.train(model, train_loader, optimizer, criterion, scheduler, regularizer=regularizer)
                if self.verbose:
                    # tqdm.write("Epoch {} | {} | lr: {:.4f}".format(epoch, train_loss_record, optimizer.param_groups[0]["lr"]))
                    self.logger("Epoch {} | {} | lr: {:.4f}".format(epoch, train_loss_record, optimizer.param_groups[0]["lr"]))
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
                        # tqdm.write("Epoch {} | save checkpoint in {}".format(epoch, self.saving_path))
                        self.logger("Epoch {} | save checkpoint in {}".format(epoch, self.saving_path))
                    
                if (epoch + 1) % self.eval_freq == 0:
                    valid_loss_record = self.evaluate(model, valid_loader, criterion, split="valid")
                    if self.verbose:
                        # tqdm.write("Epoch {} | {}".format(epoch, valid_loss_record))
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
                            # tqdm.write("Epoch {} | save best models in {}".format(epoch, self.saving_path))
                            self.logger("Epoch {} | save best models in {}".format(epoch, self.saving_path))
                    elif self.patience != -1:
                        counter += 1
                        if counter >= self.patience:
                            if self.verbose:
                                self.logger("Early stop at epoch {}".format(epoch))
                            break
                bar.update(1)

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
            loss_record.update({"train_loss": loss.sum().item()}, n=y_pred.shape[0])

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
                loss_record.update({"{}_loss".format(split): loss.sum().item()}, n=y_pred.shape[0])
        return loss_record


TRAINER_DICT = {
    'base': BaseTrainer,
}


def base_procedure(args):
    if args['model_name'] not in TRAINER_DICT.keys():
        raise NotImplementedError("Model {} not implemented".format(args['model_name']))
    
    if args['verbose']:
        logger = logging.info if args['log'] else print

    if args['wandb']:
        wandb.init(
            project=args['wandb_project'], 
            name=args['saving_name'],
            tags=[args['model'], args['dataset']],
            config=args)
    
    # load data
    if args['verbose']:
        logger("Loading {} dataset, subset is {}".format(args['dataset'], args['subset']))
    start = time()
    dataset = MyDataset(
        data_dir=args['data_dir'],
        train_ratio=args['train_ratio'],
        valid_ratio=args['valid_ratio'],
        test_ratio=args['test_ratio'],
        train_batchsize=args['train_batchsize'],
        eval_batchsize=args['eval_batchsize'],
        subset=args['subset'],
        num_workers=args['num_workers'],
        pin_memory=args['pin_memory'],
    )
    train_loader = dataset.train_loader
    valid_loader = dataset.valid_loader
    test_loader = dataset.test_loader
    if args['verbose']:
        logger("Loading data costs {: .2f}s".format(time() - start))
    
    # build model
    if args['verbose']:
        logger("Building models")
    start = time()
    trainer = TRAINER_DICT[args['model_name']](args)
    model = trainer.build_model(args)
    model = model.to(args['device'])
    optimizer = torch.optim.Adam(
        model.parameters(), 
        betas=(0.9, 0.999),
        lr=args['lr'],
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args['milestones'],
        gamma=args['gamma'],
    )
    criterion = torch.nn.MSELoss()
    
    if args['verbose']:
        logger("Model: {}".format(model))
        logger("Criterion: {}".format(criterion))
        logger("Optimizer: {}".format(optimizer))
        logger("Scheduler: {}".format(scheduler))
        logger("Building models costs {: .2f}s".format(time() - start))

    trainer.process(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
    )
