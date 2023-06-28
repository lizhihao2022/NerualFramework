from tqdm import tqdm
from .metrics import AverageRecord, Metrics
import torch.nn.functional as F


def train(train_loader, model, criterion, optimizer, epoch):
    train_loss = AverageRecord()
    model.cuda()
    model.train()
    with tqdm(total=len(train_loader)) as bar:
        for i, data in enumerate(train_loader):
            X = data[0].to('cuda')
            # compute loss
            y_pred = model(X)
            y_true = data[1]
            y_pred = y_pred.to('cpu')
            loss = criterion(y_pred, y_true)
            # compute gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # record loss and update progress bar
            train_loss.update(loss.item())
            bar.update(1)
            bar.set_postfix_str("train loss: {:.4f}".format(loss.item()))

    return train_loss.avg


def evaluate(eval_loader, model, criterion, split="valid"):
    eval_metric = Metrics(split)
    eval_loss = AverageRecord()
    model.eval()
    
    for data in eval_loader:
        X = data[0].to('cuda')
        # compute loss
        y_pred = model(X)
        y_pred = y_pred.to('cpu')
        y_true = data[1]
        loss = criterion(y_pred, y_true)
        # compute metrics
        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()
        eval_metric.update(y_pred, y_true)
        # record loss and update progress bar
        eval_loss.update(loss.item())
    
    return eval_loss.avg, eval_metric
