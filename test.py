from utils import set_seed, set_device, train, evaluate
from models import MLP
from dataset import MyDataset
from config import parser
import torch


args = parser.parse_args()
args.epochs = 100
args.subset = True

set_device(args.cuda, args.device)
set_seed(args.random_seed)

dataset = MyDataset(args.data_dir, args.train_ratio, args.valid_ratio, args.test_ratio, args.batch_size, args.subset)
train_loader, valid_loader, test_loader = dataset.train_loader, dataset.valid_loader, dataset.test_loader    

model = MLP(args)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

best_epoch = 0
best_metrics = None
counter = 0

for epoch in range(args.epochs):
    train_loss = train(train_loader, model, criterion, optimizer, epoch)
    print("Epoch {} | average train loss: {:.4f} | lr: {:.6f}".format(epoch, train_loss, optimizer.param_groups[0]["lr"]))
    valid_loss, valid_metrics = evaluate(valid_loader, model, criterion, split="valid")
    print("Epoch {} | valid loss: {:.4f} | valid metrics: {}".format(epoch, valid_loss, valid_metrics))
    
print("Optimization Finished!")
