import torch
import time
import numpy as np
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score
from torch import nn
from Utils.Avg_ import AverageMeter
def train_on_batch(num_epochs, train_iter, valid_iter, lr, net, device):
    xent_losses = AverageMeter()
    losses = AverageMeter()
    criterion_xent = nn.CrossEntropyLoss()
    trainer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999),weight_decay=0.0003)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer, T_max=num_epochs * len(train_iter), eta_min=5e-6)
    true_label=[]
    pred_label=[]
    for epoch in range(num_epochs):
        net.train()
        sum_loss = 0.0
        sum_acc = 0.0
        for (X, y) in train_iter:
            X = X.type(torch.FloatTensor)
            y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64)
            X = X.to(device)
            y = y.to(device)
            labels= net(X)
            #y_hat= net(X)#提取的特征
            loss_xent = criterion_xent(labels, y)
            loss = loss_xent
            trainer.zero_grad()
            loss.backward()
            trainer.step()
            losses.update(loss.item(), labels.size(0))
            xent_losses.update(loss_xent.item(), labels.size(0))

            sum_loss +=  loss.item() / y.shape[0]
            sum_acc += (y == labels.argmax(dim=-1)).float().mean()
        train_loss = sum_loss / len(train_iter)
        train_acc = sum_acc / len(train_iter)
        if epoch == num_epochs - 1:
            net.eval()
            sum_acc = 0.0
            for (X, y) in valid_iter:
                X = X.type(torch.FloatTensor)
                y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64)
                X = X.to(device)
                y_hat= net(X)
                sum_acc +=(y == y_hat.argmax(dim=-1)).float().mean()
                true_label+=y
                pred_label+=y_hat.argmax(dim=-1)
            val_acc = sum_acc / len(valid_iter)
        print(f"epoch{epoch + 1}, train_loss={train_loss:.3f}, train_acc={train_acc:.3f}")
    print(f'training finished at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} with final_valid_acc={val_acc:.3f}')
    kappa = cohen_kappa_score(true_label,pred_label)
    recall = recall_score(true_label,pred_label, average="macro")
    presion = precision_score(true_label,pred_label, average="macro")
    print('kappa系数为：{}'.format(kappa))
    print("查准率：{}".format(presion))
    print("查全率：{}".format(recall))
    return val_acc,kappa,recall,presion
