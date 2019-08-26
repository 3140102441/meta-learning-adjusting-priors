import sys
sys.path.append("..")
from Models.stochastic_models import get_model
from Utils.cifar100 import load_pretrain_dataset
from Utils.common import count_correct, debug, load_model_state
import torch
from torch import nn
from Data_Path import get_data_path
from torch.utils.data import DataLoader
import os
from opt import prm

prm.log_var_init = {'mean': -10, 'std': 0.1}

#data_path = get_data_path()
prm.N_Way = 64
prm.data_source = "CIFAR100"
prm.model_name = "CIFARNet"
epoch_num = 100
bz = 512
init_from_ae = True

train_dataset, test_dataset, info = load_pretrain_dataset()
train_loader = DataLoader(train_dataset, batch_size=bz, shuffle=True, num_workers = 12, pin_memory = True)
print(len(train_dataset), len(test_dataset))
test_loader = DataLoader(test_dataset, batch_size=bz, shuffle=True, num_workers = 12, pin_memory = True)
save_dir = "pretrained_cifar100"
os.makedirs(save_dir,  exist_ok=True)

net = get_model(prm)

#load_model_state(net, save_dir + "/" + "epoch-2-acc0.277.pth")


#debug()
print(net)
net = net.cuda()


loss_fn = nn.CrossEntropyLoss()

learning_rate = 1e-3
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
for epoch in range(epoch_num):
    cnt = 0
    for imgs, ys in train_loader:
        cnt += 1
        #if cnt % 100 == 0:
            #print(cnt)
        imgs = imgs.cuda()
        #print(imgs.size())
        ys = ys.cuda()
        output = net(imgs)
        optimizer.zero_grad()

        correct_cnt = count_correct(output, ys)
        #debug()
        acc = correct_cnt / ys.size(0)

        loss = loss_fn(output, ys)
        #print("loss : {}, acc : {}".format(loss, acc))
        loss.backward()
        optimizer.step()

    total_cnt = 0
    cor_cnt = 0
    for imgs, ys in test_loader:
        imgs = imgs.cuda()
        ys = ys.cuda()
        output = net(imgs)

        correct_cnt = count_correct(output, ys)
        cor_cnt += correct_cnt
        total_cnt += ys.size(0)
        #acc = correct_cnt / len(ys)

        loss = loss_fn(output, ys)
        #print("loss : {}".format(loss))

    test_acc = cor_cnt / total_cnt
    print("Epoch {}/{}, Test Acc: {}".format(epoch, epoch_num,test_acc))
    save_path = os.path.join(save_dir, "epoch-{}-acc{:.3f}.pth".format(epoch, test_acc))
    torch.save(net.state_dict(), save_path)
