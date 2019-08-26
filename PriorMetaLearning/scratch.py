import sys
sys.path.append("..")
from Models.stochastic_models import get_model
from Utils.common import count_correct, debug, write_to_log, create_result_dir
import torch
from torch import nn
from Data_Path import get_data_path
from torch.utils.data import DataLoader
import os
from Utils.data_gen import Task_Generator
import numpy as np
from opt import prm
from datetime import datetime
prm.log_var_init = {'mean': -10, 'std': 0.1}
data_path = get_data_path()
prm.data_path = get_data_path()

epoch_num = 100
#bz = 64
num_tasks = 20

#set prm params
prm.N_Way = 5
prm.K_Shot_MetaTest = 100
prm.data_source = "CIFAR100"
prm.model_name = "CIFARNet"

learning_rate = 1e-3
loss_fn = nn.CrossEntropyLoss()

prm.run_name = "scratch_cifar100_lr_{}_{}_way_{}_shot_nte_{}_epoch_{}".format(learning_rate, prm.N_Way, prm.K_Shot_MetaTest, num_tasks, epoch_num)
prm.result_dir = os.path.join("scratch_log", prm.run_name)
os.makedirs(prm.result_dir,  exist_ok=True)
#create_result_dir(prm)
time_str = datetime.now().strftime(' %Y-%m-%d %H:%M:%S')
write_to_log("Written by scratch.py at {}".format(time_str), prm, mode = "w")

task_generator = Task_Generator(prm)
data_loaders = task_generator.create_meta_batch(prm, num_tasks, meta_split='meta_test')
assert len(data_loaders) == num_tasks

best_accs = []
best_losses = []

for i_task in range(num_tasks):
    #initialize net at start
    net = get_model(prm)
    net = net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    d = data_loaders[i_task]
    train_loader = d['train']
    test_loader = d['test']
    n_train = d['n_train_samples']
    n_test = d['n_test_samples']
    best_acc = -1
    best_loss = 1000
    assert (n_test + n_train) == 3000
    #debug()

    #write_to_log("train_num", n_train, "test_num", n_test), prm)
    for epoch in range(epoch_num):
        cnt = 0
        for imgs, ys in train_loader:
            cnt += 1
            #if cnt % 100 == 0:
                #print(cnt)
            imgs = imgs.cuda()
            ys = ys.cuda()
            output = net(imgs)
            optimizer.zero_grad()

            correct_cnt = count_correct(output, ys)
            #debug()
            acc = correct_cnt / ys.size(0)

            loss = loss_fn(output, ys)
            write_to_log("Train : Task {}, Epoch {}/{}, loss : {}, acc : {}".format(i_task, epoch, epoch_num, loss, acc), prm)
            loss.backward()
            optimizer.step()

        total_cnt = 0
        cor_cnt = 0
        total_loss = 0.0
        loss_cnt = 0
        for imgs, ys in test_loader:
            imgs = imgs.cuda()
            ys = ys.cuda()
            output = net(imgs)

            correct_cnt = count_correct(output, ys)
            cor_cnt += correct_cnt
            total_cnt += ys.size(0)
            #acc = correct_cnt / len(ys)

            loss = loss_fn(output, ys)
            total_loss += loss
            loss_cnt += 1
            #print("loss : {}".format(loss))
        total_loss = total_loss / loss_cnt




        test_acc = cor_cnt / total_cnt
        write_to_log("Test : Task {} , Epoch {}/{}, Test Acc: {}".format(i_task, epoch, epoch_num,test_acc), prm)
        if test_acc > best_acc:
            best_acc = test_acc
            best_loss = loss
        #save_path = os.path.join(save_dir, "epoch-{}-acc{:.3f}.pth".format(epoch, test_acc))

    write_to_log("Task {} Best Loss : {}, Acc : {}".format(i_task, best_loss, best_acc), prm)
    #float is necessary
    #otherwise torch will keep track of this var
    best_losses.append(float(best_loss))
    best_accs.append(best_acc)



#import pudb
#pudb.set_trace()
print(best_losses)
print(best_accs)
best_accs = np.array(best_accs)
best_losses = np.array(best_losses)
avg_acc = np.mean(best_accs)
avg_loss = np.mean(best_losses)
print("Avg Loss : {}, Acc : {}".format(avg_loss, avg_acc))
print("Std Loss : {}, Acc : {}".format(best_losses.std(), best_accs.std()))



