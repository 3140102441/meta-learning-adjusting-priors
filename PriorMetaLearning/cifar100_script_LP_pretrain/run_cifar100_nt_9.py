from subprocess import call, run
import argparse
import os
import sys

os.chdir("..")

sys.path.append("/home/tlgao/meta/meta-learning-adjusting-priors")



n_train_tasks = 9
n_test_tasks = 10
task_complex_w = 0.01
meta_complex_w = 0.01

parser = argparse.ArgumentParser()

parser.add_argument('--complexity_type', type=str,
                    help=" The learning objective complexity type",
                    default='PAC_Bayes_Pentina')
# 'NoComplexity' /  'Variational_Bayes' / 'PAC_Bayes_Pentina'   NewBoundMcAllaster / NewBoundSeeger'"
pretrain = '1'

args = parser.parse_args()
complexity_type = args.complexity_type
call(['python', 'main_Meta_Bayes.py',
      #'--run-name', 'CIFAR100_{}_tasks_split_comp_{}'.format(n_train_tasks, complexity_type),
      '--run-name', 'CIFAR100__comp_{}_ntr_{}_nte_{}_task-w_{}_meta-w_{}_pre-{}'.format( complexity_type, n_train_tasks, n_test_tasks, task_complex_w, meta_complex_w, str(pretrain)),
      '--data-source', 'CIFAR100',
      '--data-transform', 'None',
      '--batch-size', '16',
      '--limit_train_samples_in_test_tasks', '2000',
      '--mode', 'MetaTrain',
      '--complexity_type',  complexity_type,
      '--model-name', 'CIFARNet',
      '--n_meta_train_epochs', '50',
      '--n_meta_test_epochs', '40',
      '--n_test_tasks', str(n_test_tasks),
      '--n_train_tasks', str(n_train_tasks),
      '--K_Shot_MetaTest', '50',
      '--K_Shot_MetaTrain', '50',
      '--N_Way', '5',
      '--meta_batch_size', '16',
      '--task_complex_w', str(task_complex_w),
      #'--task_complex_w', '0.0',
      '--meta_complex_w', str(meta_complex_w),
      '--from_pretrain', pretrain,
      ])
