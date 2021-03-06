from subprocess import call, run
import argparse
import os
import sys
sys.path.append("/home/tlgao/meta/meta-learning-adjusting-priors")

n_train_tasks = 5

parser = argparse.ArgumentParser()

parser.add_argument('--complexity_type', type=str,
                    help=" The learning objective complexity type",
                    default='NewBoundSeeger')
# 'NoComplexity' /  'Variational_Bayes' / 'PAC_Bayes_Pentina'   NewBoundMcAllaster / NewBoundSeeger'"

args = parser.parse_args()
complexity_type = args.complexity_type
call(['python', 'main_Meta_Bayes.py',
      '--run-name', 'PermutedLabels_{}_Tasks_{}_Comp'.format(n_train_tasks, complexity_type),
      '--data-source', 'Omniglot',
      '--data-transform', 'None',
      '--batch-size', '16',
      '--limit_train_samples_in_test_tasks', '2000',
      '--n_train_tasks',  str(n_train_tasks),
      '--mode', 'MetaTrain',
      '--complexity_type',  complexity_type,
      '--model-name', 'ConvNet3',
      '--n_meta_train_epochs', '100',
      '--n_meta_test_epochs', '200',
      '--n_test_tasks', '5',
      '--n_train_tasks', '50',
      '--K_Shot_MetaTest', '15',
      '--K_Shot_MetaTrain', '15',
      '--N_Way', '5',
      '--meta_batch_size', '16',
      '--task_complex_w', '0.0001',
      '--meta_complex_w', '0.1',
      ])
