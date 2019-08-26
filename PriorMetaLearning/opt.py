import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--run-name', type=str, help='Name of dir to save results in (if empty, name by time)',
                    default='')

parser.add_argument('--seed', type=int,  help='random seed',
                    default=1)

parser.add_argument('--mode', type=str, help='MetaTrain or LoadMetaModel',
                    default='MetaTrain')   # 'MetaTrain'  \ 'LoadMetaModel'

parser.add_argument('--load_model_path', type=str, help='set the path to pre-trained model, in case it is loaded (if empty - set according to run_name)',
                    default='')

parser.add_argument('--test-batch-size',type=int,  help='input batch size for testing (reduce if memory is limited)',
                    default=128)

parser.add_argument('--n_test_tasks', type=int,
                    help='Number of meta-test tasks for meta-evaluation (how many tasks to average in final result)',
                    default=10)

# ----- Task Parameters ---------------------------------------------#

parser.add_argument('--data-source', type=str, help="Data: 'MNIST' / 'CIFAR10' / Omniglot / SmallImageNet",
                    default='MNIST')

parser.add_argument('--n_train_tasks', type=int, help='Number of meta-training tasks (0 = infinite)',
                    default=5)

parser.add_argument('--data-transform', type=str, help="Data transformation:  'None' / 'Permute_Pixels' / 'Permute_Labels'/ Shuffled_Pixels ",
                    default='None')

parser.add_argument('--n_pixels_shuffles', type=int, help='In case of "Shuffled_Pixels": how many pixels swaps',
                    default=200)

parser.add_argument('--limit_train_samples_in_test_tasks', type=int,
                    help='Upper limit for the number of training samples in the meta-test tasks (0 = unlimited)',
                    default=2000)

# N-Way K-Shot Parameters:
parser.add_argument('--N_Way', type=int, help='Number of classes in a task (for Omniglot)',
                    default=5)

parser.add_argument('--K_Shot_MetaTrain', type=int,
                    help='Number of training sample per class in meta-training in N-Way K-Shot data sets',
                    default=100)  # Note:  test samples are the rest of the data

parser.add_argument('--K_Shot_MetaTest', type=int,
                    help='Number of training sample per class in meta-testing in N-Way K-Shot data sets',
                    default=100)  # Note:  test samples are the rest of the data

# SmallImageNet Parameters:
parser.add_argument('--n_meta_train_classes', type=int,
                    help='For SmallImageNet: how many categories are available for meta-training',
                    default=500)

# Omniglot Parameters:
parser.add_argument('--chars_split_type', type=str,
                    help='how to split the Omniglot characters  - "random" / "predefined_split"',
                    default='random')

parser.add_argument('--n_meta_train_chars'
                    , type=int, help='For Omniglot: how many characters to use for meta-training, if split type is random',
                    default=1200)

# ----- Algorithm Parameters ---------------------------------------------#

parser.add_argument('--complexity_type', type=str,
                    help=" The learning objective complexity type",
                    default='NewBoundSeeger')  #  'NoComplexity' /  'Variational_Bayes' / 'PAC_Bayes_Pentina'   NewBoundMcAllaster / NewBoundSeeger'"

# parser.add_argument('--override_eps_std', type=float,
#                     help='For debug: set the STD of epsilon variable for re-parametrization trick (default=1.0)',
#                     default=1.0)

parser.add_argument('--loss-type', type=str, help="Loss function",
                    default='CrossEntropy') #  'CrossEntropy' / 'L2_SVM'

parser.add_argument('--model-name', type=str, help="Define model type (hypothesis class)'",
                    default='ConvNet3')  # OmConvNet / 'FcNet3' / 'ConvNet3'

parser.add_argument('--batch-size', type=int, help='input batch size for training',
                    default=128)

parser.add_argument('--n_meta_train_epochs', type=int, help='number of epochs to train',
                    default=150)

parser.add_argument('--n_inner_steps', type=int,
                    help='For infinite tasks case, number of steps for training per meta-batch of tasks',
                    default=50)  #

parser.add_argument('--n_meta_test_epochs', type=int, help='number of epochs to train',
                    default=200)  #

parser.add_argument('--lr', type=float, help='initial learning rate',
                    default=1e-3)

parser.add_argument('--meta_batch_size', type=int, help='Maximal number of tasks in each meta-batch',
                    default=5)
# -------------------------------------------------------------------------------------------

parser.add_argument('--task_complex_w', type=float, help='weight of task complexity term',
                    default=1.0)
parser.add_argument('--meta_complex_w', type=float, help='weight of meta complexity term',
                    default=1.0)
parser.add_argument('--from_pretrain', type=int, help='whether load pretrain conv part',
                    default=None)

prm = parser.parse_args()
