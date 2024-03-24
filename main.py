from __future__ import print_function
import argparse

import os
import sys
import json
from timeit import default_timer as timer
import numpy as np
import pandas as pd
import torch
import wandb

### Internal Imports
from datasets.dataset_survival import MIL_Survival_Dataset
from utils.file_utils import save_pkl, load_pkl
from utils.core_utils import train
from utils.utils import get_custom_exp_code, get_tabular_data

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args=None):
	if args is None:
		args = setup_argparse()
	
	args = get_custom_exp_code(args)
		
	print("Experiment Name:", args.run_name)
	seed_torch(args.seed)
	
	split_name = args.split_dir
	args.split_dir = os.path.join('./splits', split_name)
	print("split_dir", args.split_dir)
	assert os.path.isdir(args.split_dir)

	if args.data_root_dir:
		feat_extractor = args.data_root_dir.split('/')[-1] if len(args.data_root_dir.split('/')[-1]) > 0 else args.data_root_dir.split('/')[-2]
		args.path_input_dim = 2048 if feat_extractor == "Res" else 768

	if args.wandb:
		args.k = 1
		args.max_epochs = 20
		args.early_stopping = 10
		wandb.init(config=vars(args))
		config = wandb.config
		for key, value in config.items():
			print(key, value)
			setattr(args, key, value)
		
		args.run_name = wandb.run.name
		args.results_dir = os.path.join(args.results_dir, split_name, "wandb", args.run_name)
	else:
		args.results_dir = os.path.join(args.results_dir, args.param_code, args.run_name + '_s{}'.format(args.seed))

	settings = vars(args)
	print('\nLoad Dataset')
	
	tabular_cols = get_tabular_data(args) if args.tabular_data not in ["None", "none", None] else []
	args.nb_tabular_data = len(tabular_cols)
	
	if args.nb_tabular_data > 0:
		suffix = ""
		if args.data_root_dir not in [None, "None", "none"]:
			suffix += "_"+args.mm_fusion_type+","+args.mm_fusion
		suffix += "_"+args.tabular_data
		args.results_dir += suffix
	
	os.makedirs(args.results_dir, exist_ok=True)
	if ('summary_latest.csv' in os.listdir(args.results_dir)) and (not args.overwrite):
		print("Exp Code <%s> already exists! Exiting script." % args.run_name)
		sys.exit()
	
	if args.csv_path is None:
		args.csv_path = f"{args.csv_dir}/"+split_name+".csv"
	print("Loading all the data ...")
	df = pd.read_csv(args.csv_path, compression="zip" if ".zip" in args.csv_path else None)

	gen_data = np.unique([i.split("_")[-1] for i in tabular_cols if i.split("_")[-1] in ["pro", "rna", "rnz", "dna", "mut", "cnv"]])
	if len(gen_data) > 0:
		for g in gen_data:
			gen_df = pd.read_csv(f"{args.csv_dir}/{split_name}_{g}.csv.zip", compression="zip")
			df = pd.merge(df, gen_df, on='case_id')#, how="outer")
	df = df.reset_index(drop=True).drop(df.index[df["event"].isna()]).reset_index(drop=True)
	# assert df.isna().any().any() == False, "There are NaN values in the dataset."
	print("Successfully loaded.")
	dataset = MIL_Survival_Dataset(
		df=df,
		data_dir= args.data_root_dir,
		mode= args.mode,
		print_info = True,
		n_bins=args.n_classes,
		indep_vars=tabular_cols
	)

	print("Saving to ", args.results_dir)
	with open(args.results_dir + '/experiment.json', 'w') as f:
		json.dump(settings, f, indent=4)

	print("################# Settings ###################")
	for key, val in settings.items():
		print("{}:  {}".format(key, val))  
		

	if args.k_start == -1:
		start = 0
	else:
		start = args.k_start
	if args.k_end == -1:
		end = args.k
	else:
		end = args.k_end

	results = None
	folds = np.arange(start, end)

	### Start 5-Fold CV Evaluation.
	for i in folds:
		start = timer()
		seed_torch(args.seed)
		results_pkl_path = os.path.join(args.results_dir, 'split_latest_test_{}_results.pkl'.format(i))
		if os.path.isfile(results_pkl_path):
			print("Skipping Split %d" % i)
			continue

		### Gets the Train + Val Dataset Loader.
		datasets, train_stats = dataset.return_splits(os.path.join(args.split_dir, f"splits_{i}.csv"))
		train_stats.to_csv(os.path.join(args.results_dir, f'train_stats_{i}.csv'))
		
	
		log, val_latest, test_latest = train(datasets, i, args)
		
		if results is None:
			results = {k: [] for k in log.keys()}
		
		for k in log.keys():
			results[k].append(log[k])
		
		if args.wandb:
			wandb.log(log)

		### Write Results for Each Split to PKL
		if test_latest != None:
			save_pkl(results_pkl_path, test_latest)
		end = timer()
		print('Fold %d Time: %f seconds' % (i, end - start))
	
		pd.DataFrame(results).to_csv(os.path.join(args.results_dir, 'summary_latest.csv'))


def setup_argparse():
	### Training settings
	parser = argparse.ArgumentParser(description='Configurations for Survival Analysis on TCGA Data.')
	parser.add_argument('--run_name',      type=str, default='run')
	parser.add_argument('--csv_path',   type=str, default=None)
	parser.add_argument('--csv_dir', type=str, default="./datasets_csv")
	parser.add_argument('--run_config_file',      type=str, default=None)
	### Checkpoint + Misc. Pathing Parameters
	parser.add_argument('--wandb',		 action='store_true', default=False)
	parser.add_argument('--data_root_dir',   type=str, default=None)
	parser.add_argument('--seed', 			 type=int, default=1, help='Random seed for reproducible experiment (default: 1)')
	parser.add_argument('--k', 			     type=int, default=5, help='Number of folds (default: 5)')
	parser.add_argument('--k_start',		 type=int, default=-1, help='Start fold (Default: -1, last fold)')
	parser.add_argument('--k_end',			 type=int, default=-1, help='End fold (Default: -1, first fold)')
	parser.add_argument('--results_dir',     type=str, default='./results', help='Results directory (Default: ./results)')

	parser.add_argument('--split_dir',       type=str, default="tcga_ov", help='Which cancer type within ./splits/<which_splits> to use for training.')
	parser.add_argument('--log_data',        action='store_true', default=True, help='Log data using tensorboard')
	parser.add_argument('--overwrite',     	 action='store_true', default=False, help='Whether or not to overwrite experiments (if already ran)')

	### Model Parameters.
	parser.add_argument('--model_type',      type=str, default='vit', help='Type of model (Default: mcat)')
	parser.add_argument('--drop_out',        default=.25, type=float, help='Enable dropout (p=0.25)')

	parser.add_argument('--n_classes', type=int, default=4)
	parser.add_argument('--surv_model', default="discrete", choices=["cont", "discrete"])
	parser.add_argument('--tabular_data', default=None)

	parser.add_argument('--mm_fusion',        type=str, choices=["crossatt", "concat", "adaptive", "multiply", "bilinear", "lrbilinear", None], default=None)
	parser.add_argument('--mm_fusion_type',   type=str, choices=["early", "mid", "late", None], default=None)

	parser.add_argument('--target_dim', type=int, default=50)

	parser.add_argument("--mlp_type", default="big", choices=["tiny", "small", "big"])
	parser.add_argument("--activation", default="relu", choices=["relu", "leakyrelu", "gelu"])
	parser.add_argument("--mlp_skip", default=True, action="store_false")
	parser.add_argument("--mlp_depth", default=7, type=int)

	parser.add_argument("--depth", default=5, type=int)
	parser.add_argument("--mha_heads", default=4, type=int)
	parser.add_argument("--model_dim", default=None, help="to decrease nb of patch features")
	parser.add_argument("--mlp_dim", default=64, type=int, help="Hidden dim during FeedForward")
	parser.add_argument("--dim_head", default=32, type=int, help="inner_dim = dim_head * heads")
	parser.add_argument("--pool", default="cls", choices=["cls", "mean"])

	### Optimizer Parameters + Survival Loss Function
	parser.add_argument('--opt',             type=str, choices = ['adam', 'sgd'], default='adam')
	parser.add_argument('--batch_size',      type=int, default=1, help='Batch Size (Default: 1, due to varying bag sizes)')
	parser.add_argument('--gc',              type=int, default=64, help='Gradient Accumulation Step during training (Gradients are calculated for every 256 patients)')
	parser.add_argument('--max_epochs',      type=int, default=30, help='Maximum number of epochs to train')
	
	parser.add_argument('--lr',				 type=float, default=0.001, help='Learning rate')
	parser.add_argument('--train_fraction',      type=float, default=.5, help='fraction of training patches')
	parser.add_argument('--reg', 			 type=float, default=0.01, help='L2-regularization weight decay')
	
	parser.add_argument('--weighted_sample', action='store_true', default=False, help='Enable weighted sampling')
	parser.add_argument('--early_stopping',  default=10, type=int, help='Enable early stopping')
	parser.add_argument('--bootstrapping', action='store_true', default=False)


	args = parser.parse_args()
	return args


### Sets Seed for reproducible experiments.
def seed_torch(seed=7):
	import random
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if device.type == 'cuda':
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

			

if __name__ == "__main__":
	args = setup_argparse()
	if args.run_config_file:
		new_run_name = args.run_name
		results_dir = args.results_dir
		data_root_dir = args.data_root_dir
		wandb = args.wandb
		cv_fold = args.k
		max_epochs = args.max_epochs
		with open(args.run_config_file, "r") as f:
			config = json.load(f)
		
		parser = argparse.ArgumentParser()
		parser.add_argument("--run_config_file")
		for k, v in config.items():
			if k != "run_config_file":
				parser.add_argument('--' + k, default=v, type=type(v))
		args = parser.parse_args()
		args.run_name = new_run_name
		args.data_root_dir = data_root_dir
		args.results_dir = results_dir
		args.wandb = wandb
		args.k = cv_fold
		args.max_epochs = max_epochs
		args.split_dir = args.split_dir.split("/")[-1]
		start = timer()
		results = main(args)
		end = timer()
		print("finished!")
		print("end script")
		print('Script Time: %f seconds' % (end - start))
	elif not args.wandb:
		start = timer()
		results = main()
		end = timer()
		print("finished!")
		print("end script")
		print('Script Time: %f seconds' % (end - start))
	else:
		
		parameter_dict = {
			"opt": {
				# "values": ["sgd", "adam"]
				"value": "adam"
			},
			"lr":{
				"values": [1e-5, 1e-4, 1e-3]
				# "value": 1e-3
			},
			"reg":{
				"values": [1e-2, 1e-4, 1e-3]
			},
			# 'drop_out': {
			# 	"values": [.25, .50, .75]
			# },
			# "gc": {
			# 	"values": [32, 64, 128]
			# },
			
		# }
		# if args.data_root_dir is not None:
		# 	parameter_dict.update({
				'model_dim': {
					# "values": [None, 128, 256]
					'value': None
				},
				'depth': {
					"values": [3, 5]
				},
				'mha_heads': {
					"values": [4, 6]
				},
				'dim_head': {
					"values": [16, 32]
				},
		# 	})
		# else:
		# 	parameter_dict.update({
				"mlp_type": {
					"values": ["tiny", "big"]
				},
				# "activation": {
				# 	"values": ["relu", "leakyrelu", "gelu"]
				# },
				# "mlp_depth": {
				# 	"values": [3, 5, 7]
				# }
			}
		sweep_config = {
			'method': 'random',
			'metric': {
				'name': 'test_cindex',
				'goal': 'maximize'
			},
			'parameters': parameter_dict
		}
		sweep_id = wandb.sweep(sweep_config, project=args.run_name+"_"+args.split_dir) 
		wandb.agent(sweep_id, function=main)
