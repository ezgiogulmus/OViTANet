import os
import math
from itertools import islice, chain
import collections
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
from models.mil_model import MIL_fc_mc
from models.vit2d import ViT
from models.mlp_model import MLP
from models.model_gmcat import GMCAT
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data(args):
	df = pd.read_csv(args.csv_path, compression="zip" if ".zip" in args.csv_path else None)
	indep_vars = []
	args.nb_tabular_data = 0
	if args.omics not in ["None", "none", None]:
		print("Selected omics variables:")
		if args.selected_features:
			omics_cols = {k: [col for col in df.columns if col[-3:]==k] for k in args.omics.split(",")}
			indep_vars = list(chain(*omics_cols.values()))
			for k, v in omics_cols.items():
				print("\t", k, len(v))
		else:
			remove_cols = {k: [col for col in df.columns if col[-3:]==k] for k in ["cli", "cnv", "rna", "pro", "mut", "dna"]}
			if "cli" in args.omics:
				cli_cols = remove_cols.pop("cli")
				print("\tcli", len(cli_cols))
				indep_vars.extend(cli_cols)
			df = df[[i for i in df.columns if i not in list(chain(*remove_cols.values()))]]
			print(df.shape)
			for g in args.omics.split(","):
				if g != "cli":
					gen_df = pd.read_csv(f"{args.dataset_dir}/{args.data_name}_{g}.csv.zip", compression="zip")
					indep_vars.extend(gen_df.columns[1:])
					print("\t", g, gen_df.shape[1]-1)
					df = pd.merge(df, gen_df, on='case_id', how="outer")
			df = df.reset_index(drop=True).drop(df.index[df["event"].isna()]).reset_index(drop=True)
		args.nb_tabular_data = len(indep_vars)
		if args.separate_branches:
			args.nb_tabular_data = []
			gen_types = np.unique([i[-3:] for i in indep_vars])
			for g in gen_types:
				args.nb_tabular_data.append(len([col for col in indep_vars if col[-3:]==g]))
	
	print("Total number of cases: {} | slides: {}" .format(len(df["case_id"].unique()), len(df)))
	return df, indep_vars


class SubsetSequentialSampler(Sampler):
	"""Samples elements sequentially from a given list of indices, without replacement.

	Arguments:
		indices (sequence): a sequence of indices
	"""
	def __init__(self, indices):
		self.indices = indices

	def __iter__(self):
		return iter(self.indices)

	def __len__(self):
		return len(self.indices)
	
def collate_MIL_separate(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)	
	label = torch.cat([item[1] for item in batch], dim = 0).type(torch.LongTensor)
	event_time = torch.FloatTensor([item[2] for item in batch])
	c = torch.FloatTensor([item[3] for item in batch])
	tabular = [torch.cat([item[4][i] for item in batch], dim=0) for i in range(len(batch[0][4]))]
	case_id = np.array([item[5] for item in batch])
	return [img, label, event_time, c, tabular, case_id]

def collate_MIL_survival(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)	
	label = torch.cat([item[1] for item in batch], dim = 0).type(torch.LongTensor)
	event_time = torch.FloatTensor([item[2] for item in batch])
	c = torch.FloatTensor([item[3] for item in batch])
	tabular = torch.cat([item[4] for item in batch], dim = 0).type(torch.FloatTensor)
	case_id = np.array([item[5] for item in batch])
	
	return [img, label, event_time, c, tabular, case_id]

def collate_MIL(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	label = torch.LongTensor(np.array([item[1] for item in batch]))
	ids = torch.LongTensor(np.array([item[2] for item in batch]))
	return [img, label, ids]

def get_simple_loader(dataset, batch_size=1):
	kwargs = {'num_workers': 4} if device.type == "cuda" else {}
	loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler.SequentialSampler(dataset), collate_fn = collate_MIL, **kwargs)
	return loader 

def get_split_loader(split_dataset, training = False, weighted = False, batch_size=1, separate_branches=False):
	"""
		return either the validation loader or training loader 
	"""
	
	collate = collate_MIL_separate if separate_branches else collate_MIL_survival

	kwargs = {'num_workers': 4} if device.type == "cuda" else {}
	
	if training:
		if weighted:
			weights = make_weights_for_balanced_classes_split(split_dataset)
			loader = DataLoader(split_dataset, batch_size=batch_size, sampler = WeightedRandomSampler(weights, len(weights)), drop_last = True if batch_size > 1 else False, collate_fn = collate, **kwargs)    
		else:
			loader = DataLoader(split_dataset, batch_size=batch_size, sampler = RandomSampler(split_dataset), drop_last = True if batch_size > 1 else False, collate_fn = collate, **kwargs)
	else:
		loader = DataLoader(split_dataset, batch_size=batch_size, sampler = SequentialSampler(split_dataset), collate_fn = collate, **kwargs)

	return loader

def model_builder(args, ckpt_path=None, print_model=False):
	model_dict = {
	"n_classes": args.n_classes if args.surv_model == "discrete" else 1,
	"drop_out": args.drop_out,
	"batch_norm": True if args.batch_size > 1 else False,
	"mlp_type": args.mlp_type,
	"mlp_skip": args.mlp_skip,
	"activation": args.activation,
	"nb_tabular_data": args.nb_tabular_data,
	"mm_fusion": args.fusion,
	"mm_fusion_type": args.fusion_location,
	"separate_branches": args.separate_branches,
	"nb_of_omics": len(args.omics.split(",")),
	"path_input_dim": args.path_input_dim,
	"depth": args.depth, 
	"mha_heads": args.mha_heads,
	"dim_head": args.dim_head,
	}
	print(f"Initiating {args.model_type.upper()} model...")
	
	if args.model_type == "mlp":
		model = MLP(**model_dict)
	elif args.model_type == "mil":
		model = MIL_fc_mc(**model_dict)
	elif args.model_type == "vit":
		model = ViT(**model_dict)
	elif args.model_type == "gmcat":
		model = GMCAT(**model_dict)
	else:
		raise NotImplementedError
	
	if print_model:
		num_params = 0
		num_params_train = 0
		print(model)
		
		for param in model.parameters():
			n = param.numel()
			num_params += n
			if param.requires_grad:
				num_params_train += n
		
		print('Total number of parameters: %d' % num_params)
		print('Total number of trainable parameters: %d' % num_params_train)
	
	if ckpt_path is not None:
		model.load_state_dict(torch.load(ckpt_path))
	model = model.to(device)
	return model


def generate_split(cls_ids, val_num, test_num, samples, n_splits = 5,
	seed = 7, label_frac = 1.0, custom_test_ids = None):
	indices = np.arange(samples).astype(int)
	
	if custom_test_ids is not None:
		indices = np.setdiff1d(indices, custom_test_ids)

	np.random.seed(seed)
	for i in range(n_splits):
		all_val_ids = []
		all_test_ids = []
		sampled_train_ids = []
		
		if custom_test_ids is not None: # pre-built test split, do not need to sample
			all_test_ids.extend(custom_test_ids)

		for c in range(len(val_num)):
			possible_indices = np.intersect1d(cls_ids[c], indices) #all indices of this class
			remaining_ids = possible_indices

			if val_num[c] > 0:
				val_ids = np.random.choice(possible_indices, val_num[c], replace = False) # validation ids
				remaining_ids = np.setdiff1d(possible_indices, val_ids) #indices of this class left after validation
				all_val_ids.extend(val_ids)

			if custom_test_ids is None and test_num[c] > 0: # sample test split

				test_ids = np.random.choice(remaining_ids, test_num[c], replace = False)
				remaining_ids = np.setdiff1d(remaining_ids, test_ids)
				all_test_ids.extend(test_ids)

			if label_frac == 1:
				sampled_train_ids.extend(remaining_ids)
			
			else:
				sample_num  = math.ceil(len(remaining_ids) * label_frac)
				slice_ids = np.arange(sample_num)
				sampled_train_ids.extend(remaining_ids[slice_ids])

		yield sorted(sampled_train_ids), sorted(all_val_ids), sorted(all_test_ids)



def make_weights_for_balanced_classes_split(dataset):
	N = float(len(dataset))                                           
	weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
	weight = [0] * int(N)                                           
	for idx in range(len(dataset)):   
		y = dataset.getlabel(idx)                        
		weight[idx] = weight_per_class[y]                                  

	return torch.DoubleTensor(weight)

def initialize_weights(module):
	for m in module.modules():
		if isinstance(m, nn.Linear):
			nn.init.xavier_normal_(m.weight)
			m.bias.data.zero_()
		
		elif isinstance(m, nn.BatchNorm1d):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)


def dfs_freeze(model):
	for name, child in model.named_children():
		for param in child.parameters():
			param.requires_grad = False
		dfs_freeze(child)


def dfs_unfreeze(model):
	for name, child in model.named_children():
		for param in child.parameters():
			param.requires_grad = True
		dfs_unfreeze(child)

def nth(iterator, n, default=None):
	if n is None:
		return collections.deque(iterator, maxlen=0)
	else:
		return next(islice(iterator,n, None), default)

def save_splits(split_datasets, column_keys, filename, boolean_style=False):
	splits = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]
	# [split_datasets[i].slide_data.to_csv(f"/home/ezgitwo/Desktop/sil{i}.csv") for i in range(len(split_datasets))]
	if not boolean_style:
		df = pd.concat(splits, ignore_index=True, axis=1)
		df.columns = column_keys
	else:
		df = pd.concat(splits, ignore_index = True, axis=0)
		index = df.values.tolist()
		one_hot = np.eye(len(split_datasets)).astype(bool)
		bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
		df = pd.DataFrame(bool_array, index=index, columns = ['train', 'val', 'test'])

	df.to_csv(filename)

def check_directories(args):
	r"""
	Updates the argparse.NameSpace with a custom experiment code.

	Args:
		- args (NameSpace)

	Returns:
		- args (NameSpace)
	"""
	
	feat_extractor = None
	if args.feats_dir:
		feat_extractor = args.feats_dir.split('/')[-1] if len(args.feats_dir.split('/')[-1]) > 0 else args.feats_dir.split('/')[-2]
		if feat_extractor == "RESNET50":
			args.path_input_dim = 2048 
		elif feat_extractor in ["PLIP", "CONCH"]:
			args.path_input_dim = 512 
		elif feat_extractor == "UNI":
			args.path_input_dim = 1024
		else:
			args.path_input_dim = 768

	args.split_dir = os.path.join('./splits', args.data_name)
	print("split_dir", args.split_dir)
	assert os.path.isdir(args.split_dir)

	param_code = args.model_type.upper()
	inputs = []
	if feat_extractor:
		param_code += "_" + feat_extractor
		inputs.append("path")
	if args.omics:
		inputs.append("tab")
	
	args.mode = ("+").join(inputs)
	if args.mode != "path+tab":
		args.fusion, args.fusion_location = None, None
	
	param_code += '_' + args.mode

	if args.omics not in ["None", "none", None]:
		suffix = ""
		if args.feats_dir not in [None, "None", "none"]:
			suffix += "_"+args.fusion_location+","+args.fusion
		suffix += "_"+args.omics
		if not args.selected_features:
			suffix += "_all"
		args.run_name += suffix
	
	args.results_dir = os.path.join(args.results_dir, param_code, args.run_name)
	args.csv_path = f"{args.dataset_dir}/"+args.data_name+".csv" if not args.selected_features else f"{args.dataset_dir}/"+args.data_name+"_selected.csv"
	print("Loading the data from ", args.csv_path)
	assert os.path.isfile(args.csv_path), f"Data file does not exist > {args.csv_path}"
	return args
