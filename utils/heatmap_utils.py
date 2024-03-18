import pandas as pd
import numpy as np
import torch
import h5py
import math
from scipy.stats import percentileofscore
from utils.wsi_utils import WholeSlideImage
from utils.utils import get_simple_loader
# from utils.file_utils import save_hdf5
from datasets.wsi_dataset import Wsi_Region
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
initiate a pandas df describing a list of slides to process
args:
	slides (df or array-like): 
		array-like structure containing list of slide ids, if df, these ids assumed to be
		stored under the 'slide_id' column
	seg_params (dict): segmentation paramters 
	filter_params (dict): filter parameters
	vis_params (dict): visualization paramters
	patch_params (dict): patching paramters
	use_heatmap_args (bool): whether to include heatmap arguments such as ROI coordinates
'''
def initialize_df(slides, seg_params, filter_params, vis_params, patch_params, 
	use_heatmap_args=False, save_patches=False):

	total = len(slides)
	if isinstance(slides, pd.DataFrame):
		slide_ids = slides.slide_id.values
	else:
		slide_ids = slides
	default_df_dict = {'slide_id': slide_ids, 'process': np.full((total), 1, dtype=np.uint8)}

	# initiate empty labels in case not provided
	if use_heatmap_args:
		default_df_dict.update({'label': np.full((total), -1)})
	
	default_df_dict.update({
		'status': np.full((total), 'tbp'),
		# seg params
		'seg_level': np.full((total), int(seg_params['seg_level']), dtype=np.int8),
		'sthresh': np.full((total), int(seg_params['sthresh']), dtype=np.uint8),
		'mthresh': np.full((total), int(seg_params['mthresh']), dtype=np.uint8),
		'close': np.full((total), int(seg_params['close']), dtype=np.uint32),
		'use_otsu': np.full((total), bool(seg_params['use_otsu']), dtype=bool),
		'keep_ids': np.full((total), seg_params['keep_ids']),
		'exclude_ids': np.full((total), seg_params['exclude_ids']),
		
		# filter params
		'a_t': np.full((total), int(filter_params['a_t']), dtype=np.float32),
		'a_h': np.full((total), int(filter_params['a_h']), dtype=np.float32),
		'max_n_holes': np.full((total), int(filter_params['max_n_holes']), dtype=np.uint32),

		# vis params
		'vis_level': np.full((total), int(vis_params['vis_level']), dtype=np.int8),
		'line_thickness': np.full((total), int(vis_params['line_thickness']), dtype=np.uint32),

		# patching params
		'use_padding': np.full((total), bool(patch_params['use_padding']), dtype=bool),
		'contour_fn': np.full((total), patch_params['contour_fn'])
		})

	if save_patches:
		default_df_dict.update({
			'white_thresh': np.full((total), int(patch_params['white_thresh']), dtype=np.uint8),
			'black_thresh': np.full((total), int(patch_params['black_thresh']), dtype=np.uint8)})

	if use_heatmap_args:
		# initiate empty x,y coordinates in case not provided
		default_df_dict.update({'x1': np.empty((total)).fill(np.NaN), 
			'x2': np.empty((total)).fill(np.NaN), 
			'y1': np.empty((total)).fill(np.NaN), 
			'y2': np.empty((total)).fill(np.NaN)})


	if isinstance(slides, pd.DataFrame):
		temp_copy = pd.DataFrame(default_df_dict) # temporary dataframe w/ default params
		# find key in provided df
		# if exist, fill empty fields w/ default values, else, insert the default values as a new column
		for key in default_df_dict.keys(): 
			if key in slides.columns:
				mask = slides[key].isna()
				slides.loc[mask, key] = temp_copy.loc[mask, key]
			else:
				slides.insert(len(slides.columns), key, default_df_dict[key])
	else:
		slides = pd.DataFrame(default_df_dict)
	
	return slides


def init_feat_extractor(feat_ex, ctp_ckpt_path=None):
	if feat_ex == "Res":
		from torchvision import models
		backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
		backbone.fc = torch.nn.Identity()
	
	elif feat_ex == "SSL":
		from transformers import ViTModel
		backbone = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
		backbone.head = torch.nn.Identity()

	elif feat_ex == "CTP":
		from models.ctran import ctranspath
		backbone = ctranspath()
		backbone.head = torch.nn.Identity()
		backbone.load_state_dict(torch.load(ctp_ckpt_path)["model"], strict=True)
	backbone.eval()
	return backbone

def load_params(df_entry, params):
	for key in params.keys():
		if key in df_entry.index:
			dtype = type(params[key])
			val = df_entry[key] 
			val = dtype(val)
			if isinstance(val, str):
				if len(val) > 0:
					params[key] = val
			elif not np.isnan(val):
				params[key] = val
	return params

def initialize_wsi(wsi_path, seg_mask_path=None, seg_params=None, filter_params=None):
	wsi_object = WholeSlideImage(wsi_path)
	target_downsample = None
	if seg_params['seg_level'] < 0:
		seg_params['seg_level'] = wsi_object.wsi.get_best_level_for_downsample(32)

	wsi_object.segmentTissue(**seg_params, filter_params=filter_params, target_downsample=target_downsample)
	wsi_object.saveSegmentation(seg_mask_path)
	return wsi_object

def score2percentile(score, ref):
	percentile = percentileofscore(ref, score)
	return percentile

def compute_from_patches(wsi_object, tab_features, model=None, feature_extractor=None, batch_size=512,  
	attn_save_path=None, ref_scores=None, fe_type="SSL", **wsi_kwargs):    
	top_left = wsi_kwargs['top_left']
	bot_right = wsi_kwargs['bot_right']
	patch_size = wsi_kwargs['patch_size']
	
	roi_dataset = Wsi_Region(wsi_object, **wsi_kwargs)
	roi_loader = get_simple_loader(roi_dataset, batch_size=batch_size)
	
	print('total number of patches to process: ', len(roi_dataset))
	num_batches = len(roi_loader)
	print('number of batches: ', len(roi_loader))
	A_list, coords_list = [], []

	for idx, (roi, coords, _) in enumerate(roi_loader):
		roi = roi.to(device)
		tab_features = tab_features.to(device) if tab_features != None else None
		coords = coords.numpy()
		
		with torch.no_grad():
			features = feature_extractor(roi)
			if fe_type == "SSL":
				features = features.last_hidden_state[:, 0, :]

			
			_, _, A = model(x_path=features, x_tabular=tab_features, return_feats=True)
			A_scores = torch.mean(A[0], dim=0).cpu().numpy()
			# A_scores: num_patches+1, num_patches+2
			A_scores = A_scores[0] # cls token
			# print(A_scores.shape)
			if tab_features is not None:
				A_scores_patches = A_scores[:-1]
				A_scores_tab = A_scores[-1]
			else:
				A_scores_patches = A_scores
				A_scores_tab = None
			A_scores_patches /= A_scores_patches.max()
			if ref_scores is not None:
				for score_idx in range(len(A_scores_patches)):
					A_scores_patches[score_idx] = score2percentile(A_scores_patches[score_idx], np.squeeze(ref_scores))
			A_list.extend(A_scores_patches)
			coords_list.extend(coords)
			
		if idx % math.ceil(num_batches * 0.05) == 0:
			print('procssed {} / {}'.format(idx, num_batches))
	
	with h5py.File(attn_save_path, "w") as f:
		f.create_dataset('coords', data=np.array(coords_list))
		f.create_dataset('attention_scores', data=np.array(A_list))
	return attn_save_path, wsi_object

def get_attention_scores(model, img_features, tab_features):
	img_features = img_features.to(device)
	tab_features = tab_features.to(device) if tab_features != None else None
	with torch.no_grad():
		logits, _, A = model(x_path=img_features, x_tabular=tab_features, return_feats=True)
		
		# A.shape = (1, heads, num_patches+1, num_patches+1) no tab data
		# A.shape = (1, heads, num_patches+1, num_patches+2) with tab data
		A_scores = torch.mean(A[0], dim=0).cpu().numpy()
		# A_scores: num_patches+1, num_patches+2
		# print("CLS mean/min/max:", np.mean(A_scores[0]), A_scores[0].min(), A_scores[0].max())
		A_scores = A_scores[0] # cls token
		# print(A_scores.shape)
		if tab_features is not None:
			A_scores_patches = A_scores[:-1]
			A_scores_tab = A_scores[-1]
		else:
			A_scores_patches = A_scores
			A_scores_tab = None

		A_scores_patches /= A_scores_patches.max()
		# print(A_scores_patches.shape, A_scores_patches.max(), A_scores_patches.min())
		hazards = torch.sigmoid(logits)
		S = torch.cumprod(1 - hazards, dim=1)
		
		indices = (S < 0.5).nonzero()
		y_pred = indices[0][-1].item() if len(indices) > 0 else len(S)
	return A_scores_patches, y_pred, hazards[0].cpu().numpy()

def drawHeatmap(scores, coords, slide_path=None, wsi_object=None, vis_level = -1, **kwargs):
	if wsi_object is None:
		wsi_object = WholeSlideImage(slide_path)
		print(wsi_object.name)
	
	wsi = wsi_object.wsi
	if vis_level < 0:
		vis_level = wsi.get_best_level_for_downsample(32)
	
	heatmap = wsi_object.visHeatmap(scores=scores, coords=coords, vis_level=vis_level, **kwargs)
	return heatmap