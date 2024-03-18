import argparse
import os
import numpy as np
import pandas as pd
import h5py
import yaml
import json

import torch
import torch.nn as nn

from utils.utils import get_tabular_data
from utils.core_utils import init_model
from utils.heatmap_utils import *
from datasets.dataset_survival import Generic_WSI_Survival_Dataset


parser = argparse.ArgumentParser()

exp_args = parser.add_argument_group("experiment")
exp_args.add_argument("--save_exp_code", default="HEATMAP_OUTPUT")
exp_args.add_argument("--save_dir", default="heatmaps/")
# exp_args.add_argument("--raw_save_dir", default="heatmaps/heatmap_raw_results")
# exp_args.add_argument("--production_save_dir", default="heatmaps/heatmap_production_results")
exp_args.add_argument("--batch_size", type=int, default=128)

data_args = parser.add_argument_group("data")
data_args.add_argument("--root_dir", default="/media/nfs/SURV/TCGA_OV/")
data_args.add_argument("--process_list", default="tcga_ov_slides.csv")
data_args.add_argument("--feat_ex", default="SSL", choices=["SSL", "CTP", "Res"])
data_args.add_argument("--slide_ext", default=".svs")

patch_args = parser.add_argument_group("patching")
patch_args.add_argument("--patch_size", default=1024, type=int)
patch_args.add_argument("--overlap", default=0.5, type=float)
patch_args.add_argument("--patch_level", default=0, type=int)

model_args = parser.add_argument_group("model")
model_args.add_argument("--load_dir", default="./results_new/tcga_ov/best_run/")
model_args.add_argument("--ckpt_name", default="s_0_checkpoint.pt")

heatmap_args = parser.add_argument_group("heatmaps")
heatmap_args.add_argument("--vis_level", default=1, type=int)
heatmap_args.add_argument("--alpha", default=0.4, type=float)
heatmap_args.add_argument("--blank_canvas", default=False, action="store_true")
heatmap_args.add_argument("--save_orig", default=True, action="store_false")
heatmap_args.add_argument("--save_ext", default="jpg")
heatmap_args.add_argument("--use_ref_scores", default=True, action="store_false")
heatmap_args.add_argument("--blur", default=False, action="store_true")
heatmap_args.add_argument("--use_center_shift", default=True, action="store_false")
heatmap_args.add_argument("--use_roi", default=False, action="store_true")
heatmap_args.add_argument("--calc_heatmap", default=True, action="store_false")
heatmap_args.add_argument("--binarize", default=False, action="store_true")
heatmap_args.add_argument("--binary_thresh", default=-1, type=float)
heatmap_args.add_argument("--cmap", default="jet")

args = parser.parse_args()


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    
    args.raw_save_dir = os.path.join(args.save_dir, "heatmap_raw_results")
    args.production_save_dir = os.path.join(args.save_dir, "heatmap_production_results")

    args.data_dir = os.path.join(args.root_dir, "Slides")
    args.feat_dir = os.path.join(args.root_dir, "Feats1024", args.feat_ex)
    args.h5_dir = os.path.join(args.root_dir, "SP1024", "patches")

    with open(os.path.join(args.load_dir, "experiment.json"), "r") as jf:
        run_config = json.load(jf)
    run_parser = argparse.ArgumentParser()
    for k, v in run_config.items():
        run_parser.add_argument("--" + k, default=v, type=type(v))

    run_args = run_parser.parse_args("")
    run_args.results_dir = args.load_dir

    patch_size = tuple([args.patch_size for i in range(2)])
    step_size = tuple((np.array(patch_size) * (1-args.overlap)).astype(int))
    print('patch_size: {} x {}, with {:.2f} overlap, step size is {} x {}'.format(patch_size[0], patch_size[1], args.overlap, step_size[0], step_size[1]))

    def_seg_params = {
        'seg_level': -1, 'sthresh': 15, 'mthresh': 11, 
        'close': 2, 'use_otsu': False, 
        'keep_ids': 'none', 'exclude_ids':'none'
    }
    def_filter_params = {
        'a_t':50.0, 'a_h': 8.0, 'max_n_holes':10
    }
    def_vis_params = {
        'vis_level': -1, 'line_thickness': 250
    }
    def_patch_params = {
        'use_padding': True, 'contour_fn': 'four_pt'
    }
    if args.process_list is None:
        if isinstance(args.data_dir, list):
            slides = []
            for data_dir in args.data_dir:
                slides.extend(os.listdir(data_dir))
        else:
            slides = sorted(os.listdir(args.data_dir))
        slides = [slide for slide in slides if args.slide_ext in slide]
        df = initialize_df(slides, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False)
        
    else:
        df = pd.read_csv(os.path.join('heatmaps/process_lists', args.process_list))
        df = initialize_df(df, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False)

    mask = df['process'] == 1
    process_stack = df[mask].reset_index(drop=True)
    total = len(process_stack)
    print('\nNb of slides to process: ', len(process_stack))

    tabular_cols = get_tabular_data(run_args) if run_args.tabular_data not in ["None", "none", None] else []
    
    tab_df = pd.read_csv(run_args.csv_path, compression="zip" if ".zip" in run_args.csv_path else None)
    # print(tab_df.shape)
    gen_data = np.unique([i.split("_")[-1] for i in tabular_cols if i.split("_")[-1] in ["rna", "dna", "mut", "cnv", "pro"]])
    if len(gen_data) > 0:
        for g in gen_data:
            gen_df = pd.read_csv(f"./datasets_csv/{os.path.basename(run_args.csv_path).split('.')[0]}_{g}.csv.zip", compression="zip")
            print(gen_df.shape, tab_df.shape)
            tab_df = pd.merge(tab_df, gen_df, on='case_id')
    tab_df = tab_df.reset_index(drop=True)
    
    # assert tab_df.isna().any().any() == False, "There are NaN values in the dataset."
    # print(tab_df.shape)
    process_stack = process_stack[process_stack["slide_id"].isin(tab_df["slide_id"].values)].reset_index(drop=True)
    survival_time_list = tab_df["survival_months"].values
    tab_df = tab_df[tab_df["slide_id"].isin(process_stack["slide_id"].values)].reset_index(drop=True)

    dataset = Generic_WSI_Survival_Dataset(
            df=tab_df,
            mode= run_args.mode,
            n_bins=run_args.n_classes,
            indep_vars=tabular_cols,
            survival_time_list = survival_time_list
        )

    
    print('\ninitializing model from ckpt: {}'.format(os.path.join(args.load_dir, args.ckpt_name)))
    model, _, _, _ =  init_model(run_args, os.path.join(args.load_dir, args.ckpt_name))
    model.eval()
    feature_extractor = init_feat_extractor(args.feat_ex, ctp_ckpt_path="/media/nfs/Desktop/MILSurv/models/ctranspath.pth")
    feature_extractor.eval()

    if torch.cuda.device_count() > 1:
        device_ids = list(range(torch.cuda.device_count()))
        feature_extractor = nn.DataParallel(feature_extractor, device_ids=device_ids).to('cuda:0')
    else:
        feature_extractor = feature_extractor.to(device)

    label_dict = {}
    for k, v in dataset.label_dict.items():
        label_dict[v] = str(round(dataset.time_breaks[k[0]+1]))+"m"+str(k[1])
    class_labels = list(label_dict.keys())
    class_encodings = list(label_dict.values())
    reverse_label_dict = {class_encodings[i]: class_labels[i] for i in range(len(class_labels))} 

    os.makedirs(args.production_save_dir, exist_ok=True)
    os.makedirs(args.raw_save_dir, exist_ok=True)
    blocky_wsi_kwargs = {
        'top_left': None, 'bot_right': None, 
        'patch_size': patch_size, 'step_size': patch_size, 
        'level': args.patch_level, 
        'use_center_shift': args.use_center_shift,
        "target_patch_size": 224
    }
    train_stats = pd.read_csv(os.path.join(run_args.results_dir, f'train_stats_{args.ckpt_name[2]}.csv'))
    train_stats = train_stats.set_index("Unnamed: 0")
    slide_data = dataset.apply_preprocessing(dataset.slide_data, train_stats)
    for i in range(len(process_stack)):
        
        slide_name = process_stack.loc[i, 'slide_id']
        if args.slide_ext not in slide_name:
            slide_name+=args.slide_ext
        print('\nprocessing: ', slide_name)	

        label = slide_data.loc[slide_data['case_id'] == slide_name[:12], 'label'].item()
        time = slide_data.loc[slide_data['case_id'] == slide_name[:12], 'survival_months'].item()
        event = slide_data.loc[slide_data['case_id'] == slide_name[:12], 'event'].item()
        grouping = label_dict[label]
        print('label: ', label, "group: ", grouping, "time: ", time, "event: ", event)
        slide_id = slide_name.replace(args.slide_ext, '')

        p_slide_save_dir = os.path.join(args.production_save_dir, args.save_exp_code, str(grouping))
        os.makedirs(p_slide_save_dir, exist_ok=True)

        r_slide_save_dir = os.path.join(args.raw_save_dir, args.save_exp_code, str(grouping),  slide_id)
        os.makedirs(r_slide_save_dir, exist_ok=True)

        if args.use_roi:
            x1, x2 = process_stack.loc[i, 'x1'], process_stack.loc[i, 'x2']
            y1, y2 = process_stack.loc[i, 'y1'], process_stack.loc[i, 'y2']
            top_left = (int(x1), int(y1))
            bot_right = (int(x2), int(y2))
        else:
            top_left = None
            bot_right = None
        
        print('slide id: ', slide_id)
        print('top left: ', top_left, ' bot right: ', bot_right)

        slide_path = os.path.join(args.data_dir, slide_name)
        
        mask_file = os.path.join(r_slide_save_dir, slide_id+'_mask.pkl')
        
        # Load segmentation and filter parameters
        seg_params = def_seg_params.copy()
        filter_params = def_filter_params.copy()
        vis_params = def_vis_params.copy()

        seg_params = load_params(process_stack.loc[i], seg_params)
        filter_params = load_params(process_stack.loc[i], filter_params)
        vis_params = load_params(process_stack.loc[i], vis_params)

        keep_ids = str(seg_params['keep_ids'])
        if len(keep_ids) > 0 and keep_ids != 'none':
            seg_params['keep_ids'] = np.array(keep_ids.split(',')).astype(int)
        else:
            seg_params['keep_ids'] = []

        exclude_ids = str(seg_params['exclude_ids'])
        if len(exclude_ids) > 0 and exclude_ids != 'none':
            seg_params['exclude_ids'] = np.array(exclude_ids.split(',')).astype(int)
        else:
            seg_params['exclude_ids'] = []

        for key, val in seg_params.items():
            print('{}: {}'.format(key, val))

        for key, val in filter_params.items():
            print('{}: {}'.format(key, val))

        for key, val in vis_params.items():
            print('{}: {}'.format(key, val))

        print('Initializing WSI object')
        wsi_object = initialize_wsi(slide_path, seg_mask_path=mask_file, seg_params=seg_params, filter_params=filter_params)
        print('Done!')

        wsi_ref_downsample = wsi_object.level_downsamples[args.patch_level]

        vis_patch_size = tuple((np.array(patch_size) * np.array(wsi_ref_downsample)).astype(int))

        block_map_save_path = os.path.join(r_slide_save_dir, '{}_blockmap.h5'.format(slide_id))
        mask_path = os.path.join(r_slide_save_dir, '{}_mask.jpg'.format(slide_id))
        if vis_params['vis_level'] < 0:
            vis_params['vis_level'] = wsi_object.wsi.get_best_level_for_downsample(32)
        mask = wsi_object.visWSI(**vis_params, number_contours=True)
        mask.save(mask_path)
        
        features_path = os.path.join(args.feat_dir, slide_id+'.pt')
        h5_path = os.path.join(args.h5_dir, slide_id+'.h5')

        features = torch.load(features_path)
        print("Nb of patches: ", len(features))
        process_stack.loc[i, 'bag_size'] = len(features)
        
        wsi_object.saveSegmentation(mask_file)

        tab_tensor = torch.tensor(slide_data.loc[slide_data['case_id'] == slide_name[:12], tabular_cols].values, dtype=torch.float32) if len(tabular_cols) > 0 else None
        
        A, y_pred, hazards = get_attention_scores(model, features, tab_tensor)
        
        del features
        if not os.path.isfile(block_map_save_path): 
            with h5py.File(h5_path, "r") as hf:
                coords = hf['coords'][:]
            with h5py.File(block_map_save_path, 'w') as hf:
                hf.create_dataset('attention_scores', data=A)
                hf.create_dataset('coords', data=coords)
            
        for c in range(run_args.n_classes):
            process_stack.loc[i, 'Hazards_{}'.format(c)] = hazards[c]
        process_stack.loc[i, 'y_pred'] = y_pred
        process_stack.loc[i, 'y_true'] = grouping

        os.makedirs('{}/results/' .format(args.save_dir), exist_ok=True)
        if args.process_list is not None:
            process_stack.to_csv('{}/results/{}.csv'.format(args.save_dir, args.process_list.replace('.csv', '')), index=False)
        else:
            process_stack.to_csv('{}/results/{}.csv'.format(args.save_dir, exp_args.save_exp_code), index=False)

        with h5py.File(block_map_save_path, 'r') as hf:
            scores = np.array(hf['attention_scores'])
            coords = np.array(hf['coords'])
        
        wsi_kwargs = {
            'top_left': top_left, 'bot_right': bot_right, 
            'patch_size': patch_size, 'step_size': step_size, 
            'level': args.patch_level, 'use_center_shift': args.use_center_shift,
            "target_patch_size": 224}

        heatmap_save_name = '{}_blockmap.tiff'.format(slide_id)
        if os.path.isfile(os.path.join(r_slide_save_dir, heatmap_save_name)):
            print("Passing..")
            pass
        else:
            heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object, cmap=args.cmap, alpha=args.alpha, use_holes=True, binarize=False, vis_level=-1, blank_canvas=False,
                            thresh=-1, patch_size = vis_patch_size, convert_to_percentiles=True)

            heatmap.save(os.path.join(r_slide_save_dir, '{}_blockmap.png'.format(slide_id)))
            del heatmap

        save_path = os.path.join(r_slide_save_dir, '{}_{}_roi_{}.h5'.format(slide_id, args.overlap, args.use_roi))

        if args.use_ref_scores:
            ref_scores = scores
        else:
            ref_scores = None

        if args.calc_heatmap:
            compute_from_patches(wsi_object=wsi_object, tab_features=tab_tensor, model=model, feature_extractor=feature_extractor, batch_size=args.batch_size, **wsi_kwargs, 
                                        attn_save_path=save_path,  ref_scores=ref_scores, fe_type=args.feat_ex)
        
        if not os.path.isfile(save_path):
            print('heatmap {} not found'.format(save_path))
            if heatmap_args.use_roi:
                save_path_full = os.path.join(r_slide_save_dir, '{}_{}_roi_False.h5'.format(slide_id, patch_args.overlap))
                print('found heatmap for whole slide')
                save_path = save_path_full
            else:
                continue
        
        file = h5py.File(save_path, 'r')
        dset = file['attention_scores']
        coord_dset = file['coords']
        scores = dset[:]
        # print("SC", scores.shape)  
        coords = coord_dset[:]
        file.close()

        heatmap_vis_args = {'convert_to_percentiles': True, 'vis_level': args.vis_level, 'blur': args.blur}
        if args.use_ref_scores:
            heatmap_vis_args['convert_to_percentiles'] = False

        heatmap_save_name = '{}_{}_roi_{}_blur_{}_rs_{}_bc_{}_a_{}_l_{}_bi_{}_{}.{}'.format(slide_id, float(args.overlap), int(args.use_roi),
                                                                                        int(args.blur), 
                                                                                        int(args.use_ref_scores), int(args.blank_canvas), 
                                                                                        float(args.alpha), int(args.vis_level), 
                                                                                        int(args.binarize), float(args.binary_thresh), args.save_ext)


        if os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
            pass

        else:    
            # print("SC", scores.shape)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
            heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object,  
                                    cmap=args.cmap, alpha=args.alpha, **heatmap_vis_args, 
                                    binarize=args.binarize, 
                                    blank_canvas=args.blank_canvas,
                                    thresh=args.binary_thresh,  patch_size = vis_patch_size,
                                    overlap=args.overlap, 
                                    top_left=top_left, bot_right = bot_right)
            if args.save_ext == 'jpg':
                heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
            else:
                heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))
        if args.save_orig:
            if args.vis_level >= 0:
                vis_level = args.vis_level
            else:
                vis_level = vis_params['vis_level']
            heatmap_save_name = '{}_orig_{}.{}'.format(slide_id,int(vis_level), args.save_ext)
            if os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
                pass
            else:
                heatmap = wsi_object.visWSI(vis_level=vis_level, view_slide_only=True)
                if args.save_ext == 'jpg':
                    heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
                else:
                    heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))
    config_dict = vars(args)
    with open(os.path.join(args.raw_save_dir, args.save_exp_code, 'config.yaml'), 'w') as outfile:
        yaml.dump(config_dict, outfile, default_flow_style=False)

