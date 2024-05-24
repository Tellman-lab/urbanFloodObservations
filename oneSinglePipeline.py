import os
from pathlib import Path
import rasterio as rio
from rasterio.merge import merge
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch
from fastai.vision.all import *

# Base path definition
base_path = Path("/media/mule/Projects/NASA/THP/Data/experimentsDataPaper/PSonlyDatasetTHPv1013/PSDataset/trainingDatasetUFO/trainingDatasetUFO1105/trainingDatasetUFO0129_256/")

# Dynamic component definitions
model_identifier = 'images0207_localNorm_256_GroupedByEvents_resnet50'

# Derived paths using base path and dynamic components
input_base_path = base_path / 'images0207_localNorm_256_GroupedByEvents'
labels_base_path = base_path / 'labels0129_256_GroupedByEvents'
log_file = base_path / f'log_{model_identifier}'
model_load_path = input_base_path / 'models'
output_folder = base_path / f'{model_identifier}_Preds'
output_base = base_path / f'{model_identifier}_Preds_ProcessedMosaics'
pred_folder = output_base
val_folder = Path("/media/mule/Projects/NASA/THP/Data/experimentsDataPaper/PSonlyDatasetTHPv1013/PSDataset/trainingDatasetUFO/trainingDatasetUFO1105/trainingDatasetUFO0129/labels0129_GroupedByEvents")
csv_output_path = val_folder / f'{model_identifier}.csv'

# Hyperparameters definition remains unchanged
hyperparameters = {'epochs': 30, 'lr': 1e-3, 'bs': 4}

#access epochs


def open_geotiff(fn, chans=None):
    with rio.open(str(fn)) as f:
        data = f.read().astype(np.float32)  # Read and convert data to float32
    im = torch.from_numpy(data).float()  # Convert to float tensor
    if chans is not None: 
        im = im[chans]  # Select specific channels if specified
    return im

class MultiChannelTensorImage(TensorImage):
    @classmethod
    def create(cls, fn, chans=None, **kwargs):
        if str(fn).endswith('.tif'): 
            return cls(open_geotiff(fn=fn, chans=chans))

    def __repr__(self):
        return f'{self.__class__.__name__} size={"x".join([str(d) for d in self.shape])}'
    
class TifSegmentationDataLoaders(DataLoaders):
    @classmethod
    @delegates(DataLoaders.from_dblock)
    def from_label_funcs(cls, path, fnames, label_func, chans=None, extensions=['.tif'], valid_pct=None, seed=None, codes=None, item_tfms=None, batch_tfms=None, **kwargs):
        dblock = DataBlock(blocks=(MultiChannelImageBlock(chans=chans), MaskBlock(codes=codes)),
                           splitter=RandomSplitter(valid_pct, seed=seed),
                           get_y=label_func,
                           item_tfms=item_tfms,
                           batch_tfms=batch_tfms)
        res = cls.from_dblock(dblock, fnames, path=path, **kwargs)
        return res

def MultiChannelImageBlock(cls=MultiChannelTensorImage, chans=None):
    return TransformBlock(partial(cls.create, chans=chans))

class CombinedLoss:
    def __init__(self, axis=1, smooth=1., alpha=1.):
        store_attr()
        self.focal_loss = FocalLossFlat(axis=axis)
        self.dice_loss = DiceLoss(axis, smooth)

    def __call__(self, pred, targ):
        return self.focal_loss(pred, targ) + self.alpha * self.dice_loss(pred, targ)

    def decodes(self, x):    return x.argmax(dim=self.axis)
    def activation(self, x): return F.softmax(x, dim=self.axis)

def get_mask_from_tif(fn):
    return open_geotiff(fn, chans=[0])[0]

def save_georeferenced_pred(pred, reference_fp, output_fp):
    with rio.open(reference_fp) as ref:
        profile = ref.profile

    profile.update(dtype=rio.uint8, count=1)

    pred_data = pred.argmax(dim=0).numpy().astype(rio.uint8)  # Get the most likely class per pixel

    with rio.open(output_fp, 'w', **profile) as dst:
        dst.write(pred_data, 1)

# Training and Inference
def run_training_and_inference():
    subfolders = [f for f in input_base_path.iterdir() if f.is_dir()]
    codes = ['other', 'water']

    for leave_out in subfolders:
        print(f"Leaving out: {leave_out.name}")
        included_folders = [sf for sf in subfolders if sf != leave_out]
        
        fnames = []
        labels = []
        
        for folder in included_folders:
            images_path = input_base_path/folder.name
            labels_path = labels_base_path/folder.name
            fnames += get_files(images_path, extensions=['.tif'], recurse=False)
            labels += [labels_path/(f.name.replace("images", "labels")) for f in fnames]
        
        def label_func(fname, labels_base_path, leave_out_name):
            prefix = fname.name.split('_')[0]
            subfolder = prefix[:3]  # Extract the first three letters of the file name
            file_path = labels_base_path/subfolder/(fname.name.replace("images", "labels"))
    
            
            if not file_path.exists():
                print(f"File not found: {file_path}")
                # Handle the case when the file is not found
                # You can return a default value or raise an appropriate exception
                return None  # Return None as a placeholder
        
            return file_path

        batch_tfms = [Rotate(), Flip(), Dihedral(), Normalize()]

        label_func_with_path = partial(label_func, labels_base_path=labels_base_path, leave_out_name=leave_out.name)
        segm = TifSegmentationDataLoaders.from_label_funcs(
            path=input_base_path,
            fnames=fnames,
            label_func=label_func_with_path,
            valid_pct=0.05,
            seed=42,
            bs=hyperparameters['bs'],
            codes=codes,
            batch_tfms=batch_tfms
        )

        learn = unet_learner(segm, resnet50, metrics=[JaccardCoeff(), Dice()], loss_func=CombinedLoss(), n_in=4, opt_func=ranger, act_cls=Mish, pretrained=False)
        learn.to_fp16()

        
        learn.fit_flat_cos(hyperparameters['epochs'], slice(hyperparameters['lr']))

        learn.save(f'modelUFO_{model_identifier}_{leave_out.name}', with_opt=False)

        with open(log_file, 'a') as log:
            val_res = learn.validate()
            log.write(f'Left out {leave_out.name}, Validation Results: {val_res}\n')

        del segm, learn
        torch.cuda.empty_cache()

    # Inference
    output_folder.mkdir(parents=True, exist_ok=True)  # Ensure the output folder exists
    
    batch_tfms = [Rotate(), Flip(), Dihedral(), Normalize()]

    segm = TifSegmentationDataLoaders.from_label_funcs(
        path=input_base_path,
        fnames = get_files(input_base_path/'GIL', extensions=['.tif'], recurse=False),
        label_func = lambda o: get_mask_from_tif(f'{labels_base_path}/GIL/{o.stem}{o.suffix}'),
        valid_pct=0.0,
        codes=codes,
        batch_tfms=batch_tfms
    )

    learn = unet_learner(segm, resnet50, metrics=[JaccardCoeff(), Dice()], loss_func=CombinedLoss(), n_in=4, opt_func=ranger, act_cls=Mish)

    for leave_out in input_base_path.iterdir():
        if not leave_out.is_dir() or leave_out.name == 'models':
            continue
        print(f"Processing: {leave_out.name}")
        
        model_name = f'modelUFO_{model_identifier}_{leave_out.name}'
        learn.load(model_load_path / model_name)
        learn.to_fp16()

        inferSet = [fn for fn in sorted(leave_out.glob('**/*.tif')) if fn.is_file()]

        test_dl = learn.dls.test_dl(inferSet, bs=16)

        preds = learn.get_preds(dl=test_dl)

        sub_output_folder = output_folder / leave_out.name
        sub_output_folder.mkdir(parents=True, exist_ok=True)

        for i, pred in enumerate(preds[0]):
            image_name = inferSet[i].name
            output_fp = sub_output_folder / image_name
            reference_fp = inferSet[i]
            
            save_georeferenced_pred(pred, reference_fp, output_fp)

        print(f"Predictions saved in: {sub_output_folder}")

    del segm, learn, test_dl
    torch.cuda.empty_cache()

# Mosaicking
def mosaic_folder(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)



    # List all files in the current directory
    files = [f for f in input_folder.iterdir() if f.is_file() and f.suffix == '.tif']
    file_groups = defaultdict(list)

    # print(f"Processing {len(files)} files in {input_folder}")

    # Group files by their prefix
    for file_path in files:
        # Assuming the first two parts of the filename before the underscores represent the group
        prefix = '_'.join(file_path.stem.split('_')[:2])
        file_groups[prefix].append(file_path)

    # Process each group of files
    for prefix, file_group in file_groups.items():

        src_files_to_mosaic = [rio.open(fp) for fp in file_group]
        # print(src_files_to_mosaic[0])
        mosaic, out_trans = rio.merge.merge(src_files_to_mosaic)
        out_meta = src_files_to_mosaic[0].meta.copy()

        # Update metadata for the mosaic
        out_meta.update({"driver": "GTiff",
                         "height": mosaic.shape[1],
                         "width": mosaic.shape[2],
                         "transform": out_trans})

        # Write the mosaic to disk
        with rio.open(output_folder / f"{prefix}.tif", "w", **out_meta) as dest:
            dest.write(mosaic)

        # Close the files
        for src in src_files_to_mosaic:
            src.close()

def process_subfolders(folder, output_base):
    for subfolder in folder.iterdir():
        if subfolder.is_dir():
            output_folder = output_base / subfolder.relative_to(folder)
            # print(f"Processing: {output_folder}, {subfolder}")
            mosaic_folder(subfolder, output_folder)

# Evaluation
def calculate_metrics(pred, true):
    pred_flat = pred.flatten()
    true_flat = true.flatten()

    tn, fp, fn, tp = confusion_matrix(true_flat, pred_flat, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Also known as recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    iou = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    return tp, fp, tn, fn, precision, sensitivity, specificity, f1, iou, accuracy

def evaluate_results(pred_folder, val_folder):
    metrics = []
    
    for subfolder in os.listdir(pred_folder):
        pred_subfolder_path = os.path.join(pred_folder, subfolder)
        val_subfolder_path = os.path.join(val_folder, subfolder)
    
        if os.path.isdir(pred_subfolder_path) and os.path.isdir(val_subfolder_path):  # Check if both are directories
            for pred_file in os.listdir(pred_subfolder_path):
                pred_file_path = os.path.join(pred_subfolder_path, pred_file)
                val_file_path = os.path.join(val_subfolder_path, pred_file)
    
                if os.path.isfile(pred_file_path) and os.path.isfile(val_file_path):  # Check if both are files
                    with rio.open(pred_file_path) as pred_src, rio.open(val_file_path) as val_src:
                        pred_img = pred_src.read(1)
                        val_img = val_src.read(1)
    
                        tp, fp, tn, fn, precision, sensitivity, specificity, f1, iou, accuracy = calculate_metrics(pred_img, val_img)
    
                        metrics.append({
                            'file': pred_file,
                            'TP': tp,
                            'FP': fp,
                            'TN': tn,
                            'FN': fn,
                            'Precision': precision,
                            'Sensitivity': sensitivity,
                            'Specificity': specificity,
                            'F1': f1,
                            'IoU': iou,
                            'Accuracy': accuracy
                        })
    
    df_metrics = pd.DataFrame(metrics)
    
    if not df_metrics.empty:  # Check if the DataFrame is not empty
        mean_scores = df_metrics.loc[:, df_metrics.columns != 'file'].mean().to_dict()
        mean_scores['file'] = 'overall_mean'
        df_metrics = df_metrics.append(mean_scores, ignore_index=True)
    
    # csv_output_path = os.path.join(val_folder, 'evaluation_results.csv')
    df_metrics.to_csv(csv_output_path, index=False)
    print(f"Evaluation results saved in: {csv_output_path}")

# Main function
def main():
    # Run training and inference
    run_training_and_inference()
    
    # Run mosaicking
    process_subfolders(output_folder, output_base)
    
    # Run evaluation
    evaluate_results(pred_folder, val_folder)

if __name__ == '__main__':
    main()


