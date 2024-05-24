import torch
from fastai.vision.all import *
from pathlib import Path
from PIL import Image
import rasterio as rio

# Function to open GeoTIFF files as tensors
def open_geotiff_as_tensor(fn, chans=None):
    with rio.open(str(fn)) as f:
        im = torch.from_numpy(f.read().astype(np.float32)).float()  # Read and convert data to float32
    return im[chans] if chans is not None else im

# Custom class for handling multi-channel images
class MultiChannelTensorImage(TensorImage):
    @classmethod
    def create(cls, fn, chans=None, **kwargs):
        return cls(open_geotiff_as_tensor(fn=fn, chans=chans))

    def __repr__(self):
        return f'{self.__class__.__name__} size={"x".join([str(d) for d in self.shape])}'

# Function for creating a TransformBlock for multi-channel images
def MultiChannelImageBlock(cls=MultiChannelTensorImage, chans=None):
    return TransformBlock(partial(cls.create, chans=chans))

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


# Combined loss function for segmentation
class CombinedLoss:
    def __init__(self, axis=1, smooth=1., alpha=1.):
        store_attr()
        self.focal_loss = FocalLossFlat(axis=axis)
        self.dice_loss = DiceLoss(axis, smooth)

    def __call__(self, pred, targ):
        return self.focal_loss(pred, targ) + self.alpha * self.dice_loss(pred, targ)

    def decodes(self, x):    return x.argmax(dim=self.axis)
    def activation(self, x): return F.softmax(x, dim=self.axis)

# Run inference function
def run_inference():
    # Define paths and constants
    paths = {
        "input": Path("/media/mule/Projects/NASA/THP/Data/experimentsDataPaper/PSonlyDatasetTHPv1013/PSDataset/trainingDatasetUFO/trainingDatasetUFO1105/trainingDatasetUFO0129_256/images0207_localNorm_256_GroupedByEvents/GIL"),
        "model": Path("/media/mule/Projects/NASA/THP/Data/experimentsDataPaper/PSonlyDatasetTHPv1013/PSDataset/trainingDatasetUFO/trainingDatasetUFO1105/trainingDatasetUFO0129_256/images0207_localNorm_256_GroupedByEvents_models"),
        "output": Path("/media/mule/Projects/NASA/THP/Data/experimentsDataPaper/PSonlyDatasetTHPv1013/PSDataset/trainingDatasetUFO/trainingDatasetUFO1105/trainingDatasetUFO0129_256/images0207_localNorm_256_GroupedByEvents_Preds"),
        "labels": Path('/media/mule/Projects/NASA/THP/Data/experimentsDataPaper/PSonlyDatasetTHPv1013/PSDataset/trainingDatasetUFO/trainingDatasetUFO1105/trainingDatasetUFO0129_256/labels0129_256_GroupedByEvents/GIL')
    }
 
    codes = ['other', 'water']
    batch_tfms = [Rotate(), Flip(), Dihedral(), Normalize()]

    # Initialize DataLoader
    segm = TifSegmentationDataLoaders.from_label_funcs(
        path=paths['input'],
        fnames=get_files(paths['input'], extensions=['.tif'], recurse=False),
        label_func=lambda o: paths['labels']/f'{o.stem}{o.suffix}',
        valid_pct=0.0,
        codes=codes,
        batch_tfms=batch_tfms
    )

    # Initialize the learner
    learn = unet_learner(segm, resnet34, metrics=[JaccardCoeff(), Dice()], loss_func=CombinedLoss(), n_in=4, opt_func=ranger, act_cls=Mish)

    # Inference loop
    for leave_out in paths['input'].iterdir():
        if not leave_out.is_dir() or leave_out.name == 'models':
            continue
        print(f"Processing: {leave_out.name}")
        learn.load(paths['model'] / f'model_{leave_out.name}')
        learn.to_fp16()

        inferSet = [fn for fn in sorted(leave_out.glob('**/*.tif')) if fn.is_file()]
        test_dl = learn.dls.test_dl(inferSet, bs=16)
        preds = learn.get_preds(dl=test_dl)

        # Save predictions
        sub_output_folder = paths['output'] / leave_out.name
        sub_output_folder.mkdir(parents=True, exist_ok=True)
        for i, pred in enumerate(preds[0]):
            pred_arg = pred.argmax(dim=0).numpy().astype(np.uint8)
            Image.fromarray(pred_arg).save(sub_output_folder / inferSet[i].name)
        print(f"Predictions saved in: {sub_output_folder}")

    del segm, learn, test_dl  # Cleanup
    torch.cuda.empty_cache()

if __name__ == '__main__':
    run_inference()
