# Archived SegFormer B2 checkpoint

The recovered `segformer_b2_best.pt` is a SegFormer B2 state dictionary with
three input channels and two output classes. It is separate from this
repository's four-band PlanetScope training pipeline.

## Access

- [Hugging Face model package](https://huggingface.co/rohitm9/ufo-segformer-b2)
  (public download; no sign-in required).
- [Original checkpoint on Drive](https://drive.google.com/file/d/15I7KA-ddi1MhqSYW_t4XSmPzImwGdYBT/view).

The Hugging Face package contains the original checkpoint, a Transformers
`config.json` and `model.safetensors`, a model card, and the verified prediction
ZIP and evaluation CSV. The original checkpoint's SHA-256 is:

```text
9dbdb73298694593bf41e8e0c7aba665e5924336ab94ce215e164185e4cc990f
```

## Architecture and conversion

| Property | Verified value |
| --- | --- |
| Encoder depths | `[3, 4, 6, 3]` |
| Decoder hidden dimension | 768 |
| Input channels | 3 |
| First convolution weight shape | `[64, 3, 7, 7]` |
| Output classes | 2 |
| Parameters | 27,348,162 |

The conversion removes the leading `model.` wrapper prefix and loads every
tensor strictly. No weights were retrained or changed. All tensors and a small
synthetic CPU forward pass were bitwise identical after saving and reloading
the safetensors checkpoint. The configuration uses the verified tensor shapes
and the [NVIDIA B2 reference](https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512/blob/main/config.json).
The original training configuration was not supplied, so non-tensor settings
use that reference.

## Input preparation

The checkpoint's original band selection, band order, scaling and normalization
are not yet confirmed. Three input channels do not establish that the inputs
were RGB. No image processor or assumed normalization is included.

Load the public checkpoint using Transformers:

```python
from transformers import AutoModelForSemanticSegmentation

model = AutoModelForSemanticSegmentation.from_pretrained(
    "rohitm9/ufo-segformer-b2"
).eval()
```

Inputs must be prepared float tensors shaped `[batch, 3, height, width]` using
the original preprocessing. The model returns logits at reduced resolution;
upsample them to the input grid before taking the class argmax. Confirm the
preprocessing before running inference on new imagery.

## Evaluation provenance

[The archived B2 prediction ZIP and CSV](EVALUATION.md) were checked against all
215 local UFO image/label pairs. This verifies those prediction artifacts and
their scores. It does not establish that this individual checkpoint generated
the masks or identify its held-out event. Original fold checkpoint identities
and training/test separation are not included in the supplied artifacts.

The supplied checkpoint does not specify a model-weights license. The UFO
dataset's CC BY 4.0 license does not establish the model-weights license.
