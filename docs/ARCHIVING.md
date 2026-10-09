# Archiving release v1.0.0 on Zenodo

This release packages the UFO SegFormer B2 training/evaluation code and the
verified archived checkpoint and prediction artifacts. The source code trains
a four-channel model. The recovered checkpoint accepts three channels; its
original preprocessing and linkage to the archived masks remain unconfirmed.
See [PRETRAINED.md](PRETRAINED.md) and [EVALUATION.md](EVALUATION.md).

## Release files

- `urbanFloodObservations-v1.0.0-source.zip`: versioned source, documentation,
  evaluation CSV and verification summary.
- `urbanFloodObservations-v1.0.0-zenodo.zip`: complete upload bundle, containing
  the same source plus the original checkpoint, converted Transformers weights,
  215 prediction masks, evaluation CSV, verification reports, model card,
  per-file checksums and release provenance.
- `SHA256SUMS`: SHA-256 checksums of both ZIP files and the metadata template.
- `zenodo-metadata.json`: a deposit metadata template preserving the existing
  Tellman Lab creator credit and linking the UFO dataset and Hugging Face model.
  The template leaves the software/model license unspecified.

## Manual upload

1. Download the complete `-zenodo.zip` asset and `SHA256SUMS` from
   [the GitHub release](https://github.com/Tellman-lab/urbanFloodObservations/releases/tag/v1.0.0).
2. Create a Zenodo upload with resource type Software. Upload the complete ZIP
   and checksum file. Use the metadata template for the title, description,
   version, creators and related identifiers. No dataset imagery or reference
   labels are included; the existing dataset DOI is a related record.
3. Select the code and model licenses before publishing. The repository and
   supplied checkpoint do not declare them. The dataset's CC BY 4.0 license
   does not determine the software or checkpoint license. Zenodo defaults to
   CC BY 4.0; review that choice rather than treating it as an existing grant.
4. Publish the record to obtain a software DOI, then add it to `CITATION.cff`
   and the repository README. The release does not claim an assigned software
   DOI or a completed Zenodo deposit.

For automatic archiving, connect and enable this repository in Zenodo and
resolve the license metadata before archiving a release. Automatic GitHub
archiving preserves repository source; use the full bundle when the checkpoint
and prediction files should also be preserved in the deposit.

References: [Zenodo GitHub documentation](https://help.zenodo.org/docs/github/),
[Zenodo licenses](https://help.zenodo.org/docs/deposit/describe-records/licenses/).
