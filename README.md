# Airport digital twin and NOx dispersion prediction

Two deep-learning experiments on Paris-Orly airport, my individual project for the "IA & Application" module
at CY Tech (ING3, March to April 2026, taught by Souhaila Arib; the NOx dataset, `losses.py` and the solvers in
`provided/` were supplied by Abdellahi Babah). Part 1 builds a 3D model of the airport from OpenALAQS/QGIS and
OpenStreetMap layers, fits a neural signed-distance field to it (DeepSDF-style MLP with Fourier features) and
turns the field back into an obstacle mask for a wind solver. Part 2 trains a 3D U-Net to predict the NOx
concentration volume of an airport scenario from wind, geometry and emission fields, on a 3 000-scenario
dataset. Part 1 gave a fragmented reconstruction, which is why the project moved to Part 2
(`docs/changement_d_approche.md`, report and slides in French in `docs/`).

## Airport geometry as a neural SDF

The geometry is an OpenALAQS study of Orly built in QGIS (buildings, runways, taxiways, aprons, gates, roads
through QuickOSM). `src/data_ingestion.py` reads it, extrudes each footprint (OSM height, `building:levels`
x 3.5 m, or 10 m by default) and exports the mesh; `sdf_sampling.py` draws 2.1 M points on and around the
surfaces and computes their signed distance with trimesh. The network maps a normalised (x, y, z) through
8 Fourier frequencies into an 8-layer, 512-unit MLP with a skip connection (1.89 M parameters), trained with a
clamped L1 loss for 200 epochs on a T4 (Adam, lr 2e-4, cosine schedule, batch 16 384, 33 s per epoch). The
checkpoint with the best occupancy IoU on a coarse probe grid (epoch 160) is evaluated on the full
321 x 321 x 65 grid (`results/deepsdf/reconstruction_metrics.json`): the terminals and the large hangars are
there, but small buildings vanish and the field is fragmented.

![OpenALAQS layers of Orly in QGIS](docs/figures/orly_openalaqs_layers.jpg)

| metric | value |
|---|---|
| occupancy IoU | 0.375 |
| precision / recall | 0.788 / 0.417 |
| Chamfer L1 / Hausdorff (approx.) | 123.7 m / 675 m |
| voxel accuracy | 0.995, with 99.3 % of the voxels free space |
| connected components | 2 415 after thresholding; the 42 grounded ones form the `(1, 1, 65, 321, 321)` obstacle mask, run through the wind solver as an integration test only |

![Reference vs prediction slices](results/deepsdf/figures/reconstruction_comparison.png)

## NOx dispersion with a 3D U-Net

The course dataset `dataset_subset_3000.h5` (124 GB, not in this repo) holds 3 000 scenarios of a Lagrangian
dispersion simulation on grids of 140, 300 or 500 columns squared by 64 layers at 5 m x 5 m x 3 m, each with
nine volumes: NOx concentration (target), wind u/v/w, obstacle mask and SDF, GSE and LTO SDFs, total emissions;
422 scenarios (14 %) have no active particle. The split is 2 400 / 300 / 300, stratified on airport category
and stability class, with the zero-particle scenarios removed from the training split only (2 072 left). The
model is a 4-level 3D U-Net (base 32 channels, InstanceNorm, Softplus output, 22.6 M parameters) with 7 input
channels: three wind components, three SDFs, emissions. It trained for 200 epochs on an H100 in bf16 on
64 x 224 x 224 crops (batch 8, AdamW 3e-4 with cosine decay, about 4.5 min per epoch), with the course's
physics-informed composite loss plus two terms I added that weight voxels above 10 and 50 ug/m3 far more than
the background. Crops are centred on the plume 85 % of the time, and the checkpoint is chosen on the plume
RMSE of 16 fixed full-volume validation scenarios rather than on the global loss (`results/nox/figures/`).

| metric (test: best epoch 163, 300 scenarios at full volume, `results/nox/logs/test_summary.json`) | value |
|---|---|
| voxel accuracy on 6 concentration bins | 0.998 |
| weighted F1 / macro F1 | 0.999 / 0.211 |
| F1 per bin (0-10 / 10-50 / 50-100 / 100-200 / 200-500 / >500 ug/m3) | 0.999 / 0.000 / 0.000 / 0.000 / 0.012 / 0.251 |
| median per-scenario RMSE | 69.7 ug/m3 (110 on the 255 scenarios with a plume) |
| median per-scenario plume RMSE (voxels above 10 ug/m3) | 6 344 ug/m3 |

![Prediction vs ground truth slices](results/nox/figures/slices_scenario_00106.png)

The model learned where plumes are and roughly how strong they are (selection plume RMSE 816 k at epoch 1,
134 k at epoch 163). It did not learn the intermediate bins, and it predicts a plume in scenarios that have
none. Voxel accuracy and weighted F1 are uninformative because 99.97 % of the 1.82 billion test voxels sit in
the 0-10 bin; macro F1 (0.21) and the per-scenario medians are the relevant numbers, and they are modest.

## Running it

```bash
pip install -r requirements.txt
python src/data_ingestion.py --config config.yaml
python src/prepare_runtime_config.py --config config.yaml --output config_runtime.yaml
python src/sdf_sampling.py --config config_runtime.yaml
python src/train.py --config config_runtime.yaml
python src/evaluate_reconstruction.py --config config_runtime.yaml --checkpoint checkpoints/best_geometry.pt
python src/generate_mask.py --config config_runtime.yaml --checkpoint checkpoints/best_geometry.pt
```

`config.yaml` expects the OpenALAQS export at `data_sources/airport_study.alaqs` (an empty string means OSM
footprints). Training needs a GPU; `notebooks/deepsdf_airport_colab.ipynb` runs the same steps on Colab. Part 2
is `notebooks/nox_dispersion_unet.ipynb`; it expects `data/dataset_subset_3000.h5` and `provided/losses.py`.

## Limitations

- Mean test errors are unusable: the 45 zero-particle test scenarios get large false plumes (mean RMSE 4.8e10
  there against 4.4e3 elsewhere) because such scenarios were removed from training; per-scenario R2 is around 0.
- The 10-500 ug/m3 bins hold 1 890 to 34 648 voxels in the whole test set, so their per-class F1 is mostly
  noise, and model selection relies on 16 validation scenarios.
- The course suggested a 2 M parameter budget and a mask-versus-SDF comparison; I used 22.6 M and SDF only.
- Part 1 fits one scene with no latent code: the 3 % validation split measures fit, not generalisation.
- The OpenALAQS importer now decodes SpatiaLite geometry blobs (1 107 Orly features); the published mesh and
  metrics predate that fix and come from 672 OSM footprints.
