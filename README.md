# Airport digital twin and NOx dispersion prediction

Two deep-learning experiments on Paris-Orly airport. The first fits a DeepSDF-style neural signed-distance field to a 3D model of the airport built from OpenALAQS, QGIS and OpenStreetMap layers. The second trains a 3D U-Net to predict the NOx concentration volume of an airport scenario from wind, geometry and emission fields.

![OpenALAQS layers of Orly in QGIS](docs/figures/orly_openalaqs_layers.jpg)

## What it does

- Extrudes the buildings, runways, taxiways, aprons, gates and roads of an OpenALAQS study into a mesh, samples 2.1 M signed distances around it and fits a Fourier-feature MLP of 1.89 M parameters.
- Turns the field back into a `(1, 1, 65, 321, 321)` obstacle mask, 42 grounded components out of 2 415, tested in the course wind solver.
- Predicts NOx volumes with a 4-level 3D U-Net of 22.6 M parameters from three wind components, three SDFs and the emissions.
- Trains on 3 000 Lagrangian dispersion scenarios of 140 to 500 columns squared by 64 layers, 422 of them (14%) without any active particle.

## How it works

```
.alaqs + OSM -> src/data_ingestion.py -> src/sdf_sampling.py -> src/train.py: clamped L1, 200 epochs on a T4
  -> src/evaluate_reconstruction.py, src/generate_mask.py on the epoch-160 checkpoint
dataset_subset_3000.h5 (124 GB) -> notebooks/nox_dispersion_unet.ipynb: split 2 400 / 300 / 300, zero-particle scenarios dropped from training only (2 072 left)
  -> 200 epochs on an H100 in bf16 on 64 x 224 x 224 crops, course physics-informed loss + two plume-weighting terms
  -> checkpoint on the plume RMSE of 16 fixed validation volumes
```

## Results

Geometry on the full 321 x 321 x 65 grid, `results/deepsdf/reconstruction_metrics.json`. NOx on the 300 test scenarios at full volume, best epoch 163, `results/nox/logs/test_summary.json`.

| part | metric | value |
|---|---|---|
| geometry | occupancy IoU, precision / recall | 0.375, 0.788 / 0.417 |
| geometry | Chamfer L1 / Hausdorff (approx.) | 123.7 m / 675 m |
| geometry | voxel accuracy | 0.995, with 99.3% of voxels free |
| NOx | voxel accuracy on 6 bins, weighted F1 / macro F1 | 0.998, 0.999 / 0.211 |
| NOx | F1 per bin, 0-10 / 10-50 / 50-100 / 100-200 / 200-500 / >500 ug/m3 | 0.999 / 0.000 / 0.000 / 0.000 / 0.012 / 0.251 |
| NOx | median per-scenario RMSE, all / 255 with a plume / plume voxels | 69.7 / 110 / 6 344 ug/m3 |

The SDF keeps the terminals and hangars but drops the small buildings, and the U-Net places the plumes but misses the intermediate bins. These are the numbers of the recorded runs, and the macro F1 stays low. 99.97% of the 1.82 billion test voxels sit in the 0-10 bin, which makes voxel accuracy and weighted F1 meaningless.

![Reference vs prediction slices](results/deepsdf/figures/reconstruction_comparison.png)
![Prediction vs ground truth slices](results/nox/figures/slices_scenario_00106.png)

## Run it

Python with `requirements.txt`, QGIS for the OpenALAQS export, a GPU for training.

```bash
pip install -r requirements.txt
python src/data_ingestion.py --config config.yaml
python src/prepare_runtime_config.py --config config.yaml --output config_runtime.yaml
python src/sdf_sampling.py --config config_runtime.yaml
python src/train.py --config config_runtime.yaml
python src/evaluate_reconstruction.py --config config_runtime.yaml --checkpoint checkpoints/best_geometry.pt
```

`config.yaml` expects the export at `data_sources/airport_study.alaqs`, or OSM footprints on an empty string. `notebooks/deepsdf_airport_colab.ipynb` runs the same steps on Colab. The NOx notebook needs `data/dataset_subset_3000.h5`, course data not in this repo, and `provided/losses.py`.

## Limitations

- Mean test errors are unusable and per-scenario R2 is around 0. The 45 zero-particle test scenarios, absent from training, get false plumes with a mean RMSE of 4.8e10 against 4.4e3 elsewhere.
- The 10-500 ug/m3 bins hold 1 890 to 34 648 test voxels, so their F1 is mostly noise, and model selection rests on 16 validation scenarios. The course suggested a 2 M parameter budget and a mask-versus-SDF comparison. I used 22.6 M and SDF only.
- The SDF fits one scene without latent code, so its 3% validation split measures fit, not generalisation. The published mesh and metrics come from 672 OSM footprints, before the importer learned to decode SpatiaLite blobs (1 107 Orly features).

Individual project for the IA & Application module at CY Tech, ING3, March to April 2026, taught by Souhaila Arib. Abdellahi Babah supplied the NOx dataset, `losses.py` and the solvers in `provided/`, and I wrote the rest. Report, slides and `docs/changement_d_approche.md`, on why the work moved to the dispersion part, are in French in `docs/`.
