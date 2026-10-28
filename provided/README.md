# Provided material

These files were supplied by the course and are kept here unchanged so that the
notebooks and `src/generate_mask.py` can import them. They are not my code.

- `wind.py`, `turbulence.py`, `solver.py`: diagnostic wind field (AUSTAL / TALdia
  style) and Lagrangian dispersion solver of the kind used to generate the NOx
  dataset. `generate_mask.py` produces the obstacle mask expected by `wind.py`.
- `losses.py`: physics-informed loss terms (boundary-weighted MSE, mass conservation,
  solid-voxel penalty, total variation) used in `notebooks/nox_dispersion_unet.ipynb`.
- `documentation_dataset.md`, `dataset_subset_3000_resume.json`: description and
  summary of `dataset_subset_3000.h5`. The HDF5 file itself is 124 GB and is not in
  the repository.
