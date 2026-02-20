# Experiments Directory Structure

## `overseec/`
OverSeeC outputs. Red path on images.
```
overseec/
  <area>/                    # e.g. onion-creek, pickle-north, pickle-south
    <prefix>_mission<N>/     # e.g. oc_mission1, pn_mission2
      plan_on_white.png      # plan on white background (used for HD)
      plan_on_rgb.png        # plan overlaid on satellite image
      costmap.tif
      metadata.txt
```

## `human-baseline/`
Human-drawn trajectories. Green path on TIFFs.
```
human-baseline/
  <area>/
    human/
      <annotator_name>/              # e.g. quattro
        <prefix>_mission<N>.tiff     # e.g. oc_mission1.tiff
```

## `compute_hd.py`
Computes modified Hausdorff distance between OverSeeC and human baselines.
```bash
python experiments/compute_hd.py
python experiments/compute_hd.py --human-names quattro alice --output results.txt
```
