# Image Editing Test Environment

This repo now includes a lightweight path for testing the mask and costmap pixel editor without GPUs, `vllm`, checkpoints, or the segmentation pipeline.

## Local venv

```bash
./scripts/setup_image_edit_test_env.sh
./scripts/run_image_edit_test_server.sh
```

The setup script creates `.venv-image-edit`, installs the minimal dependencies, and generates synthetic fixture data.

Open `http://127.0.0.1:5002` and use:

- `Browse` in the TIFF section, then select `edit_test_scene`
- `Browse` in the Masks section, then open `edit_test_scene/mask/temp_latest`
- `Browse` in the Costmap section, then open `edit_test_scene/costmap/temp_latest`

## Docker fallback

```bash
docker build -f Dockerfile.image-edit-test -t overseec-image-edit-test .
docker run --rm -it -p 5002:5002 -v "$PWD:/workspace/overseec" overseec-image-edit-test bash
```

Inside the container:

```bash
python scripts/create_image_edit_test_fixtures.py
cd frontend
python app.py
```

## Notes

- `OVERSEEC_IMAGE_EDIT_TEST_MODE=1` disables model pipeline and LLM features so the frontend can boot with a lightweight dependency set.
- GeoTIFF export stays unavailable unless GDAL is installed.
- The fixture generator writes synthetic data under `frontend/results/edit_test_scene/` and `frontend/results/final_costmaps/`.
