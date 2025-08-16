# AI Crop Disease Detector

Minimal scaffold for training, exporting, and serving plant disease detection models.

## Structure
- Notebooks in `notebooks/`
- Source code in `src/`
- API in `src/api/`
- Streamlit demo in `streamlit_app/`
- Tests in `tests/`

## Quickstart
1. Install dependencies with Poetry:
   ```bash
   poetry install
   ```
2. Activate the environment:
   ```bash
   poetry shell
   ```
3. Run API (dev):
   ```bash
   uvicorn src.api.main:app --reload
   ```
4. Launch Streamlit demo:
   ```bash
   streamlit run streamlit_app/app.py
   ```

## Notes
- PyTorch and TensorFlow are available as optional dependency groups.
- Install specific frameworks: `poetry install --with pytorch` or `poetry install --with tensorflow`
- Implement actual training in `src/model/train.py` and inference in `src/model/inference.py`.
- Test