# Streamlit Deployment Notes

## Live App

- URL: https://xg-interface.streamlit.app/

## Source Repository

- GitHub: https://github.com/fadhilra101/xg-interface

## What was changed for Streamlit Cloud

- Added a `.streamlit/config.toml` file for cloud-friendly settings.
- Added `runtime.txt` to pin the Python version.
- Added a fallback model path so the app can still launch if `xg_model.joblib` is unavailable during deployment.

## Deploy checklist

1. Make sure the GitHub repository is published and up to date.
2. Confirm Streamlit Cloud is connected to the GitHub account.
3. Set the app entry point to `app.py`.
4. Redeploy after every change to the main branch.

## Notes

- The included fallback model is intended for demo/deployment continuity.
- For production-quality predictions, keep the trained `xg_model.joblib` in the repository root or the deployment path used by the app.
