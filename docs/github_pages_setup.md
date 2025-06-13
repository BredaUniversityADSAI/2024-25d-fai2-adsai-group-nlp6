# GitHub Pages Setup Guide

This guide explains how to fix the Sphinx documentation workflow and enable GitHub Pages deployment.

## Issues Fixed

### 1. Sphinx Configuration Issues
- ✅ Added `autodoc_mock_imports` to handle missing dependencies during documentation build
- ✅ Created missing `_static` directory to resolve path warnings
- ✅ Enhanced autodoc configuration for better documentation generation

### 2. GitHub Pages Deployment Issues
- ⚠️ **Action Required**: GitHub Pages needs to be enabled in repository settings

### 3. Workflow Improvements
- ✅ Split build and deployment into separate jobs
- ✅ Added proper error handling and artifact management
- ✅ Only deploy on main branch pushes (not PRs)

## Required Actions

### Enable GitHub Pages

1. Go to your repository settings: `https://github.com/BredaUniversityADSAI/2024-25d-fai2-adsai-group-nlp6/settings/pages`

2. Under "Source", select **"GitHub Actions"**

3. Save the settings

### Verify Workflow Permissions

Ensure the workflow has the necessary permissions:

1. Go to `Settings` → `Actions` → `General`
2. Under "Workflow permissions", select **"Read and write permissions"**
3. Check **"Allow GitHub Actions to create and approve pull requests"**

## Workflow Features

### Build Job (`build-docs`)
- Installs documentation dependencies
- Creates required directories automatically
- Builds Sphinx documentation with error checking
- Uploads documentation artifact

### Deploy Job (`deploy-docs`)
- Only runs on main branch pushes
- Uses GitHub Pages environment
- Deploys documentation to GitHub Pages

## Testing the Fix

1. **Local Testing**:
   ```bash
   cd docs
   pip install -r requirements-docs.txt
   sphinx-build -b html . _build/html
   ```

2. **CI/CD Testing**:
   - Create a pull request to test the build job
   - Merge to main to test full deployment

## Dependencies Mocked for Documentation

The following dependencies are mocked during documentation build to avoid import errors:

- Data Science: `matplotlib`, `pandas`, `numpy`, `seaborn`, `sklearn`
- NLP: `nltk`, `transformers`, `sentence_transformers`, `textblob`
- ML Framework: `torch`, `torchvision`, `torchaudio`
- API: `fastapi`, `uvicorn`
- Audio/Video: `assemblyai`, `whisper`, `ffmpeg`, `pytubefix`
- Cloud: `azure`, `azureml`, `mlflow`
- Utilities: `tqdm`, `termcolor`, `tabulate`, `protobuf`, `sentencepiece`

## Troubleshooting

### Common Issues

1. **404 Error on Deployment**:
   - Ensure GitHub Pages is enabled with "GitHub Actions" as source
   - Check repository permissions

2. **Build Warnings**:
   - Mock imports handle most dependency warnings
   - Add new dependencies to `autodoc_mock_imports` if needed

3. **Missing Static Files**:
   - `_static` directory is now created automatically
   - Add custom CSS/JS files to `docs/_static/`

### Monitoring

- Check workflow status in GitHub Actions tab
- Documentation will be available at: `https://bredauniversityadsai.github.io/2024-25d-fai2-adsai-group-nlp6/`

## Next Steps

1. Enable GitHub Pages in repository settings
2. Commit and push these changes
3. Monitor the workflow execution
4. Verify documentation deployment

The documentation will automatically rebuild and deploy when changes are pushed to the main branch.
