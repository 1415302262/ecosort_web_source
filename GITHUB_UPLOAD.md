# GitHub Upload Guide

This directory is the clean GitHub package for EcoSort.

## Current Target

- GitHub owner: `1415302262`
- Repository name: `ecosort_web_source`
- Local package path: `D:\桌面\ecosort_web_source`

## Included

- Core training and evaluation code under `src/` and `experiments/`
- Flask inference backend under `backend/`
- Static frontend under `backend/static/`
- Hugging Face Space entry point under `hf_space/`
- Render deployment files
- Runtime mapping and YAML configs
- Sample dataset directory template
- `checkpoints/best_model.pth` as a Git LFS object

## Excluded

- Raw datasets
- Full checkpoint experiment folders
- WandB runs
- Logs
- Local Conda/virtual environments
- Claude/local agent settings

## Upload Commands

Run these after authenticating GitHub as the account that owns `1415302262` or has write access to `1415302262/ecosort_web_source`.

```bash
cd D:/桌面/ecosort_web_source

gh auth login -h github.com

# Install Git LFS first if `git lfs version` fails.
# Conda example:
# conda install -c conda-forge git-lfs

git lfs install
git remote set-url origin https://github.com/1415302262/ecosort_web_source.git

# If the repository already exists:
git lfs push origin main --all
git push -u origin main
```

If the repository does not exist yet, create it first:

```bash
gh repo create 1415302262/ecosort_web_source --public --source . --remote origin --push
```

## Verification

After upload, check:

```bash
gh repo view 1415302262/ecosort_web_source --web
git status -sb
```

The expected local status should be clean and no longer ahead of `origin/main`.
