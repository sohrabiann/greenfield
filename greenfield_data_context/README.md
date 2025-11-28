# Greenfield Data Context

This repository contains the Greenfield Data Context project — dataset samples and schema definitions used for data engineers and analysts.

What’s included
- `script.py` — main script for project usage.
- `samples/` — sample CSVs for local testing (small dataset samples only).
- `schemas/` — schema text files.
- `volumes/` — raw/extracted data files (large files) — excluded from git by `.gitignore`.

How to upload this project to GitHub
1. If you don't already have git installed, download Git from https://git-scm.com/ and install.
2. Configure git (only once):
   ```powershell
   git config --global user.name "Your Name"
   git config --global user.email "you@example.com"
   ```
3. Initialize and push to a new GitHub repo (see below for options).

Notes
- Large raw data or private files are kept in `volumes/` and will not be committed by default.
- If you want to push the `samples/` folder but not the `volumes/` folder, keep the current .gitignore.

See `CONTRIBUTING.md` for how to make changes and commit (optional).
