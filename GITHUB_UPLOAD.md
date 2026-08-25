# Publishing this repository on GitHub

This folder is ready to become the repository root. Do not upload the ZIP file as the only repository file; extract it first so that GitHub displays the source tree.

## Recommended route: GitHub Desktop

1. Create an empty GitHub repository named `QualifyOT` without adding a README, `.gitignore` or license.
2. Extract `QualifyOT_GitHub.zip` locally.
3. Open GitHub Desktop and choose **File → Add local repository**.
4. Select the extracted `QualifyOT` folder.
5. If prompted, choose **create a repository here**.
6. Confirm that `.gitignore`, `README.md`, `pyproject.toml`, `src/`, `tests/`, `data/`, `examples/`, `experiments/` and `results/` are visible.
7. Commit with a simple message such as `Initial public release`.
8. Click **Publish repository** and select the GitHub repository created in step 1.
9. Replace `OWNER` in `CITATION.cff` with your GitHub account or organization name and commit that edit.
10. Check the **Actions** tab. The test workflow should run automatically on Python 3.10, 3.11 and 3.12.

Before announcing the repository publicly, choose a software license with the coauthors. This archive deliberately does not grant one automatically.
