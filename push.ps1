# Push this repository to GitHub.
# 1. Create an EMPTY public repository on github.com first (no README, no .gitignore, no licence).
#    Suggested name: trust-lus-code      Suggested description: Analysis code for reliability-based B-line routing (TRUST_LUS)
# 2. Replace <YOUR-GITHUB-USER> below, then run this script from any PowerShell window.

$REPO = Join-Path $PSScriptRoot ""
$USER = "<YOUR-GITHUB-USER>"
$NAME = "trust-lus-code"

# If the local folder already is the repo root, push from here:
Set-Location $REPO
git remote remove origin 2>$null
git remote add origin "https://github.com/$USER/$NAME.git"
git branch -M main
git push -u origin main

# --- if you prefer the GitHub CLI ---
#   winget install --id GitHub.cli
#   gh auth login
#   gh repo create $NAME --public --source . --push