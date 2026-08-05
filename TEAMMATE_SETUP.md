# Quick setup — you got a full folder copy, not a fresh clone

You received the *entire* project folder (not just `git clone` — everything,
including files that are normally gitignored: `.env`, `credentials/`,
`.venv/`, `data/`, `models/`, `mlruns/`). That's a shortcut: you skip
redownloading GEE exports and redoing today's Colab training entirely. But
a few things are machine-specific and won't work as-is until you fix them.
This is **not** a replacement for `SETUP.md` — it's the shorter path
specifically for "I already have a working copy of someone else's folder."

## 1. Fix `.env` — this WILL break otherwise

Open `.env` in the repo root and find this line:

```
GEE_PRIVATE_KEY_PATH=c:\Users\kxloo\OneDrive\Documents\VS Workspace\urban-heat-cooling-priority\credentials\nus-iss-urban-heat-sg-7d159eafda34.json
```

That's a hardcoded absolute path on the original machine. Unless your
Windows username is also `kxloo` and the folder landed in the exact same
location, this is wrong for you. Change it to wherever *you* put this
folder, e.g.:

```
GEE_PRIVATE_KEY_PATH=C:\Users\<your-username>\<wherever-you-put-it>\urban-heat-cooling-priority\credentials\nus-iss-urban-heat-sg-7d159eafda34.json
```

**How to tell it's right**: after step 2 below, run
```
.venv\Scripts\python.exe -c "from src.ingest.gee import init_ee; init_ee()"
```
Expect `EE initialized OK (service account), project: nus-iss-urban-heat-sg`.
A `FileNotFoundError` here means this path is still wrong.

## 2. Recreate your own `.venv` — don't reuse the copied one

Virtual environments aren't portable between machines (activation scripts
have absolute paths baked in from when they were created). Delete the
copied `.venv/` folder and make your own:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Takes a few minutes. Skipping this and trying to reuse the copied `.venv/`
as-is can produce confusing, inconsistent breakage — not worth debugging,
just recreate it.

## 3. Decide: share the service account, or create your own

Right now you'd be authenticating as the *original owner's* GEE service
account (copied along in `credentials/` + `.env`). This works fine
technically — GEE doesn't care who's behind a service account — but it
means no separate audit trail for your own usage.

- **Fine to skip**: keep using the shared credentials as-is, nothing more
  to do. Simplest, works today.
- **If you'd rather have your own** — you're already an IAM principal on
  the shared GCP project, so this is easy:
  1. Go to `https://console.cloud.google.com/iam-admin/serviceaccounts?project=nus-iss-urban-heat-sg`
  2. **+ CREATE SERVICE ACCOUNT** → give it a name (e.g. `<yourname>-urban-heat`)
  3. **Create and continue** → in the role picker, add **both** of these
     roles (click **+ ADD ANOTHER ROLE** for the second one) — both are
     required, Earth Engine will reject requests missing either:
     - **Earth Engine Resource Writer**
     - **Service Usage Consumer**
  4. **Continue** → **Done**
  5. Click into the service account you just created → **Keys** tab →
     **Add Key → Create new key → JSON → Create**. This downloads a
     `.json` file — treat it like a password, never commit it or share it
     outside a private channel.
  6. Move that file into `credentials/` at the repo root (already
     gitignored, stays local to your machine).
  7. Update `.env`'s `GEE_SERVICE_ACCOUNT` (the email shown on the service
     account's details page) and `GEE_PRIVATE_KEY_PATH` (the full path to
     the JSON file you just downloaded) to point at your new key instead
     of the copied one.

## 4. Nothing to change — these just work as copied

- `GEE_PROJECT_ID`, `GEE_EXPORT_BUCKET`, `GCS_MODEL_BUCKET` in `.env` —
  project-level values, not machine-specific.
- `data/`, `models/`, `mlruns/`, `mlflow.db` — every cached GEE export and
  **already-trained model** (U-Net, CNN, the RF/ensemble rasters) came
  along in the copy. You do not need to retrain or re-run anything to get
  a working app — see step 6.
- The git remote — already points at the shared GitHub repo, so `git
  pull`/`git push` work immediately once your GEE auth (step 1) is sorted.

## 5. Only if you'll run training yourself later

This is a per-machine tool install — it doesn't come across in a folder
copy, and you only need it if/when you want to run
`notebooks/colab_training/*.ipynb` yourself (not required just to use the
app or inspect existing results):

1. Install **"Google Colab"** from the VS Code Extensions panel.
2. To use it: open a notebook under `notebooks/colab_training/` →
   **Select Kernel** (top-right) → **Colab** → sign in with your Google
   account → **New Colab Server** → pick **GPU** (the free T4 tier is
   enough) → connect. The notebook file itself stays local and
   git-tracked as normal — only the *execution* happens on Colab's VM.

## 6. Verify everything works

```
.venv\Scripts\python.exe -c "from src.ingest.gee import init_ee; init_ee()"
streamlit run app/Home.py
```

The app should open with real data already populating every page —
including live XGBoost/CNN counterfactuals on the Counterfactual Greening
page — since the trained models came across in the folder copy.
