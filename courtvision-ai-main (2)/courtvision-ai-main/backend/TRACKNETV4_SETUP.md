# TrackNetV4: separate temporal ball-tracking model

## Why this is a separate environment

find_moving_ball_candidates.py remains a diagnostic baseline. It looks for
yellow or bright moving shapes, so it can confuse court markings, player
clothing, and broadcast graphics for a tennis ball. TrackNetV4 instead uses a
temporal three-frame model and produces a heatmap for the ball location.

Do **not** add TensorFlow to the CourtVision .venv. The official TrackNetV4
environment is Linux-specific and pins Python 3.9 with TensorFlow 2.17. This
project uses a Windows Python 3.11 environment for Ultralytics.

## One-time setup: Windows plus WSL

These commands install software on your computer, so run them yourself after
reading them. Open **PowerShell as Administrator** only for this first command:

~~~powershell
wsl --install -d Ubuntu
~~~

Restart if Windows asks. Then open the new **Ubuntu** app and run:

~~~bash
sudo apt update
sudo apt install -y git curl bzip2
curl -L -o Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3.sh
~~~

Close and reopen Ubuntu after the Miniforge installer finishes. Create the
environment that TrackNetV4 publishes:

~~~bash
git clone https://github.com/AR4152/TrackNetV4.git ~/TrackNetV4
cd ~/TrackNetV4
conda env create -f environment.yml
conda activate TrackNetV4
~~~

Important: do not start a prediction yet. As checked on 2026-10-18, the
official TrackNetV4 results page lists "Download" placeholders rather than
public model files. This means a compatible .keras tennis checkpoint is not
available from that repository right now. We would need either to train one
from labelled data or deliberately switch to a different model that publishes
usable weights.

Check that the new Linux environment can see your NVIDIA GPU:

~~~bash
nvidia-smi
~~~

If it cannot, the model may still run on CPU but will be much slower.

## Run from CourtVision PowerShell

Only use this command after a verified compatible tennis checkpoint has been
obtained. Return to PowerShell in C:\Users\Dilos\Downloads\courtvision-ai,
activate the normal CourtVision virtual environment, and replace the example
checkpoint name with the actual model:

~~~powershell
.\.venv\Scripts\Activate.ps1
python backend\run_tracknetv4.py --runtime wsl --tracknet-root /home/YOUR_LINUX_USERNAME/TrackNetV4 --tracknet-python /home/YOUR_LINUX_USERNAME/miniforge3/envs/TrackNetV4/bin/python --model-weights /home/YOUR_LINUX_USERNAME/TrackNetV4/models/YOUR_TENNIS_CHECKPOINT.keras
~~~

The launcher saves TrackNetV4's untouched output to outputs\tracknetv4_raw\.
It asks TrackNetV4 for --queue-length 0, so the official preview has no green
history trail that could imply a ball position between observations.

## Import direct temporal observations

TrackNetV4 writes a CSV named after the input video, normally
test_rally_predict.csv. Import it like this:

~~~powershell
python backend\import_tracknetv4_predictions.py --predictions-csv outputs\tracknetv4_raw\test_rally_predict.csv
~~~

This produces:

- outputs\tracknetv4_ball_tracking.mp4 — red dots only for direct model observations.
- outputs\tracknetv4_ball_tracking_data.json — observations plus coverage,
  missed-run, and provenance information.

The importer automatically uses CSV row order when the official predictor
repeats frame IDs. That prevents the current grouped-three-frame output from
silently shifting the timeline. It never fills missing frames with made-up
points.

## What to inspect before trusting the model

Watch the first 20 to 30 seconds and answer three separate questions:

1. Is a red dot usually on the physical ball when it is visibly clear?
2. Does the dot jump to a player, logo, or court marking? Count those false positives.
3. Does the JSON report a long missing-observation run during an ordinary rally?

Keep the baseline output as a comparison. The new model becomes the primary
ball source only after it improves both false positives and long misses on
multiple clips from this camera style.
