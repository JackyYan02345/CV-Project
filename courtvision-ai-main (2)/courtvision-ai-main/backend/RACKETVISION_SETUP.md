# RacketVision BallTrack setup (WSL)

RacketVision's BallTrack module is the project path to test next because it
ships a published tennis checkpoint, unlike the current TrackNetV4 repository,
whose advertised model links are placeholders. Keep it isolated in WSL: this
CourtVision folder remains responsible only for video preparation and imports.

With the `racketvision` environment activated in WSL, first confirm PyTorch is
GPU-enabled:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

The final line must be `True`. Then install the RacketVision dependencies and
checkpoint from the repository's `source` directory:

```bash
pip install -U openmim huggingface_hub pandas tqdm scikit-learn parse
mim install "mmengine>=0.10.7"
cd ~/RacketVision/source
python download_checkpoints.py --module BallTrack
```

Confirm the checkpoint exists:

```bash
ls -lh ~/RacketVision/source/BallTrack/checkpoints/balltrack_best.pth
```

Back in Windows PowerShell at the CourtVision project root, run:

```powershell
.\.venv\Scripts\Activate.ps1
python backend\run_racketvision_balltrack.py
```

It creates `outputs/racketvision_ball_tracking.mp4` and
`outputs/racketvision_ball_tracking_data.json`. The overlay marks only frames
the model reports as visible. It intentionally does not draw a guessed ball
path through a player or other occlusion.

If you need another attempt with different inputs, choose a new rally ID so
the prepared frames stay reproducible:

```powershell
python backend\run_racketvision_balltrack.py --rally-id test_rally_attempt_2
```

For a dependency fix or interrupted external inference, rerun the same command.
CourtVision detects and safely reuses the complete prepared frame set instead
of extracting it again.
