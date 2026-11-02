# Ball trajectory analytics

Run this after a successful RacketVision tracking run:

```powershell
python backend\analyze_ball_trajectory.py
```

writes two files:

- `outputs/ball_trajectory_analytics.json`
- `outputs/ball_bounce_candidates_review.mp4`

red dots are model observations. orange `BOUNCE? review` rings are not
declared bounces: they are moments where three consecutive direct observations
make a strong 2-D direction turn inside the court. review each one
in the video before using its court coordinate.

the image speeds are pixels per second, not km/h. A court
homography maps points on the court surface (cannot convert an airborne
ball's screen motion to its true physical speed) and accurate km/h needs calibrated
3-D reconstruction or valid ball-flight model.
