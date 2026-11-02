"""register an elevated broadcast frame to the singles court, in metres."""

import cv2
import numpy as np

from learn_test_automatic_court_detection import (
    create_image_preprocessor, load_court_model, predict_landmarks,
)

WIDTH, LENGTH = 8.23, 23.77
ALLEY = 1.37
FAR_SERVICE = LENGTH / 2 - 6.40
NEAR_SERVICE = LENGTH / 2 + 6.40
COURT = np.float32([
    [-ALLEY, 0], [WIDTH + ALLEY, 0], [-ALLEY, LENGTH], [WIDTH + ALLEY, LENGTH],
    [0, 0], [0, LENGTH], [WIDTH, 0], [WIDTH, LENGTH],
    [0, FAR_SERVICE], [WIDTH, FAR_SERVICE], [0, NEAR_SERVICE], [WIDTH, NEAR_SERVICE],
    [WIDTH / 2, FAR_SERVICE], [WIDTH / 2, NEAR_SERVICE],
])
BORDERS = [(0, 1), (2, 3), (0, 2), (1, 3)]
CHECK_LINES = [(4, 5), (6, 7), (8, 9), (10, 11), (12, 13)]


def paint_samples(gray, ends, radius, scale):
    """locate a narrow bright stripe with darker court on both sides."""
    start, end = ends
    delta = end - start
    length = np.linalg.norm(delta)
    if length < 20 * scale:
        return np.empty((0, 2), np.float32)
    normal = np.array([-delta[1], delta[0]]) / length
    centers = start + np.linspace(.05, .95, 80)[:, None] * delta
    offsets = np.arange(-radius, radius + 1)
    flank = max(3, round(5 * scale))
    extended = np.arange(-radius - flank, radius + flank + 1)
    xy = centers[:, None, :] + extended[None, :, None] * normal
    profiles = cv2.remap(gray, xy[:, :, 0].astype('f'), xy[:, :, 1].astype('f'),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype('f')
    peaks = profiles[:, flank:-flank]
    contrast = peaks - np.maximum(profiles[:, :-2*flank], profiles[:, 2*flank:])
    best = contrast.argmax(axis=1)
    rows = np.arange(len(best))
    good = (contrast[rows, best] >= 12) & (peaks[rows, best] >= 100)
    return (centers[good] + offsets[best[good], None] * normal).astype('f')


def fit_border(gray, ends, scale):
    """fit a line to the court border points."""
    samples = paint_samples(gray, ends, max(4, round(20 * scale)), scale)
    if len(samples) < 36:
        return None
    vx, vy, x, y = cv2.fitLine(samples, cv2.DIST_WELSCH, 0, .01, .01).ravel()
    line = np.array([-vy, vx, vy*x - vx*y])
    errors = np.abs(samples @ line[:2] + line[2])
    supported = samples[errors <= max(1.5, 3 * scale)]
    # an ad, player or line fragment cannot supply a whole baseline.
    if len(supported) < 32 or np.linalg.norm(np.ptp(supported, axis=0)) < .6 * np.linalg.norm(ends[1]-ends[0]):
        return None
    return line


class CourtRegistration:
    def __init__(self):
        """initialize the court registration object."""
        self.model = load_court_model()
        self.preprocess = create_image_preprocessor()

    def estimate(self, frame):
        """return (image-to-court matrix or none, reason) for this frame only."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        scale = w / 1920
        points = predict_landmarks(frame, self.model, self.preprocess)
        outer = points[[0, 1, 3, 2]]
        if (not np.isfinite(outer).all() or (outer < 0).any()
                or (outer >= [w, h]).any() or not cv2.isContourConvex(outer)
                or cv2.contourArea(outer) < .035 * w * h
                or outer[0, 0] >= outer[1, 0] or outer[0, 1] >= outer[3, 1]):
            return None, "unsupported_view"

        borders = [fit_border(gray, points[[a, b]], scale) for a, b in BORDERS]
        if any(line is None for line in borders):
            return None, "insufficient_border_paint"
        far, near, left, right = borders
        corners = []
        for baseline, sideline in [(far, left), (far, right), (near, right), (near, left)]:
            cross = np.cross(baseline, sideline)
            if abs(cross[2]) < .1:
                return None, "unstable_intersection"
            corners.append(cross[:2] / cross[2])
        corners = np.float32(corners)
        if not cv2.isContourConvex(corners) or np.max(np.linalg.norm(corners-outer, axis=1)) > 36 * scale:
            return None, "inconsistent_borders"
        court_to_image = cv2.getPerspectiveTransform(COURT[[0, 1, 3, 2]], corners)
        projected = cv2.perspectiveTransform(COURT[None], court_to_image)[0]

        # these lines played no part in fitting the matrix. look for actual paint
        # beside their predicted locations in the original, unannotated frame.
        support = [len(paint_samples(gray, projected[[a, b]], max(2, round(4 * scale)), scale))
                   for a, b in CHECK_LINES]
        if min(support[:4]) < 32 or support[4] < 24:
            return None, "interior_lines_unverified"
        return cv2.getPerspectiveTransform(corners, COURT[[0, 1, 3, 2]]), "paint_verified"
