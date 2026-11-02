"""evaluate contact labels using measured wrist distances."""

import json
import statistics
from pathlib import Path


MEASUREMENTS_PATH = Path(
    "outputs/candidate_wrist_distance_measurements.json"
)

# this value sits between the largest measured racket-contact distance
# (102.1 px) and the smallest measured bounce distance (161.4 px).
#
# it is provisional and specific to this camera/video setup.
CONTACT_DISTANCE_THRESHOLD_PIXELS = 130.0

CONFIRMED_CONTACT_LABEL = "confirmed_racket_contact"


def load_measurements():
    """
    load the wrist-distance evidence saved by the measurement script.
    """
    with MEASUREMENTS_PATH.open("r", encoding="utf-8") as file:
        evidence = json.load(file)

    return evidence["measurements"]


def group_measured_distances_by_label(measurements):
    """group measured wrist distances by label."""
    distances_by_label = {}
    missing_wrist_counts = {}

    for measurement in measurements:
        label = measurement["manual_review_label"]
        nearest_wrist = measurement["nearest_wrist"]

        if nearest_wrist is None:
            missing_wrist_counts[label] = (
                missing_wrist_counts.get(label, 0) + 1
            )
            continue

        distance = nearest_wrist["ball_to_wrist_distance_pixels"]

        distances_by_label.setdefault(label, []).append(distance)

    return distances_by_label, missing_wrist_counts


def print_distance_summary(distances_by_label, missing_wrist_counts):
    """
    print count, minimum, maximum, and average distance for every event label.
    """
    all_labels = sorted(
        set(distances_by_label) | set(missing_wrist_counts)
    )

    print("Wrist-distance evidence by manual label:\n")

    for label in all_labels:
        distances = distances_by_label.get(label, [])
        missing_count = missing_wrist_counts.get(label, 0)

        print(f"{label}:")

        if distances:
            print(f"  measured events: {len(distances)}")
            print(f"  nearest distance, minimum: {min(distances):.1f} px")
            print(f"  nearest distance, maximum: {max(distances):.1f} px")
            print(
                f"  nearest distance, average: "
                f"{statistics.mean(distances):.1f} px"
            )
        else:
            print("  measured events: 0")

        print(f"  events without reliable wrist: {missing_count}")
        print()


def assign_contact_screening_outcome(
    measurement,
    contact_distance_threshold_pixels,
):
    """classify contact candidates from measured distances."""
    nearest_wrist = measurement["nearest_wrist"]

    if nearest_wrist is None:
        return "unknown_no_reliable_wrist"

    distance = nearest_wrist["ball_to_wrist_distance_pixels"]

    if distance <= contact_distance_threshold_pixels:
        return "likely_racket_contact"

    return "not_close_to_detected_wrist"


def print_contact_screening_evaluation(
    measurements,
    contact_distance_threshold_pixels,
):
    """print the contact screening results."""
    outcomes = {
        "likely_racket_contact": [],
        "not_close_to_detected_wrist": [],
        "unknown_no_reliable_wrist": [],
    }

    for measurement in measurements:
        outcome = assign_contact_screening_outcome(
            measurement,
            contact_distance_threshold_pixels,
        )
        outcomes[outcome].append(measurement)

    print(
        "Wrist-proximity screening evaluation "
        f"(threshold: {contact_distance_threshold_pixels:.1f} px):\n"
    )

    for outcome, outcome_measurements in outcomes.items():
        print(f"{outcome}: {len(outcome_measurements)} event(s)")

        labels_in_outcome = sorted(
            {
                measurement["manual_review_label"]
                for measurement in outcome_measurements
            }
        )

        for label in labels_in_outcome:
            label_count = sum(
                measurement["manual_review_label"] == label
                for measurement in outcome_measurements
            )
            print(f"  {label}: {label_count}")

        print()

    all_confirmed_contacts = [
        measurement
        for measurement in measurements
        if measurement["manual_review_label"] == CONFIRMED_CONTACT_LABEL
    ]

    contacts_with_reliable_wrist = [
        measurement
        for measurement in all_confirmed_contacts
        if measurement["nearest_wrist"] is not None
    ]

    correctly_flagged_contacts = [
        measurement
        for measurement in all_confirmed_contacts
        if assign_contact_screening_outcome(
            measurement,
            contact_distance_threshold_pixels,
        ) == "likely_racket_contact"
    ]

    measured_non_contacts = [
        measurement
        for measurement in measurements
        if (
            measurement["manual_review_label"] != CONFIRMED_CONTACT_LABEL
            and measurement["nearest_wrist"] is not None
        )
    ]

    incorrectly_flagged_non_contacts = [
        measurement
        for measurement in measured_non_contacts
        if assign_contact_screening_outcome(
            measurement,
            contact_distance_threshold_pixels,
        ) == "likely_racket_contact"
    ]

    print("Interpretation:")
    print(
        "  Confirmed contacts flagged among all contacts: "
        f"{len(correctly_flagged_contacts)}/{len(all_confirmed_contacts)}"
    )
    print(
        "  Confirmed contacts flagged when a reliable wrist exists: "
        f"{len(correctly_flagged_contacts)}/"
        f"{len(contacts_with_reliable_wrist)}"
    )
    print(
        "  Measured non-contact events incorrectly flagged: "
        f"{len(incorrectly_flagged_non_contacts)}/"
        f"{len(measured_non_contacts)}"
    )
    print(
        "\nImportant: unknown wrist cases remain unknown. "
        "They must not automatically become bounce confirmations."
    )


def main():
    """run the script's main workflow."""
    measurements = load_measurements()

    distances_by_label, missing_wrist_counts = (
        group_measured_distances_by_label(measurements)
    )

    print_distance_summary(
        distances_by_label,
        missing_wrist_counts,
    )

    print_contact_screening_evaluation(
        measurements,
        CONTACT_DISTANCE_THRESHOLD_PIXELS,
    )


if __name__ == "__main__":
    main()