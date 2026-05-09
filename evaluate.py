"""
evaluate.py

Compares the system's shot predictions against manually labeled ground truth.

Usage:
    1. Run the pipeline first:  python main.py
    2. Then run this:           python evaluate.py

A prediction is correct if the shot type matches and the frame is within
FRAME_TOLERANCE frames of the ground truth frame (~1 second at 30fps).
"""

import json
from collections import defaultdict


PREDICTIONS_PATH = "results/shots_detected.json"
GROUND_TRUTH_PATH = "data/ground_truth.json"
FRAME_TOLERANCE = 30


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def find_matching_prediction(gt_frame, predictions, tolerance):
    """Return the first prediction within tolerance frames of gt_frame, or None."""
    for pred in predictions:
        if abs(pred["frame_id"] - gt_frame) <= tolerance:
            return pred
    return None


def evaluate():
    print("Loading predictions and ground truth...")

    predictions = load_json(PREDICTIONS_PATH)
    ground_truth = load_json(GROUND_TRUTH_PATH)

    print(f"  Ground truth samples : {len(ground_truth)}")
    print(f"  Predictions made     : {len(predictions)}")
    print()

    correct = 0
    wrong_type = 0
    missed = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    results_detail = []

    for gt in ground_truth:
        gt_frame = gt["frame_id"]
        gt_type = gt["shot_type"]
        per_class_total[gt_type] += 1

        pred = find_matching_prediction(gt_frame, predictions, FRAME_TOLERANCE)

        if pred is None:
            missed += 1
            results_detail.append({"gt_frame": gt_frame, "gt_type": gt_type, "pred_type": "NOT DETECTED", "result": "MISSED"})
        elif pred["shot_type"] == gt_type:
            correct += 1
            per_class_correct[gt_type] += 1
            results_detail.append({"gt_frame": gt_frame, "gt_type": gt_type, "pred_type": pred["shot_type"], "result": "CORRECT"})
        else:
            wrong_type += 1
            results_detail.append({"gt_frame": gt_frame, "gt_type": gt_type, "pred_type": pred["shot_type"], "result": "WRONG TYPE"})

    total = len(ground_truth)
    accuracy = round(correct / total * 100, 1) if total > 0 else 0
    detection_rate = round((correct + wrong_type) / total * 100, 1) if total > 0 else 0

    print(f"{'GT Frame':<12} {'GT Type':<14} {'Predicted':<14} {'Result'}")
    print("-" * 58)
    for r in results_detail:
        print(f"{r['gt_frame']:<12} {r['gt_type']:<14} {r['pred_type']:<14} {r['result']}")

    print()
    print("=" * 58)
    print(f"  Total samples    : {total}")
    print(f"  Correct          : {correct}")
    print(f"  Wrong type       : {wrong_type}")
    print(f"  Missed           : {missed}")
    print(f"  Detection rate   : {detection_rate}%  (shot found near GT frame)")
    print(f"  Accuracy         : {accuracy}%  (correct type / total GT)")
    print("=" * 58)

    print()
    print("Per-class accuracy:")
    print(f"  {'Shot Type':<14} {'Correct':<10} {'Total':<10} {'Accuracy'}")
    print("  " + "-" * 44)
    for shot_type in sorted(per_class_total.keys()):
        c = per_class_correct[shot_type]
        t = per_class_total[shot_type]
        acc = round(c / t * 100, 1) if t > 0 else 0
        print(f"  {shot_type:<14} {c:<10} {t:<10} {acc}%")

    # confusion matrix
    all_types = sorted(per_class_total.keys())
    confusion = {gt: {pred: 0 for pred in all_types + ["NOT DETECTED"]} for gt in all_types}
    for r in results_detail:
        gt = r["gt_type"]
        pred = r["pred_type"] if r["pred_type"] in all_types else "NOT DETECTED"
        confusion[gt][pred] += 1

    print()
    print("Confusion Matrix (rows=GT, cols=Predicted):")
    header = f"{'':14}" + "".join(f"{t:14}" for t in all_types) + f"{'NOT DET':14}"
    print("  " + header)
    print("  " + "-" * len(header))
    for gt in all_types:
        row = f"  {gt:<14}" + "".join(f"{confusion[gt][p]:<14}" for p in all_types) + f"{confusion[gt]['NOT DETECTED']:<14}"
        print(row)

    report = {
        "total_ground_truth": total,
        "total_predictions": len(predictions),
        "correct": correct,
        "wrong_type": wrong_type,
        "missed": missed,
        "detection_rate_pct": detection_rate,
        "accuracy_pct": accuracy,
        "frame_tolerance_used": FRAME_TOLERANCE,
        "per_class": {
            shot_type: {
                "correct": per_class_correct[shot_type],
                "total": per_class_total[shot_type],
                "accuracy_pct": round(per_class_correct[shot_type] / per_class_total[shot_type] * 100, 1)
                if per_class_total[shot_type] > 0 else 0
            }
            for shot_type in per_class_total
        },
        "confusion_matrix": confusion,
        "detail": results_detail
    }

    with open("results/evaluation_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print()
    print("Report saved -> results/evaluation_report.json")


if __name__ == "__main__":
    evaluate()
