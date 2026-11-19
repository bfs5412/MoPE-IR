"""Aggregate first-scale MoPE usage CSVs and export figures.

Run from the project root:
    python analyze_expert_usage.py
    python analyze_expert_usage.py --input-dir /path/to/output/expert_usage
    python analyze_expert_usage.py --input-dir /path/to/new_measurements --data-origin measured
    python analyze_expert_usage.py --input-dir /path/to/synthetic --allow-simulated --preview
    python analyze_expert_usage.py --self-test

Dependencies: numpy, matplotlib. No PyTorch, model weights, or GPU is needed.
CSV ratios are in [0, 1]; figure annotations and axes use percentages.
Only scale1 (prompt1) is analyzed; other scales in input CSVs are ignored.
Measured data is the default and produces figures without simulation labels.
"""

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


SCALES = ("scale1",)
BRANCHES = {"scale1": "prompt1", "scale2": "prompt2",
            "scale3": "prompt4", "scale4": "prompt3"}
TASKS = ("denoise", "derain", "dehaze", "deblur", "lowlight")
LABELS = {
    "denoise": "Denoising", "derain": "Deraining", "dehaze": "Dehazing",
    "deblur": "Motion deblurring", "lowlight": "Low-light enhancement",
}


@dataclass
class Run:
    path: Path
    metadata: dict
    data: dict

    @property
    def task(self):
        return self.metadata["degradation"]

    @property
    def dataset(self):
        return self.metadata["dataset"]

    @property
    def sigma(self):
        return parse_sigma(self.metadata.get("sigma"))


@dataclass
class Group:
    key: str
    label: str
    task: str
    sigma: object
    image_ids: list
    ratios: dict
    counts: dict
    observations: int


def parse_sigma(value):
    if value is None or str(value).strip() == "":
        return None
    number = float(value)
    if not np.isfinite(number) or number != int(number) or number <= 0:
        raise ValueError("Invalid noise sigma: {!r}".format(value))
    return int(number)


def checkpoint_key(value):
    return str(value).replace("\\", "/").rstrip("/")


def flag_is_true(value):
    if value is None or value is False:
        return False
    if value is True:
        return True
    normalized = str(value).strip().lower()
    if normalized in ("", "0", "false", "no"):
        return False
    if normalized in ("1", "true", "yes"):
        return True
    raise ValueError("Invalid boolean data-origin flag: {!r}".format(value))


def is_simulated(metadata):
    origin = str(metadata.get("data_origin", "")).strip().lower()
    return (flag_is_true(metadata.get("synthetic"))
            or flag_is_true(metadata.get("simulated"))
            or origin in ("synthetic", "simulated"))


def discover_runs(root, sigmas, checkpoint=None, allow_simulated=False, data_origin=None):
    """Select one latest completed run per task/dataset/noise level."""
    candidates = {}
    ignored = []
    for metadata_path in sorted(root.rglob("metadata.json")):
        try:
            with metadata_path.open(encoding="utf-8-sig") as handle:
                meta = json.load(handle)
            task = meta.get("degradation")
            if task not in TASKS:
                continue
            if data_origin == "synthetic" and not is_simulated(meta):
                ignored.append({"path": str(metadata_path.parent), "reason": "measured data excluded in synthetic mode"})
                continue
            if is_simulated(meta) and not allow_simulated:
                ignored.append({"path": str(metadata_path.parent), "reason": "simulated data excluded"})
                continue
            if meta.get("status") != "complete":
                ignored.append({"path": str(metadata_path.parent), "reason": "run is not complete"})
                continue
            sigma = parse_sigma(meta.get("sigma"))
            if task == "denoise" and sigma not in sigmas:
                continue
            if task != "denoise" and sigma is not None:
                raise ValueError("Non-denoising run unexpectedly has sigma={}".format(sigma))
            if checkpoint and checkpoint_key(meta.get("checkpoint_path")) != checkpoint_key(checkpoint):
                ignored.append({"path": str(metadata_path.parent), "reason": "different checkpoint filter"})
                continue
            if not meta.get("dataset") or not meta.get("checkpoint_path"):
                raise ValueError("Missing dataset or checkpoint_path")
            key = (task, str(meta["dataset"]), sigma)
            candidates.setdefault(key, []).append((metadata_path.parent, meta))
        except (ValueError, OSError, TypeError) as error:
            ignored.append({"path": str(metadata_path.parent), "reason": str(error)})

    selected = []
    for key in sorted(candidates, key=lambda item: (item[0], item[1], item[2] or 0)):
        versions = sorted(candidates[key], key=lambda item: (item[0].name, str(item[0])))
        selected.append(versions[-1])
        for path, _ in versions[:-1]:
            ignored.append({"path": str(path), "reason": "superseded by latest complete run"})
    if not selected:
        reasons = "\n".join("  {}: {}".format(item["path"], item["reason"]) for item in ignored[:5])
        raise ValueError("No matching completed runs found under {}. Check the input directory, "
                         "metadata.json status, and --data-origin (default: measured).\n{}".format(root, reasons))
    return selected, ignored


def load_run(path, meta):
    """Read only the selected scale and recompute its ratios from counts."""
    experts = int(meta["num_experts"])
    if experts < 2 or int(meta.get("top_k", 0)) != 1:
        raise ValueError("{}: expected a multi-expert Top-1 run".format(path))
    count_columns = ["expert{}_count".format(e + 1) for e in range(experts)]
    ratio_columns = ["expert{}_ratio".format(e + 1) for e in range(experts)]
    data = {}
    csv_path = path / "per_image.csv"
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"image_name", "degradation", "dataset", "sigma", "scale", "branch",
                    "valid_token_count"}.union(count_columns, ratio_columns)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError("{}: missing CSV columns {}".format(csv_path, sorted(missing)))
        for line_number, row in enumerate(reader, 2):
            try:
                name, scale = row["image_name"], row["scale"]
                if scale not in BRANCHES:
                    raise ValueError("unknown scale")
                if scale not in SCALES:
                    continue
                if not name or row["branch"] != BRANCHES[scale]:
                    raise ValueError("invalid image name, scale, or branch")
                if is_simulated(row) and not is_simulated(meta):
                    raise ValueError("CSV row is marked synthetic but metadata identifies measured data; "
                                     "use newly collected measurement files rather than relabeling old fixtures")
                if (row["degradation"] != meta["degradation"]
                        or row["dataset"] != meta["dataset"]
                        or parse_sigma(row["sigma"]) != parse_sigma(meta.get("sigma"))):
                    raise ValueError("CSV labels disagree with metadata")
                counts = np.asarray([int(row[column]) for column in count_columns], dtype=np.int64)
                tokens = int(row["valid_token_count"])
                if tokens <= 0 or (counts < 0).any() or counts.sum() != tokens:
                    raise ValueError("expert counts must be nonnegative and sum to valid_token_count")
                ratios = counts.astype(np.float64) / tokens
                stored = np.asarray([float(row[column]) for column in ratio_columns])
                if not np.isfinite(stored).all() or not np.allclose(stored, ratios, rtol=1e-6, atol=1e-8):
                    raise ValueError("saved ratios disagree with token counts")
                image = data.setdefault(name, {})
                if scale in image:
                    raise ValueError("duplicate image/scale row")
                image[scale] = (counts, ratios)
            except (ValueError, TypeError, KeyError, OverflowError) as error:
                raise ValueError("{}:{}: {}".format(csv_path, line_number, error)) from error
    if not data:
        raise ValueError("{} contains no image records".format(csv_path))
    for name, rows in data.items():
        if set(rows) != set(SCALES):
            raise ValueError("{}: image {} lacks selected scales {}".format(csv_path, name, SCALES))
    if int(meta.get("num_images", -1)) != len(data):
        raise ValueError("{}: metadata num_images differs from CSV image count".format(path))
    return Run(path, meta, data)


def validate_runs(runs, sigmas):
    if len({is_simulated(run.metadata) for run in runs}) != 1:
        raise ValueError("Measured and simulated runs must not be combined. Select a separate input directory.")
    checkpoints = {checkpoint_key(run.metadata["checkpoint_path"]) for run in runs}
    if len(checkpoints) != 1:
        raise ValueError("Selected runs use different checkpoints: {}. Use --checkpoint to select one model.".format(
            sorted(checkpoints)))
    if len({int(run.metadata["num_experts"]) for run in runs}) != 1:
        raise ValueError("Selected runs have different expert counts")
    missing = set(TASKS).difference(run.task for run in runs)
    if missing:
        raise ValueError("Missing completed degradation datasets: {}".format(sorted(missing)))
    actual_sigmas = {run.sigma for run in runs if run.task == "denoise"}
    if actual_sigmas != set(sigmas):
        raise ValueError("Expected denoising sigmas {}, found {}".format(sigmas, sorted(actual_sigmas)))


def build_condition_groups(runs, sigmas):
    grouped = {}
    for run in runs:
        key = (run.task, run.sigma)
        entries = grouped.setdefault(key, {})
        for name, rows in run.data.items():
            image_id = (run.dataset, name)
            if image_id in entries:
                raise ValueError("Duplicate scene {} in {}".format(image_id, key))
            entries[image_id] = rows
    order = [("denoise", sigma) for sigma in sigmas] + [(task, None) for task in TASKS[1:]]
    groups = []
    for task, sigma in order:
        entries = grouped[(task, sigma)]
        ids = sorted(entries)
        ratios = {s: np.stack([entries[n][s][1] for n in ids]) for s in SCALES}
        counts = {s: np.stack([entries[n][s][0] for n in ids]) for s in SCALES}
        key = "denoise_sigma_{}".format(sigma) if task == "denoise" else task
        label = "Denoising (sigma={})".format(sigma) if task == "denoise" else LABELS[task]
        groups.append(Group(key, label, task, sigma, ids, ratios, counts, len(ids)))
    return groups


def merge_denoising(groups):
    """Average sigma versions within each scene before averaging/bootstrapping scenes."""
    noise = [group for group in groups if group.task == "denoise"]
    ids = noise[0].image_ids
    if any(group.image_ids != ids for group in noise[1:]):
        raise ValueError("Denoising sigmas must contain the same (dataset, image_name) scenes. "
                         "Cannot treat repeated noise levels as independent images.")
    ratios = {s: np.mean(np.stack([g.ratios[s] for g in noise]), axis=0) for s in SCALES}
    counts = {s: np.sum(np.stack([g.counts[s] for g in noise]), axis=0) for s in SCALES}
    merged = Group("denoise", LABELS["denoise"], "denoise", None, ids, ratios, counts,
                   sum(g.observations for g in noise))
    return [merged] + [g for g in groups if g.task != "denoise"]


def bootstrap_mean(samples, repeats, rng):
    estimates = np.empty((repeats, samples.shape[1]), dtype=np.float64)
    for start in range(0, repeats, 128):
        stop = min(start + 128, repeats)
        indices = rng.integers(0, len(samples), size=(stop - start, len(samples)))
        estimates[start:stop] = samples[indices].mean(axis=1)
    return np.quantile(estimates, [0.025, 0.975], axis=0)


def calculate_statistics(groups, repeats, seed):
    rng = np.random.default_rng(seed)
    stats = {}
    for group in groups:
        for scale in SCALES:
            samples = group.ratios[scale]
            mean = samples.mean(axis=0)
            std = samples.std(axis=0, ddof=1 if len(samples) > 1 else 0)
            low, high = bootstrap_mean(samples, repeats, rng)
            positive = mean[mean > 0]
            entropy = float(-np.sum(positive * np.log(positive)))
            stats[(group.key, scale)] = {
                "mean": mean, "std": std, "ci_low": low, "ci_high": high,
                "usage_entropy": entropy / np.log(len(mean)),
                "effective_experts": float(np.exp(entropy)),
            }
    return stats


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_results(output, runs, conditions, tasks, stats, ignored, args):
    experts = int(runs[0].metadata["num_experts"])
    synthetic = is_simulated(runs[0].metadata)
    origin = "synthetic" if synthetic else "measured"
    selected = [{"degradation": r.task, "dataset": r.dataset,
                 "sigma": r.sigma if r.sigma is not None else "", "num_images": len(r.data),
                 "num_experts": experts, "checkpoint_path": r.metadata["checkpoint_path"],
                 "run_dir": str(r.path), "data_origin": origin} for r in runs]
    write_csv(output / "selected_runs.csv", list(selected[0]), selected)

    image_fields = ["degradation", "dataset", "sigma", "image_name", "scale", "branch", "run_dir",
                    "valid_token_count", "data_origin"]
    image_fields += ["expert{}_count".format(e + 1) for e in range(experts)]
    image_fields += ["expert{}_ratio".format(e + 1) for e in range(experts)]
    def image_rows():
        for run in runs:
            for name in sorted(run.data):
                for scale in SCALES:
                    counts, ratios = run.data[name][scale]
                    row = {"degradation": run.task, "dataset": run.dataset,
                           "sigma": run.sigma if run.sigma is not None else "", "image_name": name,
                           "scale": scale, "branch": BRANCHES[scale], "run_dir": str(run.path),
                           "valid_token_count": int(counts.sum()), "data_origin": origin}
                    for e in range(experts):
                        row["expert{}_count".format(e + 1)] = int(counts[e])
                        row["expert{}_ratio".format(e + 1)] = float(ratios[e])
                    yield row
    write_csv(output / "per_image_merged.csv", image_fields, image_rows())

    fields = ["condition", "degradation", "sigma", "scale", "branch", "expert",
              "num_scenes", "num_observations", "expert_token_count", "mean_ratio",
              "std_ratio", "ci95_low", "ci95_high", "image_usage_fraction", "data_origin"]
    def summary_rows(groups):
        for group in groups:
            for scale in SCALES:
                values = stats[(group.key, scale)]
                for e in range(experts):
                    yield dict(zip(fields, [group.key, group.task,
                        group.sigma if group.sigma is not None else "", scale, BRANCHES[scale],
                        e + 1, len(group.image_ids), group.observations, int(group.counts[scale][:, e].sum()),
                        float(values["mean"][e]), float(values["std"][e]),
                        float(values["ci_low"][e]), float(values["ci_high"][e]),
                        float((group.ratios[scale][:, e] > 0).mean()), origin]))
    write_csv(output / "expert_usage_5tasks.csv", fields, summary_rows(tasks))
    write_csv(output / "expert_usage_by_noise_level.csv", fields, summary_rows(conditions))

    heat_fields = ["degradation", "data_origin"] + ["expert{}".format(e + 1) for e in range(experts)]
    for prefix, groups in (("heatmap_5tasks", tasks), ("heatmap_by_noise_level", conditions)):
        for scale in SCALES:
            rows = [dict(zip(heat_fields, [g.key, origin] + stats[(g.key, scale)]["mean"].tolist())) for g in groups]
            write_csv(output / "{}_{}.csv".format(prefix, scale), heat_fields, rows)

    metric_fields = ["condition", "scale", "num_scenes", "usage_entropy_normalized",
                     "effective_experts", "max_expert_share", "active_experts", "data_origin"]
    all_groups = [tasks[0]] + conditions
    rows = []
    for group in all_groups:
        for scale in SCALES:
            values = stats[(group.key, scale)]
            rows.append(dict(zip(metric_fields, [group.key, scale, len(group.image_ids),
                values["usage_entropy"], values["effective_experts"],
                float(values["mean"].max()), int((values["mean"] > 0).sum()), origin])))
    write_csv(output / "usage_metrics.csv", metric_fields, rows)

    macro = [{"scale": s, "data_origin": origin, **{"expert{}".format(e + 1): float(np.mean(
        [stats[(g.key, s)]["mean"][e] for g in tasks])) for e in range(experts)}} for s in SCALES]
    write_csv(output / "macro_average_5tasks.csv", ["scale", "data_origin"] + heat_fields[2:], macro)
    metadata = {
        "generated_utc": datetime.utcnow().isoformat() + "Z",
        "data_origin": origin, "synthetic": synthetic,
        "preview": synthetic and bool(getattr(args, "preview", False)),
        "warning": "SIMULATED DATA - visualization only, not model measurements" if synthetic else None,
        "checkpoint_path": runs[0].metadata["checkpoint_path"], "num_experts": experts,
        "analyzed_scales": list(SCALES),
        "bootstrap_repeats": args.bootstrap, "seed": args.seed,
        "sigmas": list(args.sigmas), "csv_ratio_units": "fraction in [0, 1]",
        "aggregation": "per-image ratios averaged with equal image weights",
        "denoising": "equal sigma weights within each matched scene; bootstrap resamples scenes jointly across sigmas",
        "macro_average": "equal weight for each of the five degradation categories",
        "confidence_intervals": "95% percentile bootstrap over images/scenes; descriptive within each test set",
        "entropy": "normalized entropy of image-averaged hard expert usage, not soft gating entropy",
        "selected_runs": selected, "ignored_runs": ignored,
    }
    with (output / "analysis_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=True)
    (output / "README.txt").write_text(
        ("SIMULATED DATA - visualization only, not measured model results\n\n" if synthetic else "") +
        "MoPE expert usage analysis: scale1 (prompt1) only\n\n"
        "Other input scales are ignored. Simulated figure names have a SIMULATED_ prefix.\n"
        "--preview replaces central simulation watermarks with an external provenance footer.\n"
        "heatmap_5tasks.png/pdf: five degradation categories at the first scale.\n"
        "heatmap_by_noise_level.png/pdf: denoising sigmas shown separately.\n"
        "expert_utilization_ci95.png/pdf: category-wise usage and 95% bootstrap intervals.\n"
        "expert_usage_5tasks.csv: means, image-level standard deviations, and intervals.\n"
        "heatmap_*_scale*.csv: matrices for further plotting (fractions, not percentages).\n"
        "per_image_merged.csv: validated records from selected runs.\n"
        "selected_runs.csv: exact source folders, checkpoints, and dataset sizes.\n"
        "macro_average_5tasks.csv: equal average over the five categories.\n"
        "usage_metrics.csv: entropy of hard expert usage, not soft routing entropy.\n\n"
        "Each image contributes equally, regardless of image dimensions.\n"
        "Denoising sigmas are averaged within matched scenes before confidence estimation.\n"
        "num_scenes counts independent scene identifiers; num_observations includes sigma versions.\n"
        "Raw token totals are provided for auditing; use mean_ratio for cross-category comparisons.\n"
        "Only the first-scale expert pool is analyzed.\n"
        "Interrupted runs are excluded; only the latest completed run per dataset/sigma is selected.\n"
        "All selected runs must use the same recorded checkpoint path.\n",
        encoding="utf-8")


def make_plots(output, tasks, conditions, stats, experts, dpi, simulated=False, preview=False):
    import matplotlib
    matplotlib.use("Agg")  
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    cmap = LinearSegmentedColormap.from_list("mope_usage",
        ["#f7fbfc", "#c2e1e8", "#5aa6b8", "#176078", "#103443"])
    names = ["E{}".format(e + 1) for e in range(experts)]

    def save(fig, filename):
        if simulated:
            if preview:
                footer_y = -0.2 if filename == "expert_utilization_ci95" else -0.035
                fig.text(0.5, footer_y, "Simulated preview - not measured model results",
                         fontsize=9, color="#555555", ha="center", va="top")
                filename = "SIMULATED_PREVIEW_" + filename
            else:
                filename = "SIMULATED_" + filename
                for ax in fig.axes:
                    if ax.get_title():
                        ax.set_title(ax.get_title() + " [SIMULATED]", fontweight="bold")
                fig.suptitle("SIMULATED DATA - visualization only", color="#a32929", fontsize=15)
                fig.text(0.5, 0.5, "SIMULATED", fontsize=58, color="#a32929",
                         alpha=0.12, rotation=25, ha="center", va="center", zorder=100)
        fig.savefig(output / (filename + ".png"), dpi=dpi, bbox_inches="tight")
        fig.savefig(output / (filename + ".pdf"), bbox_inches="tight")
        plt.close(fig)

    for filename, groups in (("heatmap_5tasks", tasks), ("heatmap_by_noise_level", conditions)):
        fig, ax = plt.subplots(figsize=(8.2, 6.2 if len(groups) > 5 else 5.2),
                               constrained_layout=True)
        axes = [ax]
        for scale, ax in zip(SCALES, axes):
            matrix = np.stack([stats[(g.key, scale)]["mean"] for g in groups]) * 100
            image = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect="auto", interpolation="nearest")
            ax.set_xticks(range(experts))
            ax.set_xticklabels(names)
            ax.set_yticks(range(len(groups)))
            ax.set_yticklabels([g.label for g in groups])
            ax.set_xlabel("Expert")
            ax.set_title("Scale 1", fontweight="bold")
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            for row in range(len(groups)):
                for col in range(experts):
                    value = matrix[row, col]
                    ax.text(col, row, "{:.1f}".format(value), ha="center", va="center",
                            color="white" if value >= 55 else "#17242b", fontsize=10)
        bar = fig.colorbar(image, ax=axes, shrink=0.85, pad=0.025)
        bar.set_label("Mean token allocation (%)")
        save(fig, filename)

    colors = ("#247b9e", "#df8f44", "#4c956c", "#bf6068", "#777777")
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    axes = [ax]
    x = np.arange(experts)
    width = 0.8 / len(tasks)
    for scale, ax in zip(SCALES, axes):
        for task_index, group in enumerate(tasks):
            values = stats[(group.key, scale)]
            positions = x + (task_index - (len(tasks) - 1) / 2) * width
            mean = values["mean"] * 100
            low, high = values["ci_low"] * 100, values["ci_high"] * 100
            ax.bar(positions, mean, width=width, color=colors[task_index], label=group.label)
            ax.vlines(positions, low, high, colors="#333333", linewidth=0.8)
            ax.hlines(low, positions - width * 0.2, positions + width * 0.2, colors="#333333", linewidth=0.8)
            ax.hlines(high, positions - width * 0.2, positions + width * 0.2, colors="#333333", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel("Mean token allocation (%)")
        ax.set_xlabel("Expert")
        ax.set_title("Scale 1", fontweight="bold")
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 0.0), ncol=3,
               title="Error bars: 95% bootstrap confidence intervals", title_fontsize=9)
    save(fig, "expert_utilization_ci95")


def self_test(no_plots):
    """Small synthetic end-to-end test, including unequal image sizes and failed runs."""
    import tempfile
    from dataclasses import replace
    from types import SimpleNamespace

    assert not is_simulated({"synthetic": "false", "data_origin": "measured"})
    assert not is_simulated({"synthetic": False})
    assert is_simulated({"synthetic": "true"})
    assert is_simulated({"simulated": "true"})
    assert is_simulated({"data_origin": "synthetic"})
    with tempfile.TemporaryDirectory(prefix="mope_usage_test_") as temporary:
        root = Path(temporary)
        sigmas = [15, 25, 50]
        settings = [("denoise", sigma) for sigma in sigmas] + [(task, None) for task in TASKS[1:]]
        for task, sigma in settings:
            path = root / task / "synthetic" / str(sigma) / "run_20260101T000000_000000Z"
            path.mkdir(parents=True)
            meta = {"status": "complete", "degradation": task, "dataset": "synthetic",
                    "sigma": sigma, "num_experts": 5, "top_k": 1, "num_images": 2,
                    "checkpoint_path": "/synthetic/unified.ckpt"}
            (path / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
            fields = ["image_name", "degradation", "dataset", "sigma", "scale", "branch", "valid_token_count"]
            fields += ["expert{}_count".format(e + 1) for e in range(5)]
            fields += ["expert{}_ratio".format(e + 1) for e in range(5)]
            rows = []
            shift = sigmas.index(sigma) if task == "denoise" else 0
            for image_index, tokens in enumerate((8, 80)):
                expert = (image_index + shift) % 3
                for scale in BRANCHES:
                    row = dict(zip(fields[:7], ["image{}".format(image_index), task,
                        "synthetic", sigma if sigma is not None else "", scale, BRANCHES[scale], tokens]))
                    for e in range(5):
                        row["expert{}_count".format(e + 1)] = tokens if e == expert else 0
                        row["expert{}_ratio".format(e + 1)] = 1.0 if e == expert else 0.0
                    rows.append(row)
            write_csv(path / "per_image.csv", fields, rows)
        failed = root / "derain" / "synthetic" / "None" / "run_20990101T000000_000000Z"
        failed.mkdir(parents=True)
        (failed / "metadata.json").write_text(json.dumps({"status": "running", "degradation": "derain"}),
                                             encoding="utf-8")
        older = failed.parent / "run_20250101T000000_000000Z"
        older.mkdir()
        older_meta = {"status": "complete", "degradation": "derain", "dataset": "synthetic",
                      "sigma": None, "checkpoint_path": "/synthetic/unified.ckpt"}
        (older / "metadata.json").write_text(json.dumps(older_meta), encoding="utf-8")
        selected, ignored = discover_runs(root, sigmas)
        assert len(selected) == 7 and len(ignored) == 2
        runs = [load_run(path, meta) for path, meta in selected]
        validate_runs(runs, sigmas)
        conditions = build_condition_groups(runs, sigmas)
        tasks = merge_denoising(conditions)
        np.testing.assert_allclose(tasks[1].ratios["scale1"].mean(axis=0), [0.5, 0.5, 0, 0, 0])
        np.testing.assert_allclose(tasks[0].ratios["scale1"].mean(axis=0), [1/3, 1/3, 1/3, 0, 0])
        assert len(tasks[0].image_ids) == 2 and tasks[0].observations == 6
        stats = calculate_statistics([tasks[0]] + conditions, 100, 0)
        assert np.isclose(stats[("derain", "scale1")]["usage_entropy"], np.log(2) / np.log(5))
        np.testing.assert_allclose(stats[("denoise", "scale1")]["std"], 0, atol=1e-12)
        for group in [tasks[0]] + conditions:
            for scale in SCALES:
                assert np.isclose(stats[(group.key, scale)]["mean"].sum(), 1.0)
        try:
            wrong_meta = dict(runs[-1].metadata, checkpoint_path="/different/model.ckpt")
            validate_runs(runs[:-1] + [replace(runs[-1], metadata=wrong_meta)], sigmas)
        except ValueError as error:
            assert "different checkpoints" in str(error)
        else:
            raise AssertionError("Mixed checkpoints were not rejected")
        try:
            merge_denoising([replace(conditions[0], image_ids=list(reversed(conditions[0].image_ids)))]
                            + conditions[1:])
        except ValueError as error:
            assert "same" in str(error)
        else:
            raise AssertionError("Mismatched denoising scenes were not rejected")
        output = root / "analysis"
        output.mkdir()
        args = SimpleNamespace(bootstrap=100, seed=0, sigmas=sigmas)
        export_results(output, runs, conditions, tasks, stats, ignored, args)
        with (output / "expert_usage_5tasks.csv").open(newline="", encoding="utf-8") as handle:
            exported = list(csv.DictReader(handle))
            assert len(exported) == 5 * len(SCALES) * 5
            assert {row["scale"] for row in exported} == {"scale1"}
        assert all(set(rows) == {"scale1"} for run in runs for rows in run.data.values())
        first_scale_only = root / "scale1_only"
        first_scale_only.mkdir()
        with (runs[0].path / "per_image.csv").open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            first_scale_rows = [row for row in reader if row["scale"] == "scale1"]
        write_csv(first_scale_only / "per_image.csv", fields, first_scale_rows)
        assert len(load_run(first_scale_only, runs[0].metadata).data) == len(runs[0].data)
        source_fields = fields + ["data_origin"]
        marked_rows = [dict(row, data_origin="synthetic") for row in first_scale_rows]
        write_csv(first_scale_only / "per_image.csv", source_fields, marked_rows)
        try:
            load_run(first_scale_only, runs[0].metadata)
        except ValueError as error:
            assert "marked synthetic" in str(error)
        else:
            raise AssertionError("Synthetic rows were accepted as measured data")
        synthetic_only = root / "synthetic_only"
        synthetic_only.mkdir()
        (synthetic_only / "metadata.json").write_text(
            json.dumps(dict(runs[0].metadata, synthetic=True)), encoding="utf-8")
        try:
            discover_runs(synthetic_only, sigmas, data_origin="measured")
        except ValueError as error:
            assert "simulated data excluded" in str(error)
        else:
            raise AssertionError("Measured mode selected synthetic metadata")
        target = runs[0].path / "per_image.csv"
        with target.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields, duplicate = reader.fieldnames, next(reader)
        with target.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(duplicate)
        try:
            load_run(runs[0].path, runs[0].metadata)
        except ValueError as error:
            assert "duplicate" in str(error)
        else:
            raise AssertionError("Duplicate record was not rejected")
        if not no_plots:
            make_plots(output, tasks, conditions, stats, 5, 72)
            make_plots(output, tasks, conditions, stats, 5, 72, simulated=True, preview=True)
            for name in ("heatmap_5tasks", "heatmap_by_noise_level", "expert_utilization_ci95"):
                for suffix in (".png", ".pdf"):
                    assert (output / (name + suffix)).stat().st_size > 0
                    assert (output / ("SIMULATED_PREVIEW_" + name + suffix)).stat().st_size > 0
    print("Self-test passed: scale1 filtering, selection, equal-image weights, paired sigmas, CSV export, and duplicate detection.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent / "output" / "expert_usage")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", default=None, help="Filter by the exact checkpoint_path stored in metadata")
    parser.add_argument("--sigmas", type=int, nargs="+", default=[15, 25, 50])
    parser.add_argument("--bootstrap", type=int, default=2000, help="Image/scene bootstrap repeats for 95%% CIs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-plots", action="store_true", help="Export tables only; matplotlib is not required")
    parser.add_argument("--self-test", action="store_true", help="Run synthetic checks without server CSVs")
    origin_options = parser.add_mutually_exclusive_group()
    origin_options.add_argument("--data-origin", choices=("measured", "synthetic"), default=None,
                                help="Select measured data (default, no watermarks) or labeled synthetic fixtures")
    origin_options.add_argument("--allow-simulated", dest="data_origin", action="store_const", const="synthetic",
                                help="Alias for --data-origin synthetic; not needed for measured experiments")
    parser.add_argument("--preview", action="store_true",
                        help="For simulated plots, use an external provenance footer instead of central watermarks")
    args = parser.parse_args()
    args.data_origin = args.data_origin or "measured"
    if args.bootstrap < 1 or args.dpi < 1 or any(s <= 0 for s in args.sigmas):
        parser.error("bootstrap, dpi, and sigmas must be positive")
    if len(set(args.sigmas)) != len(args.sigmas):
        parser.error("sigmas must not contain duplicates")
    args.sigmas = sorted(args.sigmas)
    if args.self_test:
        self_test(args.no_plots)
        return
    root = args.input_dir.resolve()
    if not root.is_dir():
        raise ValueError("Input directory does not exist: {}".format(root))
    print("Data source: {}. Analyzing scale1 / prompt1 only.".format(args.data_origin))
    selected, ignored = discover_runs(root, args.sigmas, args.checkpoint,
                                     allow_simulated=args.data_origin == "synthetic", data_origin=args.data_origin)
    for item in ignored:
        print("[Skip] {}: {}".format(item["path"], item["reason"]))
    runs = [load_run(path, meta) for path, meta in selected]
    validate_runs(runs, args.sigmas)
    for run in runs:
        print("[Use] {} dataset={} sigma={} images={}\n      {}".format(
            run.task, run.dataset, run.sigma, len(run.data), run.path))
    conditions = build_condition_groups(runs, args.sigmas)
    tasks = merge_denoising(conditions)
    stats = calculate_statistics([tasks[0]] + conditions, args.bootstrap, args.seed)
    output = args.output_dir or (root / "analysis" / datetime.utcnow().strftime("run_%Y%m%dT%H%M%S_%fZ"))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    export_results(output, runs, conditions, tasks, stats, ignored, args)
    if not args.no_plots:
        make_plots(output, tasks, conditions, stats, int(runs[0].metadata["num_experts"]), args.dpi,
                   simulated=is_simulated(runs[0].metadata), preview=args.preview)
    print("\nAnalysis saved to {}".format(output))
    if not args.no_plots:
        prefix = "SIMULATED_" if is_simulated(runs[0].metadata) else ""
        if prefix and args.preview:
            prefix = "SIMULATED_PREVIEW_"
        print("Use {}heatmap_5tasks.png/.pdf for the five-degradation comparison.".format(prefix))
    print("Denoising: {} matched scenes, {} observations across sigmas {}.".format(
        len(tasks[0].image_ids), tasks[0].observations, args.sigmas))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, ImportError) as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        sys.exit(1)
