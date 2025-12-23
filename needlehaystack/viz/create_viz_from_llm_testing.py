"""Small CLI script to reproduce the Needle In A Haystack heatmap from JSON results.

Usage:
    uv run needlehaystack/viz/create_viz_from_llm_testing.py /path/to/results_dir [output.png]

The script expects JSON files with at least the following keys:
- "depth_percent"  -> becomes "Document Depth"
- "context_length" -> becomes "Context Length"
- "score"          -> becomes "Score"

It will aggregate scores by (Document Depth, Context Length) using the mean
and then plot a heatmap similar to the original notebook.
"""
import sys
import json
import glob
from pathlib import Path

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class NoResultsFoundError(RuntimeError):
    """Raised when no JSON result files are present for visualization."""


@dataclass
class VizConfig:
    """Configuration for generating a visualization heatmap."""

    results_dir: Path
    output_path: Optional[Path] = None

    def resolve(self) -> "VizConfig":
        return VizConfig(
            results_dir=self.results_dir.expanduser().resolve(),
            output_path=self.output_path.expanduser().resolve()
            if self.output_path is not None
            else None,
        )


def get_model_name(results_dir: Path) -> str:
    """Read the model name from the first JSON file in the directory."""
    pattern = str(results_dir / "*.json")
    json_files = sorted(glob.glob(pattern))
    if not json_files:
        return "Unknown model"

    with open(json_files[0], "r") as f:
        json_data = json.load(f)

    return str(json_data.get("model", "Unknown model"))


def build_dataframe(results_dir: Path) -> pd.DataFrame:
    """Load all JSON files in ``results_dir`` into a flat DataFrame.

    Each row corresponds to a single test with:
    - Document Depth
    - Context Length
    - Score
    """
    pattern = str(results_dir / "*.json")
    json_files = glob.glob(pattern)

    if not json_files:
        raise NoResultsFoundError(f"No JSON files found in directory: {results_dir}")

    data = []
    for file in json_files:
        with open(file, "r") as f:
            json_data = json.load(f)

        document_depth = json_data.get("depth_percent")
        context_length = json_data.get("context_length")
        score = json_data.get("score")

        data.append(
            {
                "Document Depth": document_depth,
                "Context Length": context_length,
                "Score": score,
            }
        )

    df = pd.DataFrame(data)
    return df


def build_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Create the pivot table used for the heatmap."""
    pivot_table = pd.pivot_table(
        df,
        values="Score",
        index=["Document Depth", "Context Length"],
        aggfunc="mean",
    ).reset_index()

    pivot_table = pivot_table.pivot(
        index="Document Depth", columns="Context Length", values="Score"
    )

    return pivot_table


def plot_heatmap(
    pivot_table: pd.DataFrame, model_name: str, output_path: Path | None = None
) -> None:
    """Plot the heatmap and either show it or save to ``output_path``."""

    # Custom colormap matching the notebook
    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"]
    )

    plt.figure(figsize=(17.5, 8))
    sns.heatmap(
        pivot_table,
        fmt="g",
        cmap=cmap,
        cbar_kws={"label": "Score"},
    )

    plt.title(
        f"Pressure Testing {model_name} Context" "\n"
        'Fact Retrieval Across Context Lengths ("Needle In A HayStack")'
    )
    plt.xlabel("Token Limit")
    plt.ylabel("Depth Percent")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Saved heatmap to {output_path}")
    else:
        plt.show()


def generate_heatmap(config: VizConfig) -> None:
    cfg = config.resolve()

    if not cfg.results_dir.is_dir():
        raise NoResultsFoundError(f"Not a directory or missing results: {cfg.results_dir}")

    model_name = get_model_name(cfg.results_dir)
    df = build_dataframe(cfg.results_dir)
    if df.empty:
        raise NoResultsFoundError(f"No rows to visualize in: {cfg.results_dir}")

    pivot_table = build_pivot(df)
    plot_heatmap(pivot_table, model_name, cfg.output_path)


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        script = Path(argv[0]).name
        msg = (
            f"Usage: uv run needlehaystack/viz/{script} /path/to/results_dir [output.png]\n"
            "Example: uv run needlehaystack/viz/create_viz_from_llm_testing.py "
            "original_results/Anthropic_Original_Results heatmap.png"
        )
        raise SystemExit(msg)

    config = VizConfig(
        results_dir=Path(argv[1]),
        output_path=Path(argv[2]) if len(argv) >= 3 else None,
    )
    generate_heatmap(config)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv)
