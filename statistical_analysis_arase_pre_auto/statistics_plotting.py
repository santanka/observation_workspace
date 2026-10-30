"""Shared styling and artifact policy for statistics figures."""

GRID_CONFIG = {"minor_ticks": True, "which": "both", "alpha": 0.3}


def detail_grid(ax):
    ax.minorticks_on()
    ax.grid(True, which=GRID_CONFIG["which"], axis="both", alpha=GRID_CONFIG["alpha"])


def save_figure(fig, path, **kwargs):
    """Render to a sibling temporary file, then replace the completed figure."""
    from pathlib import Path
    import tempfile
    path = Path(path)
    with tempfile.NamedTemporaryFile(prefix=".arase_figure_", suffix=path.suffix,
                                     dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        fig.savefig(temporary, **kwargs)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def add_output_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--overwrite", action="store_true", help="Regenerate figure/CSV artifacts")
    group.add_argument("--skip-existing", action="store_true", help="Skip complete artifact sets; reject partial sets")


def should_write(base, overwrite=True, extensions=(".png", ".pdf", ".csv")):
    paths = [base.with_suffix(suffix) for suffix in extensions]
    existing = [path.exists() for path in paths]
    if overwrite or not any(existing):
        return True
    if all(path.is_file() and path.stat().st_size > 0 for path in paths):
        print(f"Skip existing: {base}", flush=True)
        return False
    raise FileExistsError(f"Partial figure/CSV set: {base}; use --overwrite to regenerate")
