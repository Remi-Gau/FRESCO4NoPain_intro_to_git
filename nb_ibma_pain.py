import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Image-based meta-analysis of 21 pain studies

    Adapted from "Run an image-based meta-analysis (IBMA) workflow" in nimare

    Bases on NIDM Results for 21 pain studies from NeuroVault.

    see https://nimare.readthedocs.io/en/stable/auto_examples/02_meta-analyses/02_plot_ibma.html
    """)
    return


@app.cell
def _():
    import os
    from pathlib import Path

    import marimo as mo
    from nilearn.plotting import plot_stat_map, show
    from nimare.dataset import Dataset
    from nimare.extract import download_nidm_pain
    from nimare.reports.base import run_reports
    from nimare.utils import get_resource_path
    from nimare.workflows.ibma import IBMAWorkflow

    return (
        Dataset,
        IBMAWorkflow,
        Path,
        download_nidm_pain,
        get_resource_path,
        mo,
        os,
        plot_stat_map,
        run_reports,
        show,
    )


@app.cell
def _(Dataset, download_nidm_pain, get_resource_path, os):
    dset_dir = download_nidm_pain()

    dset_file = os.path.join(get_resource_path(), "nidm_pain_dset.json")
    dset = Dataset(dset_file)
    dset.update_path(dset_dir)
    return (dset,)


@app.cell
def _(IBMAWorkflow, dset):
    workflow = IBMAWorkflow(estimator="stouffers")
    result = workflow.fit(dset)
    return (result,)


@app.cell
def _():
    plotting_params = {
        "cut_coords": [0, 0, -8],
        "draw_cross": False,
        "symmetric_cbar": True,
        "vmax": 4,
    }
    return (plotting_params,)


@app.cell
def _(Path):
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    return (output_dir,)


@app.cell
def _(output_dir, result):
    img = result.get_map("z_corr-FDR_method-indep")
    img.to_filename(output_dir / "z_corr-FDR_method-indep.nii")
    return (img,)


@app.cell
def _(img, plot_stat_map, plotting_params, show):
    fig = plot_stat_map(
        img, threshold=1.65, **plotting_params  # voxel_thresh p < .05, one-tailed
    )
    show()
    return (fig,)


@app.cell
def _(fig, output_dir):
    fig.savefig(output_dir / "ibma_pain.png")
    return


@app.cell
def _(result):
    result.tables["z_corr-FDR_method-indep_tab-clust"]
    return


@app.cell
def _(output_dir, result):
    result.tables["z_corr-FDR_method-indep_tab-clust"].to_csv(output_dir / "z_corr-FDR_method-indep_tab-clust.tsv", sep="\t")
    return


@app.cell
def _(result):
    result.tables["z_corr-FDR_method-indep_diag-Jackknife_tab-counts"]
    return


@app.cell
def _(output_dir, result, run_reports):
    run_reports(result, output_dir)
    return


if __name__ == "__main__":
    app.run()
