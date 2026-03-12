import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Compare results with Stouffers estimator using use sample sizes for weights

    Specifically checking for differences in the ACC as defined in the Harvard Oxford atlas.
    """)
    return


@app.cell
def _():
    import os

    import marimo as mo
    from nilearn.datasets import fetch_atlas_harvard_oxford
    from nilearn.image import load_img, new_img_like
    from nilearn.maskers import NiftiMasker
    from nilearn.plotting import (plot_bland_altman, plot_img_comparison,
                                  plot_stat_map, show)
    from nimare.dataset import Dataset
    from nimare.extract import download_nidm_pain
    from nimare.meta.ibma import Stouffers
    from nimare.transforms import ImageTransformer
    from nimare.utils import get_resource_path

    return (
        Dataset,
        ImageTransformer,
        NiftiMasker,
        Stouffers,
        download_nidm_pain,
        fetch_atlas_harvard_oxford,
        get_resource_path,
        load_img,
        mo,
        new_img_like,
        os,
        plot_bland_altman,
        plot_img_comparison,
        plot_stat_map,
        show,
    )


@app.cell
def _(Dataset, ImageTransformer, download_nidm_pain, get_resource_path, os):
    dset_dir = download_nidm_pain()

    dset_file = os.path.join(get_resource_path(), "nidm_pain_dset.json")
    dset = Dataset(dset_file)
    dset.update_path(dset_dir)

    # Calculate missing images
    xformer = ImageTransformer(target=["varcope", "z"])
    dset = xformer.transform(dset)
    return (dset,)


@app.cell
def _(Stouffers, dset):
    meta = Stouffers()
    results = meta.fit(dset)
    img = results.get_map("z")

    meta_with_sample_size = Stouffers(use_sample_size=True)
    results_with_sample_size = meta_with_sample_size.fit(dset)
    img_with_sample_size = results_with_sample_size.get_map("z")
    return img, img_with_sample_size


@app.cell
def _(NiftiMasker, fetch_atlas_harvard_oxford, img, load_img, new_img_like):
    atlas = fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    print(f"{atlas.labels[29]=}")
    maps = load_img(atlas.maps)
    mask = new_img_like(maps, maps.get_fdata() == 29)

    masker = NiftiMasker(mask).fit(img)
    report = masker.generate_report()
    report
    return (masker,)


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
def _(img, img_with_sample_size, plot_stat_map, plotting_params, show):
    plot_stat_map(img, title="use_sample_size=False", **plotting_params)
    plot_stat_map(img_with_sample_size, title="use_sample_size=True", **plotting_params)
    show()
    return


@app.cell
def _(
    img,
    img_with_sample_size,
    masker,
    plot_bland_altman,
    plot_img_comparison,
    show,
):
    plot_img_comparison(
        img,
        img_with_sample_size,
        ref_label="use_sample_size=False",
        src_label="use_sample_size=True",
        masker=masker,
    )
    plot_bland_altman(
        img,
        img_with_sample_size,
        ref_label="use_sample_size=False",
        src_label="use_sample_size=True",
        masker=masker,
    )
    show()
    return


if __name__ == "__main__":
    app.run()
