# -*- coding: utf-8 -*-
"""
Main script running the experiments.

Here is the outcome list:
* lri
* resp_copd
* cvd_ihd
* cvd_stroke
* diabetes
* inj_homicide
* inj_suicide
* ckd
* inj_drowning
* neonatal
"""

from pathlib import Path

import dill
import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from temperature.mtslice import (
    adjust_agg_std,
    adjust_mean,
    aggregate_mtslice,
    fit_trend,
)
from temperature.process import load_data, sample_surface
from temperature.score import scorelator
from temperature.surface import fit_surface, plot_surface
from temperature.utils import create_grid_points_alt
from temperature.viz import plot_slice_uncertainty


def main(
    outcome: str,
    data: str,
    result: str,
    n_samples: int = 1000,
):
    data = Path(data)
    result = Path(result)
    # dataif = DataInterface(data=data, result=result)

    # # get temp stuff
    # df = pd.read_csv(path_to_data)
    # annual_temps = []
    # daily_temps = []
    # for annual_temp in np.arange(df.meanTempDegree.min(), df.meanTempDegree.max() + 1, 1):
    #     at_dt_temps = np.arange(df.loc[df.meanTempDegree == annual_temp, 'dailyTempCat'].min(),
    #                             df.loc[df.meanTempDegree == annual_temp, 'dailyTempCat'].max() + 0.1, 0.1)
    #     annual_temps += [np.repeat(annual_temp, at_dt_temps.size)]
    #     daily_temps += [at_dt_temps]
    # annual_temps = np.hstack(annual_temps)
    # daily_temps = np.hstack(daily_temps)
    # del df

    # load data
    # -------------------------------------------------------------------------
    tdata = load_data(data, outcome)
    tdata = adjust_mean(tdata)
    with open(result / f"{outcome}_tdata.pkl", "wb") as f:
        dill.dump(tdata, f)

    tdata_agg = aggregate_mtslice(tdata)
    tdata_agg = adjust_agg_std(tdata_agg)
    with open(result / f"{outcome}_tdata_agg.pkl", "wb") as f:
        dill.dump(tdata_agg, f)

    # fit the mean surface
    # -------------------------------------------------------------------------
    linear_no_mono = "inj" in outcome
    surface_result = fit_surface(tdata_agg, linear_no_mono=linear_no_mono)
    with open(result / f"{outcome}_surface_result.pkl", "wb") as f:
        dill.dump(surface_result, f)

    # fit the study structure in the residual
    # -------------------------------------------------------------------------
    trend_result, tdata_residual = fit_trend(
        tdata, surface_result, inlier_pct=0.95
    )
    with open(result / f"{outcome}_trend_result.pkl", "wb") as f:
        dill.dump(trend_result, f)
    with open(result / f"{outcome}_tdata_residual.pkl", "wb") as f:
        dill.dump(tdata_residual, f)

    # predict surface with UI
    # -----------------------------------------------------------------------------
    annual_temps, daily_temps = create_grid_points_alt(
        np.unique(tdata_agg.mean_temp), 0.1, tdata
    )
    curve_samples = sample_surface(
        mt=annual_temps,
        dt=daily_temps,
        num_samples=n_samples,
        surface_result=surface_result,
        trend_result=trend_result,
        include_re=True,
    )
    curve_samples_df = pd.DataFrame(
        np.vstack([annual_temps, daily_temps, curve_samples]).T,
        columns=["annual_temperature", "daily_temperature"]
        + [f"draw_{i}" for i in range(n_samples)],
    )
    curve_samples_df.to_parquet(result / f"{outcome}_curve_samples.parquet")

    evidence_score = scorelator(
        curve_samples_df,
        trend_result,
        tdata,
        outcome,
        result,
        n_samples=n_samples,
    )
    evidence_score.to_csv(result / f"{outcome}_score.csv", index=False)

    del curve_samples, curve_samples_df

    # plot the result
    # -------------------------------------------------------------------------
    # 3D surface and the level plot
    plot_surface(tdata_agg, surface_result)
    plt.savefig(
        result / f"{outcome}_surface.pdf",
        bbox_inches="tight",
    )
    # plot uncertainty for each mean temp (can be subset of this)
    plt.figure(figsize=(8, 6))
    for mt in trend_result.mean_temp:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        plot_slice_uncertainty(
            mt, tdata, surface_result, trend_result, ylim=[-1.0, 1.0], ax=ax
        )
        ax.set_xlabel("daily temperature")
        ax.set_title(outcome + " at mean temperature %i" % mt)
        fig.savefig(
            result / f"{outcome}_slice_{int(mt)}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)


if __name__ == "__main__":
    fire.Fire(main)
