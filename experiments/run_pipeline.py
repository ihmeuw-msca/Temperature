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

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pplkit.data.interface import DataInterface

from temperature import (
    actions,
    process,
    score,
    utils,
    viz,
)


def main(
    outcome: str,
    data: str,
    result: str,
    n_samples: int = 1000,
):
    dataif = DataInterface(data=data, result=result)

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
    tdata = process.load_data(dataif.data, outcome)
    tdata = actions.mtslice.adjust_mean(tdata)
    dataif.dump_result(tdata, f"{outcome}_tdata.pkl")

    tdata_agg = actions.mtslice.aggregate_mtslice(tdata)
    tdata_agg = actions.mtslice.adjust_agg_std(tdata_agg)
    dataif.dump_result(tdata_agg, f"{outcome}_tdata_agg.pkl")

    # fit the mean surface
    # -------------------------------------------------------------------------
    linear_no_mono = "inj" in outcome
    surface_result = actions.surface.fit_surface(
        tdata_agg, linear_no_mono=linear_no_mono
    )
    dataif.dump_result(surface_result, f"{outcome}_surface_result.pkl")

    # fit the study structure in the residual
    # -------------------------------------------------------------------------
    trend_result, tdata_residual = actions.mtslice.fit_trend(
        tdata, surface_result, inlier_pct=0.95
    )
    dataif.dump_result(trend_result, f"{outcome}_trend_result.pkl")
    dataif.dump_result(tdata_residual, f"{outcome}_tdata_residual.pkl")

    # predict surface with UI
    # -----------------------------------------------------------------------------
    annual_temps, daily_temps = utils.create_grid_points_alt(
        np.unique(tdata_agg.mean_temp), 0.1, tdata
    )
    curve_samples = process.sample_surface(
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
    dataif.dump_result(curve_samples_df, f"{outcome}_curve_samples.parquet")

    evidence_score = score.scorelator(
        curve_samples_df, trend_result, tdata, outcome, dataif.result
    )
    dataif.dump_result(evidence_score, f"{outcome}_score.csv")

    del curve_samples, curve_samples_df

    # plot the result
    # -------------------------------------------------------------------------
    # 3D surface and the level plot
    actions.surface.plot_surface(tdata_agg, surface_result)
    plt.savefig(
        dataif.result / f"{outcome}_surface.pdf",
        bbox_inches="tight",
    )
    # plot uncertainty for each mean temp (can be subset of this)
    plt.figure(figsize=(8, 6))
    for mt in trend_result.mean_temp:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        viz.plot_slice_uncertainty(
            mt, tdata, surface_result, trend_result, ylim=[-1.0, 1.0], ax=ax
        )
        ax.set_xlabel("daily temperature")
        ax.set_title(outcome + " at mean temperature %i" % mt)
        fig.savefig(
            dataif.result / f"{outcome}_slice_{int(mt)}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)


if __name__ == "__main__":
    fire.Fire(main)
