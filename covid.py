#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import os
import re
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import pandas as pd
from pmdarima import arima
from pmdarima.model_selection import train_test_split
from sklearn.metrics import r2_score


def adjust_date(s):
    t = s.split("/")
    return f"20{t[2]}-{int(t[0]):02d}-{int(t[1]):02d}"


def adjust_name(s):
    return re.sub(r"\*|\,|\(|\)|\*|\ |\'", "_", s)


def draw(province):
    draw_(province, True)
    draw_(province, False)


def draw_(province, isDaily):
    # 模型训练
    model = arima.AutoARIMA(
        start_p=0,
        max_p=4,
        d=None,
        start_q=0,
        max_q=1,
        start_P=0,
        max_P=1,
        D=None,
        start_Q=0,
        max_Q=1,
        m=7,
        seasonal=True,
        test="kpss",
        trace=True,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )
    if isDaily:
        data = df[province].diff().dropna()
        model.fit(data)
    else:
        data = df[province]
        model.fit(data)

    # 模型验证
    train, test = train_test_split(data, train_size=0.8)
    pred_test = model.predict_in_sample(start=train.shape[0], dynamic=False)
    validating = pd.Series(pred_test, index=test.index)
    r2 = r2_score(test, pred_test)

    # 开始预测
    pred, pred_ci = model.predict(n_periods=14, return_conf_int=True)
    idx = pd.date_range(data.index.max() + pd.Timedelta("1D"), periods=14, freq="D")
    forecasting = pd.Series(pred, index=idx)

    # 绘图呈现
    plt.figure(figsize=(15, 6))

    plt.plot(data.index, data, label="Actual Value", color="blue")
    plt.plot(validating.index, validating, label="Check Value", color="orange")
    plt.plot(forecasting.index, forecasting, label="Predict Value", color="red")
    # plt.fill_between(forecasting.index, pred_ci[:, 0], pred_ci[:, 1], color="black", alpha=.25)

    plt.legend()
    plt.ticklabel_format(style="plain", axis="y")
    # plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    if isDaily:
        plt.title(
            f"Daily Confirmed Cases Forecasting - {province}\nARIMA {model.model_.order}x{model.model_.seasonal_order} (R2 = {r2:.6f})"
        )
        plt.savefig(
            os.path.join("figures", f"covid-{adjust_name(province)}-daily.svg"),
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.title(
            f"Accumulative Confirmed Cases Forecasting - {province}\nARIMA {model.model_.order}x{model.model_.seasonal_order} (R2 = {r2:.6f})"
        )
        plt.savefig(
            os.path.join("figures", f"covid-{adjust_name(province)}.svg"),
            bbox_inches="tight",
        )
        plt.close()


if __name__ == "__main__":
    # 准备数据
    df = (
        pd.read_csv(
            "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"
        )
        .drop(
            columns=[
                "UID",
                "iso2",
                "iso3",
                "code3",
                "FIPS",
                "Admin2",
                "Country_Region",
                "Lat",
                "Long_",
                "Combined_Key",
            ]
        )
        .groupby("Province_State")
        .sum()
        .transpose()
        .drop(columns=["Diamond Princess", "Grand Princess"])
    )
    df.index = pd.DatetimeIndex(df.index.map(adjust_date))

    provinces = df.columns.to_list()

    # 线程池
    with ProcessPoolExecutor() as pool:
        pool.map(draw, provinces)
    pool.shutdown(wait=True)

    # 编制索引
    with codecs.open("README.md", "w", "utf-8") as f:
        f.write(
            "[![check status](https://github.com/winsphinx/covid-us/actions/workflows/check.yml/badge.svg)](https://github.com/winsphinx/covid-us/actions/workflows/check.yml)\n"
        )
        f.write(
            "[![build status](https://github.com/winsphinx/covid-us/actions/workflows/build.yml/badge.svg)](https://github.com/winsphinx/covid-us/actions/workflows/build.yml)\n"
        )
        f.write(
            "[![Data From](https://img.shields.io/badge/Data%20From-https://github.com/CSSEGISandData/COVID--19-brightgreen)](https://github.com/CSSEGISandData/COVID-19)\n"
        )
        f.write("# COVID-19 Forecasting\n\n")
        for province in provinces:
            f.write(f"## {province}\n\n")
            f.write(f"![img](figures/covid-{adjust_name(province)}.svg)\n\n")
            f.write(f"![img](figures/covid-{adjust_name(province)}-daily.svg)\n\n")
