# -*- coding: utf-8 -*-
"""第二问模块：visualization.py。由原单文件按职责拆分。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import SHOW_FIGURES, CHINESE_FONT_PATH, FIGURE_DPI
from utils import require


def make_figures(physics, exported, out, logger):
    import matplotlib

    if SHOW_FIGURES:
        matplotlib.use("TkAgg")
    else:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Patch

    if CHINESE_FONT_PATH:
        require(Path(CHINESE_FONT_PATH).is_file(), "指定中文字体不存在")
        font_manager.fontManager.addfont(CHINESE_FONT_PATH)
        font_name = font_manager.FontProperties(
            fname=CHINESE_FONT_PATH
        ).get_name()
    else:
        available = {f.name for f in font_manager.fontManager.ttflist}
        preferred = [
            "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
            "Source Han Sans SC", "SimSun",
        ]
        font_name = next((n for n in preferred if n in available), None)
        require(
            font_name is not None,
            "未找到中文字体，请填写CHINESE_FONT_PATH后重新绘图",
        )
    plt.rcParams["font.sans-serif"] = [font_name]
    plt.rcParams["axes.unicode_minus"] = False

    trips, deliveries, uav_timeline, battery_timeline, usage = exported
    d = physics.data

    def finish(fig, filename):
        fig.tight_layout()
        fig.savefig(out / filename, dpi=FIGURE_DPI, bbox_inches="tight")
        if not SHOW_FIGURES:
            plt.close(fig)

    # 1. 运输路线：颜色表示机型，线路标签表示架次。
    colors = {"A": "#2878B5", "B": "#E89818", "C": "#3B8E5A"}
    fig, ax = plt.subplots(figsize=(10, 8))
    for trip in trips:
        route = json.loads(trip["route"])
        xs = [d.longitude_deg[i] for i in route]
        ys = [d.latitude_deg[i] for i in route]
        ax.plot(xs, ys, color=colors[trip["uav_type"]], alpha=0.55, lw=1.1)
        if len(route) > 2:
            mid = route[1]
            ax.annotate(
                trip["trip_id"],
                (d.longitude_deg[mid], d.latitude_deg[mid]),
                fontsize=6, alpha=0.65,
            )
    for i in d.nodes:
        marker = "*" if i == "O01" else "o"
        ax.scatter(
            d.longitude_deg[i], d.latitude_deg[i],
            marker=marker, s=150 if i == "O01" else 30, color="black",
        )
        ax.annotate(
            i, (d.longitude_deg[i], d.latitude_deg[i]),
            xytext=(4, 4), textcoords="offset points", fontsize=8,
        )
    ax.set(
        title="第二问运输路线",
        xlabel="经度（°）", ylabel="纬度（°）",
    )
    ax.legend(handles=[
        Patch(color=colors[g], label=f"{g}型") for g in colors
    ])
    ax.grid(alpha=0.2)
    finish(fig, "q2_routes.png")

    # 2. 无人机甘特图。
    phase_colors = {
        "准备装载": "#A6A6A6",
        "爬升": "#76B7B2",
        "巡航": "#4E79A7",
        "下降": "#59A14F",
        "交接": "#F28E2B",
    }
    resources = sorted(d.uavs)
    fig, ax = plt.subplots(figsize=(13, 5))
    for r in uav_timeline:
        y = resources.index(r["uav_id"])
        ax.barh(
            y, (r["end_s"] - r["start_s"]) / 60,
            left=r["start_s"] / 60, height=0.65,
            color=phase_colors[r["phase"]],
        )
    ax.set_yticks(range(len(resources)), resources)
    ax.set(title="运输无人机作业甘特图", xlabel="时间（min）", ylabel="无人机")
    ax.legend(
        handles=[Patch(color=v, label=k) for k, v in phase_colors.items()],
        ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.15),
    )
    ax.grid(axis="x", alpha=0.2)
    finish(fig, "q2_uav_gantt.png")

    # 3. 电池周转甘特图。
    resources = sorted(d.batteries)
    fig, ax = plt.subplots(figsize=(13, 7))
    for r in battery_timeline:
        y = resources.index(r["battery_id"])
        ax.barh(
            y, (r["end_s"] - r["start_s"]) / 60,
            left=r["start_s"] / 60, height=0.65,
            color="#4E79A7" if r["phase"] == "任务占用" else "#F28E2B",
        )
    ax.set_yticks(range(len(resources)), resources)
    ax.set(title="共享电池任务与充电时间线", xlabel="时间（min）", ylabel="电池")
    ax.legend(handles=[
        Patch(color="#4E79A7", label="任务占用"),
        Patch(color="#F28E2B", label="充电"),
    ])
    ax.grid(axis="x", alpha=0.2)
    finish(fig, "q2_battery_gantt.png")

    # 4. 逐箱交付与时间要求。
    fig, ax = plt.subplots(figsize=(15, 6))
    x = np.arange(len(deliveries))
    actual = [r["delivery_s"] / 60 for r in deliveries]
    expected = [r["expected_delivery_s"] / 60 for r in deliveries]
    ax.scatter(x, actual, s=17, label="实际交付", color="#2878B5")
    ax.scatter(x, expected, s=18, marker="_", label="期望时间", color="#777777")
    hard_x = [i for i, r in enumerate(deliveries)
              if r["hard_deadline_s"] is not None]
    hard_y = [deliveries[i]["hard_deadline_s"] / 60 for i in hard_x]
    ax.scatter(hard_x, hard_y, s=25, marker="x",
               label="有效硬截止", color="#C23B22")
    ax.set_xticks(x, [r["cargo_id"] for r in deliveries], rotation=90, fontsize=6)
    ax.set(title="逐箱交付与时限对照", xlabel="货箱编号", ylabel="时间（min）")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    finish(fig, "q2_delivery_deadlines.png")

    # 5. 资源利用率：统一使用运输最晚返航时刻作为统计窗口。
    fig, ax = plt.subplots(figsize=(13, 5))
    labels = [r["resource_id"] for r in usage]
    task = [r["task_busy_s"] / r["horizon_s"] for r in usage]
    charge = [
        r["charge_busy_s_within_makespan"] / r["horizon_s"] for r in usage
    ]
    ax.bar(labels, task, label="任务占用", color="#4E79A7")
    ax.bar(labels, charge, bottom=task, label="充电占用", color="#F28E2B")
    ax.set(
        title="资源利用率（截至最晚返航时刻）",
        xlabel="资源编号", ylabel="占调度时域比例", ylim=(0, 1.05),
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    finish(fig, "q2_resource_utilization.png")

    logger.info("已保存5类图件")
    if SHOW_FIGURES:
        plt.show()
