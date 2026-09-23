# -*- coding: utf-8 -*-
"""第二问模块：baseline.py。由原单文件按职责拆分。"""
from __future__ import annotations

import math
from collections import defaultdict


def greedy_schedule(physics, candidates, single_only=False):
    """
    仅作基线/初始解，不将构造失败当作原问题不可行。
    电池在起飞时就绪，准备可与该电池充电重叠。
    """
    d = physics.data
    allowed = [
        c for c in candidates
        if not single_only or len(c.route) == 1
    ]
    by_box = defaultdict(list)
    for c in allowed:
        for b in c.boxes:
            by_box[b].append(c)

    remaining = set(d.cargo)
    uav_ready = {u: 0.0 for u in d.uavs}
    battery_ready = {c: 0.0 for c in d.batteries}
    schedule = []

    while remaining:
        seed = min(remaining, key=lambda b: (
            d.hard_deadline_s[b] or math.inf,
            d.expected_delivery_s[b],
            -d.priority_weight[b],
            b,
        ))
        choices = []
        for c in by_box[seed]:
            if not set(c.boxes).issubset(remaining):
                continue
            g = c.uav_type
            u = min(d.uavs_by_type[g], key=lambda x: (uav_ready[x], x))
            bat = min(
                d.batteries_by_type[g],
                key=lambda x: (battery_ready[x], x),
            )
            start = max(
                0.0, uav_ready[u], battery_ready[bat] - c.prep_s
            )
            if start > c.latest_start_s + 1e-8:
                continue
            tardiness = sum(
                d.priority_weight[b] * max(
                    0.0,
                    start + c.delivery_offset_s[b] - d.expected_delivery_s[b],
                )
                for b in c.boxes
            )
            score = (
                tardiness / len(c.boxes),
                (start + c.duration_s) / len(c.boxes),
                c.energy_kwh / len(c.boxes),
                c.candidate_id,
            )
            choices.append((score, c, u, bat, start))

        if not choices:
            return None

        _, c, u, bat, start = min(choices, key=lambda x: x[0])
        schedule.append({
            "candidate": c,
            "uav_id": u,
            "battery_id": bat,
            "start_s": start,
        })
        uav_ready[u] = start + c.duration_s
        battery_ready[bat] = start + c.duration_s + c.charge_s
        remaining.difference_update(c.boxes)

    return schedule
