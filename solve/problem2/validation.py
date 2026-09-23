# -*- coding: utf-8 -*-
"""第二问模块：validation.py。由原单文件按职责拆分。"""
from __future__ import annotations

from collections import Counter, defaultdict

from config import AUDIT_TIME_TOL_S, AUDIT_ENERGY_TOL_KWH
from candidates import evaluate_candidate
from utils import write_csv


def audit_schedule(physics, schedule, output_path):
    d = physics.data
    rows = []

    def add(name, ok, detail):
        rows.append({"check": name, "passed": bool(ok), "detail": detail})

    observed = Counter()
    u_intervals = defaultdict(list)
    b_intervals = defaultdict(list)
    deliveries = {}

    for item in schedule:
        c = item["candidate"]
        start = item["start_s"]
        g = c.uav_type
        fresh = evaluate_candidate(physics, g, c.boxes, c.route)
        add(
            f"{c.candidate_id}/物理可行",
            fresh is not None,
            "重新检查质量、体积、逐段载荷、能耗及最早交付",
        )
        if fresh is None:
            continue

        add(
            f"{c.candidate_id}/能耗复算",
            abs(fresh.energy_kwh - c.energy_kwh) <= AUDIT_ENERGY_TOL_KWH,
            fresh.energy_kwh,
        )
        add(
            f"{c.candidate_id}/时间复算",
            abs(fresh.duration_s - c.duration_s) <= AUDIT_TIME_TOL_S,
            fresh.duration_s,
        )
        add(
            f"{c.candidate_id}/资源兼容",
            item["uav_id"] in d.uavs_by_type[g]
            and item["battery_id"] in d.batteries_by_type[g],
            f"{item['uav_id']}/{item['battery_id']}",
        )
        add(
            f"{c.candidate_id}/开始时刻",
            start >= -AUDIT_TIME_TOL_S,
            start,
        )
        add(
            f"{c.candidate_id}/返航SOC",
            fresh.return_soc >= d.reserve_ratio[g] - 1e-8,
            fresh.return_soc,
        )

        end = start + fresh.duration_s
        u_intervals[item["uav_id"]].append(
            (start, end, c.candidate_id)
        )
        b_intervals[item["battery_id"]].append(
            (start + fresh.prep_s, end + fresh.charge_s, c.candidate_id)
        )
        for b in c.boxes:
            observed[b] += 1
            t = start + fresh.delivery_offset_s[b]
            deliveries[b] = t
            if d.is_first_batch[b]:
                add(
                    f"{b}/首批时限",
                    t <= d.first_deadline_s[b] + AUDIT_TIME_TOL_S,
                    f"{t:.9f} <= {d.first_deadline_s[b]}",
                )
            if d.is_medical[b]:
                add(
                    f"{b}/医疗时限",
                    t <= d.expected_delivery_s[b] + AUDIT_TIME_TOL_S,
                    f"{t:.9f} <= {d.expected_delivery_s[b]}",
                )

    add(
        "80箱恰好交付一次",
        set(observed) == set(d.cargo)
        and all(observed[b] == 1 for b in d.cargo),
        dict(observed),
    )

    for label, intervals in [
        ("无人机", u_intervals), ("电池飞行及充电", b_intervals)
    ]:
        for resource, tasks in intervals.items():
            tasks.sort()
            for previous, following in zip(tasks[:-1], tasks[1:]):
                add(
                    f"{label}/{resource}/{previous[2]}->{following[2]}",
                    following[0] + AUDIT_TIME_TOL_S >= previous[1],
                    f"下一次开始{following[0]}，前次释放{previous[1]}",
                )

    write_csv(output_path, rows)
    return all(r["passed"] for r in rows)
