# -*- coding: utf-8 -*-
"""第二问模块：reporting.py。由原单文件按职责拆分。"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from config import TRAJECTORY_STEP_S, PROTOCOL_ROOT, PROTOCOL_FILES
from candidates import evaluate_candidate
from utils import encode_list, write_csv, raw_metrics, sha256


def export_results(physics, schedule, out, run_id, solver_report):
    d = physics.data
    schedule = sorted(
        schedule,
        key=lambda x: (x["start_s"], x["uav_id"],
                       x["candidate"].candidate_id),
    )
    trips, deliveries, uav_timeline, battery_timeline = [], [], [], []
    trajectory, trajectory_segments = [], []

    for index, item in enumerate(schedule, 1):
        trip_id = f"T{index:03d}"
        c = item["candidate"]
        start = item["start_s"]
        fresh, phases = evaluate_candidate(
            physics, c.uav_type, c.boxes, c.route, with_phases=True
        )
        end = start + fresh.duration_s
        takeoff = start + fresh.prep_s

        # 强制接口字段均保留，扩展字段在其后。
        trips.append({
            "trip_id": trip_id,
            "uav_id": item["uav_id"],
            "uav_type": c.uav_type,
            "battery_id": item["battery_id"],
            "start_s": start,
            "route": encode_list(("O01",) + c.route + ("O01",)),
            "return_s": end,
            "energy_kwh": fresh.energy_kwh,
            "takeoff_s": takeoff,
            "cargo_ids": encode_list(c.boxes),
            "candidate_id": c.candidate_id,
            "payload_kg": sum(d.cargo_weight_kg[b] for b in c.boxes),
            "volume_m3": sum(d.cargo_volume_m3[b] for b in c.boxes),
            "return_soc": fresh.return_soc,
            "run_id": run_id,
        })

        for b in c.boxes:
            delivered = start + fresh.delivery_offset_s[b]
            deliveries.append({
                "cargo_id": b,
                "trip_id": trip_id,
                "service_area": d.service_area_of[b],
                "delivery_s": delivered,
                "cargo_type": d.cargo[b]["cargo_type"],
                "is_first_batch": d.is_first_batch[b],
                "is_medical": d.is_medical[b],
                "first_deadline_s": d.first_deadline_s[b],
                "expected_delivery_s": d.expected_delivery_s[b],
                "hard_deadline_s": d.hard_deadline_s[b],
                "lateness_s": max(0.0, delivered - d.expected_delivery_s[b]),
                "priority_weight": d.priority_weight[b],
                "run_id": run_id,
            })

        for phase in phases:
            a = start + phase["start_offset_s"]
            z = start + phase["end_offset_s"]
            p0, p1 = phase["p0"], phase["p1"]
            uav_timeline.append({
                "uav_id": item["uav_id"],
                "trip_id": trip_id,
                "phase": phase["phase"],
                "start_s": a,
                "end_s": z,
                "run_id": run_id,
            })
            trajectory_segments.append({
                "trip_id": trip_id,
                "phase": phase["phase"],
                "start_s": a,
                "end_s": z,
                "start_lon_deg": p0[0],
                "start_lat_deg": p0[1],
                "start_altitude_m": p0[2],
                "end_lon_deg": p1[0],
                "end_lat_deg": p1[1],
                "end_altitude_m": p1[2],
                "payload_kg": phase["payload_kg"],
                "interpolation": "linear_in_lon_lat_altitude",
            })
            pieces = max(1, math.ceil((z - a) / TRAJECTORY_STEP_S))
            for fraction in np.linspace(0.0, 1.0, pieces + 1):
                point = [
                    p0[j] + fraction * (p1[j] - p0[j]) for j in range(3)
                ]
                trajectory.append({
                    "trip_id": trip_id,
                    "time_s": a + fraction * (z - a),
                    "phase": phase["phase"],
                    "lon_deg": point[0],
                    "lat_deg": point[1],
                    "altitude_m": point[2],
                    "payload_kg": phase["payload_kg"],
                })

        battery_timeline.extend([
            {
                "battery_id": item["battery_id"],
                "uav_type": c.uav_type,
                "trip_id": trip_id,
                "phase": "任务占用",
                "start_s": takeoff,
                "end_s": end,
                "soc_start": 1.0,
                "soc_end": fresh.return_soc,
                "run_id": run_id,
            },
            {
                "battery_id": item["battery_id"],
                "uav_type": c.uav_type,
                "trip_id": trip_id,
                "phase": "充电",
                "start_s": end,
                "end_s": end + fresh.charge_s,
                "soc_start": fresh.return_soc,
                "soc_end": 1.0,
                "run_id": run_id,
            },
        ])

    deliveries.sort(key=lambda r: r["cargo_id"])
    write_csv(out / "q2_transport_trips.csv", trips)
    write_csv(out / "q2_cargo_delivery.csv", deliveries)
    write_csv(out / "q2_uav_timeline.csv", uav_timeline)
    write_csv(out / "q2_battery_timeline.csv", battery_timeline)
    write_csv(out / "q2_trajectories.csv", trajectory)
    write_csv(out / "q2_trajectory_segments.csv", trajectory_segments)

    metrics = raw_metrics(physics, schedule)
    makespan = metrics[1]
    resource_usage = []
    for u in d.uavs:
        busy = sum(
            r["return_s"] - r["start_s"] for r in trips if r["uav_id"] == u
        )
        resource_usage.append({
            "resource_type": "无人机",
            "resource_id": u,
            "task_busy_s": busy,
            "charge_busy_s_within_makespan": 0.0,
            "utilization": busy / makespan,
            "horizon_s": makespan,
        })
    for bat in d.batteries:
        task = sum(
            max(0.0, min(r["end_s"], makespan) - r["start_s"])
            for r in battery_timeline
            if r["battery_id"] == bat and r["phase"] == "任务占用"
        )
        charging = sum(
            max(0.0, min(r["end_s"], makespan) - r["start_s"])
            for r in battery_timeline
            if r["battery_id"] == bat and r["phase"] == "充电"
        )
        resource_usage.append({
            "resource_type": "电池",
            "resource_id": bat,
            "task_busy_s": task,
            "charge_busy_s_within_makespan": charging,
            "utilization": (task + charging) / makespan,
            "horizon_s": makespan,
        })
    write_csv(out / "q2_resource_utilization.csv", resource_usage)
    write_csv(out / "q2_metrics.csv", [{
        "weighted_tardiness": metrics[0],
        "transport_makespan_s": metrics[1],
        "total_energy_kwh": metrics[2],
        "sorties": metrics[3],
        "delivered_boxes": len(deliveries),
        "run_id": run_id,
        **solver_report,
    }])
    return trips, deliveries, uav_timeline, battery_timeline, resource_usage


def candidate_records(candidates):
    for c in candidates:
        yield {
            "candidate_id": c.candidate_id,
            "uav_type": c.uav_type,
            "cargo_ids": encode_list(c.boxes),
            "route": encode_list(c.route),
            "prep_s": c.prep_s,
            "duration_s": c.duration_s,
            "energy_kwh": c.energy_kwh,
            "return_soc": c.return_soc,
            "charge_s": c.charge_s,
            "latest_start_s": (
                c.latest_start_s if math.isfinite(c.latest_start_s) else None
            ),
            "delivery_offsets": json.dumps(
                c.delivery_offset_s, ensure_ascii=False
            ),
        }


def export_manifest(data, physics, out):
    records = []
    files = list(data.source_files) + [physics.dem_path]
    files.extend(
        PROTOCOL_ROOT / name for name in PROTOCOL_FILES
        if (PROTOCOL_ROOT / name).is_file()
    )
    files.extend(sorted(Path(__file__).resolve().parent.glob("*.py")))
    for path in files:
        records.append({
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    write_csv(out / "source_manifest.csv", records)
