# -*- coding: utf-8 -*-
"""第二问模块：candidates.py。由原单文件按职责拆分。"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import defaultdict
from dataclasses import dataclass

from config import PHYSICAL_TOL, MAX_BOXES_PER_CANDIDATE, EXACT_ORDER_LIMIT
from utils import require


@dataclass
class Candidate:
    candidate_id: str
    uav_type: str
    boxes: tuple
    route: tuple
    prep_s: float
    duration_s: float
    energy_kwh: float
    return_soc: float
    charge_s: float
    latest_start_s: float
    delivery_offset_s: dict


def evaluate_candidate(physics, g, boxes, route, with_phases=False):
    d = physics.data
    boxes = tuple(sorted(boxes))
    route = tuple(route)
    require(len(set(boxes)) == len(boxes), "候选货箱重复")
    require(
        len(route) == len(set(route))
        and set(route) == {d.service_area_of[b] for b in boxes},
        "候选访问顺序与货箱目的地不一致",
    )
    mass = sum(d.cargo_weight_kg[b] for b in boxes)
    volume = sum(d.cargo_volume_m3[b] for b in boxes)
    if (
        mass > d.max_payload_kg[g] + PHYSICAL_TOL
        or volume > d.max_volume_m3[g] + PHYSICAL_TOL
    ):
        return None

    prep = d.prep_time_s[g] + len(boxes) * d.load_time_per_box_s[g]
    elapsed = prep
    energy = 0.0
    remaining = mass
    delivery = {}
    phases = []

    if with_phases:
        phases.append({
            "phase": "准备装载",
            "start_offset_s": 0.0,
            "end_offset_s": prep,
            "p0": physics.position("O01"),
            "p1": physics.position("O01"),
            "payload_kg": mass,
        })

    current = "O01"
    for destination in route + ("O01",):
        cruise = physics.cruise_altitude_m[current, destination]
        phase_specs = [
            (
                "爬升",
                physics.climb_m[current, destination] / d.climb_speed_mps[g],
                physics.position(current),
                physics.position(current, cruise),
            ),
            (
                "巡航",
                physics.horizontal_distance_m[current, destination]
                / d.cruise_speed_mps[g],
                physics.position(current, cruise),
                physics.position(destination, cruise),
            ),
            (
                "下降",
                physics.descent_m[current, destination] / d.descent_speed_mps[g],
                physics.position(destination, cruise),
                physics.position(destination),
            ),
        ]
        energy += physics.segment_energy_kwh(
            g, current, destination, remaining
        )
        for phase, duration, p0, p1 in phase_specs:
            if with_phases and duration > 1e-10:
                phases.append({
                    "phase": phase,
                    "start_offset_s": elapsed,
                    "end_offset_s": elapsed + duration,
                    "p0": p0,
                    "p1": p1,
                    "payload_kg": remaining,
                })
            elapsed += duration

        if destination != "O01":
            local = [b for b in boxes if d.service_area_of[b] == destination]
            service = (
                d.handover_base_s[g]
                + len(local) * d.handover_per_box_s[g]
            )
            if with_phases:
                phases.append({
                    "phase": "交接",
                    "start_offset_s": elapsed,
                    "end_offset_s": elapsed + service,
                    "p0": physics.position(destination),
                    "p1": physics.position(destination),
                    "payload_kg": remaining,
                })
            elapsed += service
            for b in local:
                delivery[b] = elapsed
            remaining -= sum(d.cargo_weight_kg[b] for b in local)
        current = destination

    require(abs(remaining) < 1e-7, "返航载荷未归零")
    budget = (1.0 - d.reserve_ratio[g]) * d.usable_energy_kwh[g]
    if energy > budget + PHYSICAL_TOL:
        return None

    latest = math.inf
    for b in boxes:
        if d.hard_deadline_s[b] is not None:
            latest = min(latest, d.hard_deadline_s[b] - delivery[b])
    if latest < -1e-8:
        return None
    latest = max(0.0, latest)

    soc = 1.0 - energy / d.usable_energy_kwh[g]
    key = json.dumps([g, boxes, route], ensure_ascii=False)
    identifier = "K" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    candidate = Candidate(
        identifier, g, boxes, route, prep, elapsed, energy, soc,
        physics.charge_time_s(g, soc), latest, delivery,
    )
    return (candidate, phases) if with_phases else candidate


def candidate_orders(sites):
    sites = tuple(sorted(sites))
    require(
        len(sites) <= EXACT_ORDER_LIMIT,
        "当前代码仅枚举短路线；请勿在未补充长路线算法时提高max_stops",
    )
    return itertools.permutations(sites)


def generate_pool(physics, max_stops, previous_pool, logger):
    """
    单箱、双箱短路线 + 确定性贪心插入扩充。
    所有访问顺序在允许的短路线范围内枚举。
    不是穷举全部货箱子集；生成限制在配置和日志中公开。
    """
    d = physics.data
    pool = dict(previous_pool)

    def add(g, boxes):
        sites = {d.service_area_of[b] for b in boxes}
        if len(sites) > max_stops or len(boxes) > MAX_BOXES_PER_CANDIDATE:
            return []
        if sum(d.cargo_weight_kg[b] for b in boxes) > d.max_payload_kg[g] + 1e-9:
            return []
        if sum(d.cargo_volume_m3[b] for b in boxes) > d.max_volume_m3[g] + 1e-9:
            return []
        accepted = []
        for order in candidate_orders(sites):
            c = evaluate_candidate(physics, g, boxes, order)
            if c is not None:
                pool[c.candidate_id] = c
                accepted.append(c)
        return accepted

    box_ids = sorted(d.cargo)
    for g in sorted(d.uav_types):
        for b in box_ids:
            add(g, [b])
        for pair in itertools.combinations(box_ids, 2):
            add(g, pair)

        for index, seed in enumerate(box_ids):
            origin = d.service_area_of[seed]
            other = [b for b in box_ids if b != seed]

            def dist(b):
                target = d.service_area_of[b]
                return (
                    0.0 if origin == target
                    else physics.horizontal_distance_m[origin, target]
                )

            rules = [
                lambda b: (
                    d.service_area_of[b] != origin,
                    dist(b),
                    d.hard_deadline_s[b] or math.inf,
                    b,
                ),
                lambda b: (
                    d.hard_deadline_s[b] or math.inf,
                    dist(b),
                    -d.priority_weight[b],
                    b,
                ),
                lambda b: (
                    d.service_area_of[b] != origin,
                    d.cargo_volume_m3[b] / d.max_volume_m3[g]
                    + d.cargo_weight_kg[b] / d.max_payload_kg[g],
                    dist(b),
                    b,
                ),
            ]
            for rule in rules:
                selected = [seed]
                for b in sorted(other, key=rule):
                    if len(selected) >= MAX_BOXES_PER_CANDIDATE:
                        break
                    if add(g, selected + [b]):
                        selected.append(b)
            if index % 20 == 0:
                logger.info(
                    "候选生成：机型%s，种子%d/%d，累计%d",
                    g, index + 1, len(box_ids), len(pool),
                )
    return pool


def choose_candidates(pool, budget, protected_ids):
    """
    保留上轮全部候选和每箱每机型可行单箱候选。
    其余按货箱轮转取候选，避免全局排序只保留少数服务区。
    """
    chosen = set(protected_ids)
    chosen.update(
        c.candidate_id for c in pool.values() if len(c.boxes) == 1
    )
    by_box = defaultdict(list)
    for c in pool.values():
        for b in c.boxes:
            by_box[b].append(c)

    for b, values in by_box.items():
        # 综合使用交付、持续时间与能耗排序轮换，不改变主目标。
        values.sort(key=lambda c: (
            c.duration_s / len(c.boxes),
            c.delivery_offset_s[b],
            c.energy_kwh / len(c.boxes),
            c.candidate_id,
        ))

    target = max(budget, len(chosen))
    rank = 0
    while len(chosen) < target:
        changed = False
        for b in sorted(by_box):
            values = by_box[b]
            if rank < len(values):
                c = values[rank]
                if c.candidate_id not in chosen:
                    chosen.add(c.candidate_id)
                    changed = True
                    if len(chosen) >= target:
                        break
        if not changed and all(rank >= len(v) - 1 for v in by_box.values()):
            break
        rank += 1

    return [pool[k] for k in sorted(chosen)]
