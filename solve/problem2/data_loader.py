# -*- coding: utf-8 -*-
"""第二问模块：data_loader.py。由原单文件按职责拆分。"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from utils import (
    require, number, integer, workbook_rows, records_from_header,
    unique_insert, write_csv,
)


class InputData:
    def __init__(self, data_root, out, logger):
        self.data_root = Path(data_root)
        self.out = Path(out)
        self.logger = logger

        self.nodes = {}
        self.cargo = {}
        self.uav_types = {}
        self.uavs = {}
        self.batteries = {}
        self.source_files = []
        self.data_checks = []

        self._read_nodes()
        self._read_cargo()
        self._read_uavs()
        self._validate()
        self._make_symbol_dictionaries()
        self._export()

    def source(self, name):
        path = (
            self.data_root
            / "无人机应急物资运输基础数据"
            / f"{name}.xlsx"
        )
        require(path.is_file(), f"缺少附件：{path}")
        if path not in self.source_files:
            self.source_files.append(path)
        return path

    def check(self, name, passed, detail):
        self.data_checks.append({
            "check": name,
            "passed": bool(passed),
            "detail": detail,
        })
        require(passed, f"数据检查失败：{name}；{detail}")

    def _read_nodes(self):
        rows = workbook_rows(self.source("调度中心与服务区"), "数据")
        for r in rows:
            if not isinstance(r[0], str):
                continue
            if r[0] != "O01" and not re.fullmatch(r"S\d{3}", r[0]):
                continue
            identifier = r[0]
            lon = number(r[2], f"{identifier}经度")
            lat = number(r[3], f"{identifier}纬度")
            require(-180 <= lon <= 180 and -90 <= lat <= 90,
                    f"{identifier}经纬度越界")
            z = number(r[4], f"{identifier}海拔")
            population = (
                None if identifier == "O01"
                else integer(r[5], f"{identifier}保障人口", positive=True)
            )
            unique_insert(
                self.nodes, identifier,
                {
                    "node_id": identifier,
                    "name": r[1],
                    "longitude_deg": lon,
                    "latitude_deg": lat,
                    "ground_elevation_m": z,
                    "operation_altitude_m": z + (
                        0.0 if identifier == "O01" else 30.0
                    ),
                    "population": population,
                },
                "节点",
            )

    def _read_cargo(self):
        path = self.source("物资需求与配送时限")
        headers = [
            "货箱编号", "服务区编号", "物资类型",
            "单箱质量（kg）", "单箱体积（m³）", "是否首批保障",
            "首批截止时间（s）", "期望送达时间（s）", "应急优先系数",
        ]
        rows = workbook_rows(path, "逐箱货箱清单")
        records = records_from_header(rows, "货箱编号", headers)

        for r in records:
            b = r["货箱编号"]
            require(isinstance(b, str) and b, "货箱编号缺失")
            flag = r["是否首批保障"]
            require(flag in {"是", "否"}, f"{b}首批标志非法")
            first = flag == "是"
            medical = r["物资类型"] == "医疗物资"

            deadline = r["首批截止时间（s）"]
            if first:
                deadline = number(deadline, f"{b}首批截止", positive=True)
            else:
                require(deadline is None, f"{b}非首批箱却填写首批截止")

            expected = number(
                r["期望送达时间（s）"], f"{b}期望时间", positive=True
            )
            hard = []
            if first:
                hard.append(deadline)
            if medical:
                hard.append(expected)

            unique_insert(
                self.cargo, b,
                {
                    "cargo_id": b,
                    "service_area": r["服务区编号"],
                    "cargo_type": r["物资类型"],
                    "cargo_weight_kg": number(
                        r["单箱质量（kg）"], f"{b}质量", positive=True
                    ),
                    "cargo_volume_m3": number(
                        r["单箱体积（m³）"], f"{b}体积", positive=True
                    ),
                    "is_first_batch": int(first),
                    "is_medical": int(medical),
                    "first_deadline_s": deadline,
                    "expected_delivery_s": expected,
                    "priority_weight": number(
                        r["应急优先系数"], f"{b}优先系数", positive=True
                    ),
                    "hard_deadline_s": min(hard) if hard else None,
                },
                "货箱",
            )

        summary_headers = [
            "服务区编号", "物资类型", "总需求箱数", "首批必须送达箱数",
            "单箱质量（kg）", "单箱体积（m³）", "应急优先系数",
            "首批截止时间（s）", "期望送达时间（s）",
        ]
        summary = records_from_header(
            workbook_rows(path, "数据"), "服务区编号", summary_headers
        )
        seen = set()
        for r in summary:
            key = (r["服务区编号"], r["物资类型"])
            require(key not in seen, f"需求汇总键重复：{key}")
            seen.add(key)
            boxes = [
                b for b in self.cargo.values()
                if (b["service_area"], b["cargo_type"]) == key
            ]
            require(len(boxes) == r["总需求箱数"], f"{key}箱数不一致")
            require(
                sum(b["is_first_batch"] for b in boxes)
                == r["首批必须送达箱数"],
                f"{key}首批数量不一致",
            )
            for b in boxes:
                pairs = [
                    ("cargo_weight_kg", "单箱质量（kg）"),
                    ("cargo_volume_m3", "单箱体积（m³）"),
                    ("priority_weight", "应急优先系数"),
                    ("expected_delivery_s", "期望送达时间（s）"),
                ]
                for left, right in pairs:
                    require(
                        abs(b[left] - float(r[right])) < 1e-9,
                        f"{b['cargo_id']}与汇总表字段不一致：{right}",
                    )
                if b["is_first_batch"]:
                    require(
                        b["first_deadline_s"] == r["首批截止时间（s）"],
                        f"{b['cargo_id']}首批截止不一致",
                    )
        require(
            seen == {
                (b["service_area"], b["cargo_type"])
                for b in self.cargo.values()
            },
            "需求汇总表与逐箱表的服务区—类型集合不一致",
        )

    def _read_uavs(self):
        rows = workbook_rows(self.source("运输无人机数据"), "数据")
        header = [
            "机型编号", "机型名称", "含电池空载总质量（kg）",
            "最大载货质量（kg）", "可用装载体积（m³）",
            "计划巡航速度（m/s）", "空载标准航程（m）",
            "满载标准航程（m）", "电池可用能量（kWh）",
            "返航电量下限（%）", "工位固定准备时间（s）",
            "每箱装载时间（s）", "接收点基础交接时间（s）",
            "每箱增加交接时间（s）", "最大爬升速度（m/s）",
            "最大下降速度（m/s）", "爬升能耗效率", "下降能耗效率",
        ]
        code_names = [
            "uav_type", "name", "empty_mass_kg",
            "max_payload_kg", "max_volume_m3", "cruise_speed_mps",
            "empty_range_m", "full_range_m", "usable_energy_kwh",
            "reserve_ratio", "prep_time_s", "load_time_per_box_s",
            "handover_base_s", "handover_per_box_s",
            "climb_speed_mps", "descent_speed_mps",
            "climb_efficiency", "descent_efficiency",
        ]
        for r in records_from_header(rows, "机型编号", header):
            record = dict(zip(code_names, [r[h] for h in header]))
            g = record["uav_type"]
            for key in code_names[2:]:
                record[key] = number(
                    record[key], f"{g}/{key}",
                    nonnegative=(key == "descent_efficiency"),
                    positive=(key != "descent_efficiency"),
                )
            record["reserve_ratio"] /= 100.0
            require(0 < record["reserve_ratio"] < 1, f"{g}安全余量非法")
            require(
                record["empty_range_m"] >= record["full_range_m"],
                f"{g}空满载航程关系异常",
            )
            require(0 < record["climb_efficiency"] <= 1,
                    f"{g}爬升效率异常")
            require(record["descent_efficiency"] == 0,
                    "当前物理规则仅对应题设下降附加能耗为0")
            unique_insert(self.uav_types, g, record, "机型")

        for r in records_from_header(
            rows, "无人机编号", ["无人机编号", "机型编号", "初始位置"]
        ):
            u = r["无人机编号"]
            unique_insert(
                self.uavs, u,
                {
                    "uav_id": u,
                    "uav_type": r["机型编号"],
                    "initial_node": r["初始位置"],
                },
                "实体无人机",
            )

        inventory = records_from_header(
            rows, "机型编号",
            ["机型编号", "共享电池组总数（组）", "等效完全充电时间（s）"],
        )
        inventory_types = set()
        for r in inventory:
            g = r["机型编号"]
            require(g in self.uav_types and g not in inventory_types,
                    f"电池库存机型非法或重复：{g}")
            inventory_types.add(g)
            count = integer(r["共享电池组总数（组）"], f"{g}电池数量", True)
            full = number(
                r["等效完全充电时间（s）"], f"{g}充电时间", positive=True
            )
            self.uav_types[g]["full_charge_time_s"] = full
            for i in range(1, count + 1):
                c = f"{g}{i:02d}"
                self.batteries[c] = {
                    "battery_id": c,
                    "uav_type": g,
                    "initial_soc": 1.0,
                    "full_charge_time_s": full,
                    "id_origin": "由附件库存派生",
                }

    def _validate(self):
        self.check("节点数量", len(self.nodes) == 16, "O01与15个服务区")
        self.check("货箱数量", len(self.cargo) == 80, "80个不可拆货箱")
        self.check("机型集合", set(self.uav_types) == {"A", "B", "C"},
                   "三种运输机型")
        self.check("实体机数量", len(self.uavs) == 8, "8架实体运输机")
        self.check("电池数量", len(self.batteries) == 14, "含初始及备用电池")

        for b in self.cargo.values():
            require(
                b["service_area"] in self.nodes and b["service_area"] != "O01",
                f"{b['cargo_id']}目的地不存在",
            )
        for u in self.uavs.values():
            require(u["uav_type"] in self.uav_types, "实体机机型不存在")
            require(u["initial_node"] == "O01", "实体机初始位置不为O01")

        expected_uavs = {"A": 4, "B": 2, "C": 2}
        expected_batteries = {"A": 6, "B": 4, "C": 4}
        require(
            Counter(u["uav_type"] for u in self.uavs.values())
            == expected_uavs,
            "实体机分机型库存与协议不一致",
        )
        require(
            Counter(c["uav_type"] for c in self.batteries.values())
            == expected_batteries,
            "电池分机型库存与协议不一致",
        )
        self.check("汇总与逐箱一致性", True, "已逐条核对数量和属性")
        self.check("对象主键与外键", True, "读取时检查，无按名称去重")

    def _make_symbol_dictionaries(self):
        # 与00_全局符号表.md建议代码名保持一致。
        node_fields = [
            "longitude_deg", "latitude_deg",
            "ground_elevation_m", "operation_altitude_m",
        ]
        cargo_fields = [
            "cargo_weight_kg", "cargo_volume_m3", "priority_weight",
            "first_deadline_s", "expected_delivery_s",
            "is_first_batch", "is_medical", "hard_deadline_s",
        ]
        type_fields = [
            "empty_mass_kg", "max_payload_kg", "max_volume_m3",
            "cruise_speed_mps", "climb_speed_mps", "descent_speed_mps",
            "empty_range_m", "full_range_m", "usable_energy_kwh",
            "reserve_ratio", "climb_efficiency", "prep_time_s",
            "load_time_per_box_s", "handover_base_s",
            "handover_per_box_s", "full_charge_time_s",
        ]
        for field in node_fields:
            setattr(self, field, {i: r[field] for i, r in self.nodes.items()})
        for field in cargo_fields:
            setattr(self, field, {b: r[field] for b, r in self.cargo.items()})
        for field in type_fields:
            setattr(self, field, {g: r[field] for g, r in self.uav_types.items()})

        self.service_area_of = {
            b: r["service_area"] for b, r in self.cargo.items()
        }
        self.uavs_by_type = {
            g: sorted(u for u, r in self.uavs.items() if r["uav_type"] == g)
            for g in self.uav_types
        }
        self.batteries_by_type = {
            g: sorted(c for c, r in self.batteries.items()
                      if r["uav_type"] == g)
            for g in self.uav_types
        }

    def _export(self):
        for name, records in [
            ("nodes", self.nodes),
            ("cargo", self.cargo),
            ("uav_types", self.uav_types),
            ("uavs", self.uavs),
            ("batteries", self.batteries),
        ]:
            write_csv(self.out / f"processed_{name}.csv", records.values())
        write_csv(self.out / "input_audit.csv", self.data_checks)
        self.logger.info(
            "读取完成：节点%d，货箱%d，无人机%d，电池%d；总质量%.3f kg",
            len(self.nodes), len(self.cargo), len(self.uavs),
            len(self.batteries), sum(self.cargo_weight_kg.values()),
        )
