# -*- coding: utf-8 -*-
"""第二问模块：physics.py。由原单文件按职责拆分。"""
from __future__ import annotations

import itertools
import math
import numpy as np
import rasterio
from pyproj import Geod

from config import ENERGY_FORMULA_CONFIRMED, PHYSICAL_TOL
from utils import require, write_csv


class Physics:
    def __init__(self, data, out, logger):
        self.data = data
        self.out = out
        self.logger = logger
        paths = sorted(data.data_root.rglob("*.tif"))
        require(len(paths) == 1, "请确保数据目录中只有一份目标DEM tif")
        self.dem_path = paths[0]

        with rasterio.open(self.dem_path) as src:
            require(src.count == 1, "DEM应为单波段")
            require(src.crs is not None and src.crs.to_epsg() == 4326,
                    "DEM坐标系应为EPSG:4326")
            self.dem = src.read(1, masked=True)
            self.transform = src.transform
            self.height, self.width = src.height, src.width
            self.dem_tags = src.tags()

        # GDAL/rasterio返回的transform已统一为像元角点变换。
        # 对PixelIsPoint GeoTIFF不再人工加减半像元。
        require(abs(self.transform.b) < 1e-12
                and abs(self.transform.d) < 1e-12,
                "当前直线像元遍历要求无旋转栅格")
        self.inverse = ~self.transform
        self.geod = Geod(ellps="WGS84")

        self.horizontal_distance_m = {}
        self.max_dem_elevation_m = {}
        self.cruise_altitude_m = {}
        self.climb_m = {}
        self.descent_m = {}
        self.flight_time_s = {}

        self._build_segments()

    def pixel_xy(self, lon, lat):
        return self.inverse * (lon, lat)

    @staticmethod
    def touching_indices(value):
        nearest = round(value)
        if abs(value - nearest) <= 1e-9:
            return [nearest - 1, nearest]
        return [math.floor(value)]

    def segment_cells(self, i, j):
        """按栅格边界切分直线，保留边界接触像元，不用稀疏采样代替。"""
        d = self.data
        x0, y0 = self.pixel_xy(d.longitude_deg[i], d.latitude_deg[i])
        x1, y1 = self.pixel_xy(d.longitude_deg[j], d.latitude_deg[j])

        for x, y in [(x0, y0), (x1, y1)]:
            require(0 <= x < self.width and 0 <= y < self.height,
                    f"节点或航段端点超出DEM：{i}->{j}")

        cuts = [0.0, 1.0]
        for a, b in [(x0, x1), (y0, y1)]:
            if abs(b - a) > 1e-12:
                for edge in range(
                    math.floor(min(a, b)) + 1,
                    math.ceil(max(a, b)),
                ):
                    t = (edge - a) / (b - a)
                    if 0 < t < 1:
                        cuts.append(t)

        cuts = sorted(set(cuts))
        probes = cuts + [
            (a + b) / 2 for a, b in zip(cuts[:-1], cuts[1:])
        ]
        cells = set()
        for t in probes:
            x = x0 + (x1 - x0) * t
            y = y0 + (y1 - y0) * t
            for col in self.touching_indices(x):
                for row in self.touching_indices(y):
                    if 0 <= row < self.height and 0 <= col < self.width:
                        cells.add((row, col))
        require(bool(cells), f"未找到航段像元：{i}->{j}")
        return cells

    def _build_segments(self):
        d = self.data
        rows = []
        node_checks = []
        for i in d.nodes:
            col, row = self.pixel_xy(d.longitude_deg[i], d.latitude_deg[i])
            require(0 <= col < self.width and 0 <= row < self.height,
                    f"节点{i}超出DEM")
            value = self.dem[math.floor(row), math.floor(col)]
            require(not np.ma.is_masked(value) and np.isfinite(value),
                    f"节点{i}落在无效DEM")
            node_checks.append({
                "node_id": i,
                "table_ground_m": d.ground_elevation_m[i],
                "dem_ground_m": float(value),
                "difference_m": d.ground_elevation_m[i] - float(value),
            })

        for i, j in itertools.permutations(d.nodes, 2):
            cells = self.segment_cells(i, j)
            elevations = []
            for row, col in cells:
                z = self.dem[row, col]
                require(not np.ma.is_masked(z) and np.isfinite(z),
                        f"{i}->{j}经过无效DEM，不能插补后直接飞行")
                elevations.append(float(z))
            max_dem = max(elevations)
            cruise = max_dem + 50.0
            up = cruise - d.operation_altitude_m[i]
            down = cruise - d.operation_altitude_m[j]
            require(up >= -1e-8 and down >= -1e-8,
                    f"{i}->{j}巡航海拔低于作业高度，请检查海拔口径")
            _, _, distance = self.geod.inv(
                d.longitude_deg[i], d.latitude_deg[i],
                d.longitude_deg[j], d.latitude_deg[j],
            )
            self.horizontal_distance_m[i, j] = distance
            self.max_dem_elevation_m[i, j] = max_dem
            self.cruise_altitude_m[i, j] = cruise
            self.climb_m[i, j] = max(0.0, up)
            self.descent_m[i, j] = max(0.0, down)

            for g in d.uav_types:
                ft = (
                    self.climb_m[i, j] / d.climb_speed_mps[g]
                    + distance / d.cruise_speed_mps[g]
                    + self.descent_m[i, j] / d.descent_speed_mps[g]
                )
                self.flight_time_s[g, i, j] = ft
                rows.append({
                    "from_node": i,
                    "to_node": j,
                    "distance_m": distance,
                    "max_dem_m": max_dem,
                    "cruise_altitude_m": cruise,
                    "climb_m": self.climb_m[i, j],
                    "descent_m": self.descent_m[i, j],
                    "uav_type": g,
                    "flight_time_s": ft,
                })
        write_csv(self.out / "segment_physics.csv", rows)
        write_csv(self.out / "node_dem_checks.csv", node_checks)
        self.logger.info("已建立240条有向航段、720条机型—航段记录")

    def equivalent_range_m(self, g, q):
        d = self.data
        require(-PHYSICAL_TOL <= q <= d.max_payload_kg[g] + PHYSICAL_TOL,
                f"{g}载荷越界：{q}")
        q = max(0.0, q)
        return (
            d.empty_range_m[g]
            - (d.empty_range_m[g] - d.full_range_m[g])
            * (q / d.max_payload_kg[g]) ** 1.5
        )

    def horizontal_energy_kwh(self, g, i, j, q):
        """
        待核验展开式：
            E_hor = E_use * distance / equivalent_range(q)
        不是已确认的题面引文；必须通过配置区核验门禁。
        """
        return (
            self.data.usable_energy_kwh[g]
            * self.horizontal_distance_m[i, j]
            / self.equivalent_range_m(g, q)
        )

    def climb_energy_kwh(self, g, i, j, q):
        """
        待核验展开式：
            E_up = (empty_mass + payload) * 9.81 * climb
                   / (climb_efficiency * 3.6e6)
        不是已确认的题面引文；必须通过配置区核验门禁。
        """
        d = self.data
        return (
            (d.empty_mass_kg[g] + q) * 9.81 * self.climb_m[i, j]
            / (d.climb_efficiency[g] * 3.6e6)
        )

    def segment_energy_kwh(self, g, i, j, q):
        require(ENERGY_FORMULA_CONFIRMED,
                "能耗分项公式尚未核验，禁止生成运输结果")
        return (
            self.horizontal_energy_kwh(g, i, j, q)
            + self.climb_energy_kwh(g, i, j, q)
        )

    def charge_time_s(self, g, soc):
        require(-1e-9 <= soc <= 1 + 1e-9, f"SOC越界：{soc}")
        soc = min(1.0, max(0.0, soc))
        full = self.data.full_charge_time_s[g]
        if soc < 0.90:
            return full * (0.65 * (0.90 - soc) / 0.90 + 0.35)
        return full * 0.35 * (1.0 - soc) / 0.10

    def position(self, node, altitude=None):
        d = self.data
        return (
            d.longitude_deg[node],
            d.latitude_deg[node],
            d.operation_altitude_m[node] if altitude is None else altitude,
        )
