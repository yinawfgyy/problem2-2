# -*- coding: utf-8 -*-
"""
第二问求解器：
SciPy/HiGHS + 连续时间联合MILP + 资源互斥约束增量生成。

替换原 optimizer.py。

保留调用接口：
    solve_joint_milp(
        physics, candidates, initial, out, label, logger
    )

返回：
    schedule, report

特点：
1. 不依赖 Gurobi 许可证。
2. 保留原模型的候选选择、逐箱交付、无人机、电池、硬时限和目标。
3. 起始模型不展开全部候选对的资源排序。
4. 发现资源冲突后，为相关候选对增加完整的双资源排序约束。
5. 冲突解仅用于发现缺失约束，不作为有效方案。
6. 时间不足时，只能返回已审计的完整可行解。
7. SciPy milp 不提供本代码所需的 MIP-start 接口，因此初始解用于：
   - 保留可行结果；
   - 提供目标上界；
   - 预添加初始方案涉及的候选对约束。
   不宣称已向求解器注入热启动。

本文件未运行、未进行求解测试。
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

import config as cfg
from utils import require, write_csv, raw_metrics
from validation import audit_schedule


# 新参数采用默认值，旧config.py不增加这些字段也能导入本模块。
MAX_CONFLICT_ITERATIONS = int(
    getattr(cfg, "MAX_CONFLICT_ITERATIONS", 40)
)
HIGHS_SLICE_TIME_S = float(
    getattr(cfg, "HIGHS_SLICE_TIME_S", 30.0)
)
HIGHS_DISP = bool(
    getattr(cfg, "HIGHS_DISP", True)
)

# 互斥冲突判定不得宽于最终审计口径。
CONFLICT_TOL_S = min(
    float(getattr(cfg, "AUDIT_TIME_TOL_S", 1e-3)),
    1e-5,
)
PRIMAL_TOL = 1e-5
INTEGER_TOL = 1e-5
CAP_CHECK_TOL = 1e-7

OBJECTIVE_NAMES = [
    "weighted_tardiness",
    "makespan_s",
    "energy_kwh",
    "sorties",
]


# ============================================================
# 1. 稀疏线性模型容器
# ============================================================

class SparseMILP:
    """
    保存变量上下界、整数类型和稀疏线性约束。
    每次补充约束后重新构造SciPy的LinearConstraint。
    不执行任何模型求解之外的启发式搜索。
    """

    def __init__(self):
        self.names = []
        self.lower = []
        self.upper = []
        self.integrality = []

        self.row_indices = []
        self.col_indices = []
        self.coefficients = []
        self.row_lower = []
        self.row_upper = []

    @property
    def nvars(self):
        return len(self.names)

    @property
    def nrows(self):
        return len(self.row_lower)

    def variable(self, name, lower=0.0, upper=np.inf, binary=False):
        index = self.nvars
        self.names.append(name)
        self.lower.append(float(lower))
        self.upper.append(1.0 if binary else float(upper))
        self.integrality.append(1 if binary else 0)
        return index

    def constraint(self, terms, lower=-np.inf, upper=np.inf):
        """
        terms:
            {variable_index: coefficient}
        或:
            [(variable_index, coefficient), ...]

        同一行重复变量会先合并。
        """
        if isinstance(terms, dict):
            terms = terms.items()

        merged = defaultdict(float)
        for index, coefficient in terms:
            merged[int(index)] += float(coefficient)

        row = self.nrows
        for index, coefficient in merged.items():
            if coefficient != 0.0:
                self.row_indices.append(row)
                self.col_indices.append(index)
                self.coefficients.append(coefficient)

        self.row_lower.append(float(lower))
        self.row_upper.append(float(upper))

    def equal(self, terms, value):
        self.constraint(terms, lower=value, upper=value)

    def matrix(self):
        matrix = coo_matrix(
            (
                np.asarray(self.coefficients, dtype=float),
                (
                    np.asarray(self.row_indices, dtype=np.int32),
                    np.asarray(self.col_indices, dtype=np.int32),
                ),
            ),
            shape=(self.nrows, self.nvars),
        ).tocsc()
        matrix.sum_duplicates()
        matrix.sort_indices()
        return matrix

    def solve(self, objective_terms, time_limit_s):
        objective = np.zeros(self.nvars, dtype=float)
        for index, coefficient in objective_terms.items():
            objective[index] = coefficient

        matrix = self.matrix()
        constraints = LinearConstraint(
            matrix,
            np.asarray(self.row_lower, dtype=float),
            np.asarray(self.row_upper, dtype=float),
        )

        result = milp(
            c=objective,
            integrality=np.asarray(self.integrality, dtype=np.int32),
            bounds=Bounds(
                np.asarray(self.lower, dtype=float),
                np.asarray(self.upper, dtype=float),
            ),
            constraints=constraints,
            options={
                "disp": HIGHS_DISP,
                "presolve": True,
                "time_limit": max(0.01, float(time_limit_s)),
                "mip_rel_gap": 0.0,
            },
        )
        return result, matrix

    def check_primal(self, x, matrix):
        """检查求解器返回的数值解，避免直接四舍五入非法解。"""
        if x is None or len(x) != self.nvars:
            return False, "没有完整变量向量"

        x = np.asarray(x, dtype=float)
        if not np.isfinite(x).all():
            return False, "变量中存在非有限数"

        lower = np.asarray(self.lower)
        upper = np.asarray(self.upper)

        if np.any(x < lower - PRIMAL_TOL):
            return False, "变量违反下界"
        if np.any(x > upper + PRIMAL_TOL):
            return False, "变量违反上界"

        integer_mask = np.asarray(self.integrality) == 1
        if np.any(
            np.abs(x[integer_mask] - np.rint(x[integer_mask]))
            > INTEGER_TOL
        ):
            return False, "整数变量未满足整数性"

        values = matrix @ x
        row_lower = np.asarray(self.row_lower)
        row_upper = np.asarray(self.row_upper)

        if np.any(values < row_lower - PRIMAL_TOL):
            return False, "线性约束下界残差过大"
        if np.any(values > row_upper + PRIMAL_TOL):
            return False, "线性约束上界残差过大"

        return True, "通过"


# ============================================================
# 2. 完整资源冲突检测
# ============================================================

def find_resource_conflicts(schedule):
    """
    扫描所有实际重叠，不只检查相邻区间。

    无人机：
        [准备开始, 返航)

    电池：
        [起飞, 返航后充满)

    返回：
        {(较小候选索引, 较大候选索引), ...}
    """
    uav_intervals = defaultdict(list)
    battery_intervals = defaultdict(list)

    for item in schedule:
        k = item["_candidate_index"]
        c = item["candidate"]
        start = item["start_s"]
        end = start + c.duration_s

        uav_intervals[item["uav_id"]].append(
            (start, end, k)
        )
        battery_intervals[item["battery_id"]].append(
            (start + c.prep_s, end + c.charge_s, k)
        )

    pairs = set()

    for groups in (uav_intervals, battery_intervals):
        for intervals in groups.values():
            intervals.sort(key=lambda row: (row[0], row[1], row[2]))
            active = []

            for start, end, k in intervals:
                active = [
                    row for row in active
                    if row[1] > start + CONFLICT_TOL_S
                ]

                for previous_start, previous_end, l in active:
                    overlap = min(end, previous_end) - max(
                        start, previous_start
                    )
                    if overlap > CONFLICT_TOL_S:
                        pairs.add(tuple(sorted((k, l))))

                active.append((start, end, k))

    return pairs


def remove_internal_fields(schedule):
    return [
        {
            "candidate": item["candidate"],
            "uav_id": item["uav_id"],
            "battery_id": item["battery_id"],
            "start_s": item["start_s"],
        }
        for item in schedule
    ]


def finite_result_value(result, name):
    value = getattr(result, name, None)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


# ============================================================
# 3. 主求解函数
# ============================================================

def solve_joint_milp(
    physics, candidates, initial, out, label, logger
):
    d = physics.data
    out = Path(out)
    count = len(candidates)

    require(count > 0, "候选集合为空")
    require(MAX_CONFLICT_ITERATIONS > 0,
            "MAX_CONFLICT_ITERATIONS必须为正")
    require(HIGHS_SLICE_TIME_S > 0,
            "HIGHS_SLICE_TIME_S必须为正")
    require(len(cfg.LEX_TOLERANCES) == 4,
            "LEX_TOLERANCES应有4个值")
    require(all(float(v) >= 0 for v in cfg.LEX_TOLERANCES),
            "字典序容差不能为负")

    candidate_index = {
        c.candidate_id: k for k, c in enumerate(candidates)
    }
    require(
        len(candidate_index) == count,
        "候选编号重复",
    )

    cover = {b: [] for b in d.cargo}
    for k, c in enumerate(candidates):
        require(c.boxes, "候选不能为空")
        require(c.duration_s >= c.prep_s >= 0,
                f"{c.candidate_id}时间参数异常")
        require(c.charge_s >= 0,
                f"{c.candidate_id}充电时间异常")
        for b in c.boxes:
            require(b in cover, f"候选包含未知货箱：{b}")
            cover[b].append(k)

    for b, indices in cover.items():
        require(indices, f"候选集合未覆盖货箱：{b}")

    # 初始方案必须属于当前候选集并通过完整审计。
    incumbent = None
    if initial:
        converted = []
        for item in initial:
            identifier = item["candidate"].candidate_id
            require(
                identifier in candidate_index,
                f"初始方案候选不在当前集合：{identifier}",
            )
            converted.append({
                "candidate": candidates[candidate_index[identifier]],
                "uav_id": item["uav_id"],
                "battery_id": item["battery_id"],
                "start_s": float(item["start_s"]),
            })

        if audit_schedule(
            physics,
            converted,
            out / f"{label}_initial_audit.csv",
        ):
            incumbent = converted
        else:
            logger.warning(
                "%s：初始方案未通过审计，不用作可行上界", label
            )

    # 保留原模型的保守时域推导。
    largest = sorted(
        (c.duration_s + c.charge_s for c in candidates),
        reverse=True,
    )[:len(d.cargo)]

    horizon = (
        sum(largest) + max(d.expected_delivery_s.values())
    )
    if incumbent:
        horizon = max(
            horizon,
            raw_metrics(physics, incumbent)[1]
            + max(c.charge_s for c in candidates),
        )

    max_charge = max(c.charge_s for c in candidates)
    max_prep = max(c.prep_s for c in candidates)
    resource_big_m = horizon + max_charge + max_prep

    model = SparseMILP()

    # --------------------------------------------------------
    # 3.1 变量
    # --------------------------------------------------------

    trip_selected = {}
    trip_start_s = {}
    trip_return_s = {}
    uav_assignment = {}
    battery_assignment = {}

    for k, c in enumerate(candidates):
        trip_selected[k] = model.variable(
            f"trip_selected[{k}]", binary=True
        )
        trip_start_s[k] = model.variable(
            f"trip_start_s[{k}]", upper=horizon
        )
        trip_return_s[k] = model.variable(
            f"trip_return_s[{k}]", upper=horizon
        )

        for u in d.uavs_by_type[c.uav_type]:
            uav_assignment[k, u] = model.variable(
                f"uav_assignment[{k},{u}]", binary=True
            )
        for bat in d.batteries_by_type[c.uav_type]:
            battery_assignment[k, bat] = model.variable(
                f"battery_assignment[{k},{bat}]", binary=True
            )

    delivery_time_s = {
        b: model.variable(f"delivery_time_s[{b}]", upper=horizon)
        for b in d.cargo
    }
    lateness_s = {
        b: model.variable(f"lateness_s[{b}]", upper=horizon)
        for b in d.cargo
    }
    transport_makespan_s = model.variable(
        "transport_makespan_s", upper=horizon
    )

    # --------------------------------------------------------
    # 3.2 覆盖、时间、兼容性与硬截止
    # --------------------------------------------------------

    for b in d.cargo:
        model.equal(
            {trip_selected[k]: 1.0 for k in cover[b]},
            1.0,
        )

        # L_b >= T_b - D_exp
        model.constraint(
            {
                lateness_s[b]: 1.0,
                delivery_time_s[b]: -1.0,
            },
            lower=-d.expected_delivery_s[b],
        )

        if d.is_first_batch[b]:
            model.constraint(
                {delivery_time_s[b]: 1.0},
                upper=d.first_deadline_s[b],
            )
        if d.is_medical[b]:
            model.constraint(
                {delivery_time_s[b]: 1.0},
                upper=d.expected_delivery_s[b],
            )

    for k, c in enumerate(candidates):
        z = trip_selected[k]
        s = trip_start_s[k]
        e = trip_return_s[k]

        latest = min(
            horizon - c.duration_s,
            c.latest_start_s,
        )
        require(latest >= -1e-8,
                f"候选{k}最晚开始时间为负")
        latest = max(0.0, latest)

        # 未选候选开始与返回均为0。
        model.constraint(
            {s: 1.0, z: -latest},
            upper=0.0,
        )
        model.equal(
            {e: 1.0, s: -1.0, z: -c.duration_s},
            0.0,
        )
        model.constraint(
            {transport_makespan_s: 1.0, e: -1.0},
            lower=0.0,
        )

        model.equal(
            [(uav_assignment[k, u], 1.0)
             for u in d.uavs_by_type[c.uav_type]]
            + [(z, -1.0)],
            0.0,
        )
        model.equal(
            [(battery_assignment[k, bat], 1.0)
             for bat in d.batteries_by_type[c.uav_type]]
            + [(z, -1.0)],
            0.0,
        )

        # SciPy milp不提供Gurobi式指示约束，
        # 这里按已知变量界线性化：
        # z=1 -> T_b = s_k + delta_bk
        #
        # z=0时s_k=0且0<=T_b<=H。
        # delta<=duration<=H，因此M=H足以关闭两条约束。
        for b in c.boxes:
            delta = float(c.delivery_offset_s[b])
            require(0 <= delta <= c.duration_s + 1e-8,
                    f"候选{k}交付偏移越界")

            model.constraint(
                {
                    delivery_time_s[b]: 1.0,
                    s: -1.0,
                    z: horizon,
                },
                upper=delta + horizon,
            )
            model.constraint(
                {
                    delivery_time_s[b]: 1.0,
                    s: -1.0,
                    z: -horizon,
                },
                lower=delta - horizon,
            )

    objectives = [
        {
            lateness_s[b]: float(d.priority_weight[b])
            for b in d.cargo
        },
        {transport_makespan_s: 1.0},
        {
            trip_selected[k]: c.energy_kwh
            for k, c in enumerate(candidates)
        },
        {trip_selected[k]: 1.0 for k in range(count)},
    ]

    # --------------------------------------------------------
    # 3.3 按需增加候选对资源约束
    # --------------------------------------------------------

    active_pairs = set()

    def add_resource_pair(k, l):
        """
        一旦候选对发生冲突，即对该对候选的所有兼容资源加约束。
        防止只换一个资源编号就逃避之前发现的冲突。
        """
        k, l = sorted((k, l))
        if (k, l) in active_pairs:
            return False

        ck, cl = candidates[k], candidates[l]
        if ck.uav_type != cl.uav_type:
            return False

        # 两个候选共享货箱时不可能同时被选。
        if set(ck.boxes) & set(cl.boxes):
            return False

        M = resource_big_m
        y = model.variable(f"uav_order[{k},{l}]", binary=True)
        r = model.variable(f"battery_order[{k},{l}]", binary=True)

        for u in d.uavs_by_type[ck.uav_type]:
            ak = uav_assignment[k, u]
            al = uav_assignment[l, u]

            # s_l >= e_k - M(3-ak-al-y)
            model.constraint(
                {
                    trip_start_s[l]: 1.0,
                    trip_return_s[k]: -1.0,
                    ak: -M,
                    al: -M,
                    y: -M,
                },
                lower=-3.0 * M,
            )

            # s_k >= e_l - M(2-ak-al+y)
            model.constraint(
                {
                    trip_start_s[k]: 1.0,
                    trip_return_s[l]: -1.0,
                    ak: -M,
                    al: -M,
                    y: M,
                },
                lower=-2.0 * M,
            )

        for bat in d.batteries_by_type[ck.uav_type]:
            bk = battery_assignment[k, bat]
            bl = battery_assignment[l, bat]

            # s_l+p_l >= e_k+chi_k - M(3-bk-bl-r)
            model.constraint(
                {
                    trip_start_s[l]: 1.0,
                    trip_return_s[k]: -1.0,
                    bk: -M,
                    bl: -M,
                    r: -M,
                },
                lower=ck.charge_s - cl.prep_s - 3.0 * M,
            )

            # s_k+p_k >= e_l+chi_l - M(2-bk-bl+r)
            model.constraint(
                {
                    trip_start_s[k]: 1.0,
                    trip_return_s[l]: -1.0,
                    bk: -M,
                    bl: -M,
                    r: M,
                },
                lower=cl.charge_s - ck.prep_s - 2.0 * M,
            )

        active_pairs.add((k, l))
        return True

    # 只预添加初始可行计划涉及的候选对，而非全体候选对。
    if incumbent:
        initial_indices = [
            candidate_index[item["candidate"].candidate_id]
            for item in incumbent
        ]
        for position, k in enumerate(initial_indices):
            for l in initial_indices[position + 1:]:
                add_resource_pair(k, l)

    initial_nvars = model.nvars
    initial_nrows = model.nrows

    # --------------------------------------------------------
    # 3.4 提取解
    # --------------------------------------------------------

    def extract_schedule(x):
        schedule = []
        for k, c in enumerate(candidates):
            if x[trip_selected[k]] <= 0.5:
                continue

            selected_uavs = [
                u for u in d.uavs_by_type[c.uav_type]
                if x[uav_assignment[k, u]] > 0.5
            ]
            selected_batteries = [
                bat for bat in d.batteries_by_type[c.uav_type]
                if x[battery_assignment[k, bat]] > 0.5
            ]
            require(
                len(selected_uavs) == len(selected_batteries) == 1,
                "整数解资源分配不唯一",
            )

            start = float(x[trip_start_s[k]])
            require(start >= -PRIMAL_TOL, "开始时刻明显为负")

            schedule.append({
                "_candidate_index": k,
                "candidate": c,
                "uav_id": selected_uavs[0],
                "battery_id": selected_batteries[0],
                "start_s": max(0.0, start),
            })

        return schedule

    logger.info(
        "%s：HiGHS初始模型，候选%d，变量%d，线性约束%d，"
        "已展开候选对%d",
        label, count, model.nvars, model.nrows, len(active_pairs),
    )
    logger.info(
        "%s：THREADS、RANDOM_SEED不传入SciPy milp；"
        "仅使用其公开支持的参数",
        label,
    )

    # --------------------------------------------------------
    # 3.5 分层求解与冲突分离
    # --------------------------------------------------------

    iteration_rows = []
    stage_rows = []
    locked_caps = []
    four_stages_completed = False

    def obeys_previous_caps(metrics):
        return all(
            metrics[index] <= cap + CAP_CHECK_TOL
            for index, cap in enumerate(locked_caps)
        )

    for stage_index in range(4):
        objective_name = OBJECTIVE_NAMES[stage_index]
        objective_terms = objectives[stage_index]
        tolerance = float(cfg.LEX_TOLERANCES[stage_index])

        stage_begin = time.perf_counter()
        stage_deadline = (
            stage_begin + float(cfg.TIME_LIMIT_PER_STAGE_S)
        )

        stage_proven_optimal = False
        stage_best_bound = None
        stage_reason = "time_or_iteration_limit"
        full_feasible_solver_solutions = 0
        calls = 0

        # 已审计方案可作为该层目标上界。
        # 这不是MIP-start，不会把变量解直接传给HiGHS。
        if incumbent:
            current_metrics = raw_metrics(physics, incumbent)
            require(
                obeys_previous_caps(current_metrics),
                "保留方案违反此前字典序上界",
            )
            model.constraint(
                objective_terms,
                upper=current_metrics[stage_index] + tolerance,
            )

        for iteration in range(1, MAX_CONFLICT_ITERATIONS + 1):
            remaining = stage_deadline - time.perf_counter()
            if remaining <= 0.05:
                stage_reason = "stage_time_limit"
                break

            calls += 1
            slice_time = min(HIGHS_SLICE_TIME_S, remaining)

            logger.info(
                "%s：目标%d/%d=%s，冲突迭代%d，变量%d，约束%d，"
                "本次时限%.1f s",
                label, stage_index + 1, 4, objective_name,
                iteration, model.nvars, model.nrows, slice_time,
            )

            result, matrix = model.solve(
                objective_terms, slice_time
            )

            lower_bound = finite_result_value(
                result, "mip_dual_bound"
            )
            if lower_bound is not None:
                stage_best_bound = (
                    lower_bound if stage_best_bound is None
                    else max(stage_best_bound, lower_bound)
                )

            row = {
                "stage": stage_index + 1,
                "objective": objective_name,
                "iteration": iteration,
                "status_code": int(result.status),
                "message": str(result.message),
                "relaxation_objective": finite_result_value(result, "fun"),
                "relaxation_dual_bound": lower_bound,
                "relaxation_mip_gap": finite_result_value(result, "mip_gap"),
                "variables": model.nvars,
                "linear_constraints": model.nrows,
                "expanded_pairs_before": len(active_pairs),
                "conflicting_pairs": None,
                "new_pairs_added": 0,
                "full_feasible": False,
            }

            if result.status == 2:
                row["message"] = (
                    "当前候选集合、已添加约束和已锁定目标上界下不可行；"
                    + str(result.message)
                )
                iteration_rows.append(row)

                # 已知完整可行解满足所有合法增量约束。
                # 若同一模型又报告不可行，不能悄悄忽略。
                if incumbent is not None:
                    raise RuntimeError(
                        "HiGHS报告不可行，但存在已审计且满足目标上界的方案。"
                        "请检查数值尺度和本轮日志，不能将此解释为原题无解。"
                    )
                stage_reason = "restricted_model_infeasible"
                break

            if result.status in (3, 4):
                iteration_rows.append(row)
                stage_reason = (
                    "unbounded_or_solver_error"
                )
                logger.error(
                    "%s：HiGHS状态%d：%s",
                    label, result.status, result.message,
                )
                break

            if result.x is None:
                iteration_rows.append(row)
                stage_reason = "no_integer_solution_in_slice"
                # 不原样重启同一模型消耗剩余预算。
                break

            valid, detail = model.check_primal(result.x, matrix)
            if not valid:
                row["message"] += "；数值解检查失败：" + detail
                iteration_rows.append(row)
                stage_reason = "primal_validation_failed"
                logger.warning("%s：%s", label, detail)
                break

            tentative = extract_schedule(result.x)
            conflicts = find_resource_conflicts(tentative)
            row["conflicting_pairs"] = len(conflicts)

            if conflicts:
                added = sum(
                    add_resource_pair(k, l)
                    for k, l in sorted(conflicts)
                )
                row["new_pairs_added"] = added
                iteration_rows.append(row)

                logger.info(
                    "%s：中间解存在%d对资源冲突，新增%d对约束；"
                    "该中间解不作为结果",
                    label, len(conflicts), added,
                )

                if added == 0:
                    # 已添加完整互斥约束仍出现冲突，属于数值或实现问题。
                    stage_reason = "conflict_in_existing_constraints"
                    logger.error(
                        "%s：冲突涉及已展开候选对，停止本层，保留已审计方案",
                        label,
                    )
                    break
                continue

            clean_schedule = remove_internal_fields(tentative)
            audit_path = (
                out
                / f"{label}_stage{stage_index + 1}"
                  f"_iter{iteration}_audit.csv"
            )
            if not audit_schedule(
                physics, clean_schedule, audit_path
            ):
                row["message"] += "；完整审计失败"
                iteration_rows.append(row)
                raise RuntimeError(
                    f"无明显资源冲突的解未通过完整审计：{audit_path}"
                )

            metrics = raw_metrics(physics, clean_schedule)
            if not obeys_previous_caps(metrics):
                row["message"] += "；真实指标违反前层上界"
                iteration_rows.append(row)
                stage_reason = "previous_objective_cap_violation"
                break

            row["full_feasible"] = True
            full_feasible_solver_solutions += 1
            iteration_rows.append(row)

            if incumbent is None:
                incumbent = clean_schedule
            else:
                old_metrics = raw_metrics(physics, incumbent)
                # 在已锁定前层目标的范围内比较当前及后续目标。
                if tuple(metrics[stage_index:]) < tuple(
                    old_metrics[stage_index:]
                ):
                    incumbent = clean_schedule

            # 松弛模型的最优解若同时满足全部原始资源约束，
            # 则它也是本层完整模型的最优解（数值容差范围内）。
            solver_value = finite_result_value(result, "fun")
            true_value = metrics[stage_index]
            match_tol = 1e-6 * max(1.0, abs(true_value))

            if (
                result.status == 0
                and solver_value is not None
                and abs(true_value - solver_value) <= match_tol
            ):
                stage_proven_optimal = True
                stage_reason = "optimal_and_full_feasible"
            else:
                stage_reason = "full_feasible_not_proven_optimal"

            # 已得到完整可行解；进入下一层。
            # 若本层未证最优，报告中明确标记。
            break

        elapsed = time.perf_counter() - stage_begin

        if incumbent is None:
            stage_rows.append({
                "stage": stage_index + 1,
                "objective": objective_name,
                "reason": stage_reason,
                "objective_value": None,
                "best_bound": stage_best_bound,
                "full_feasible_gap": None,
                "proven_optimal_current_stage": False,
                "runtime_s": elapsed,
                "solver_calls": calls,
                "full_feasible_solver_solutions":
                    full_feasible_solver_solutions,
                "lock_tolerance": tolerance,
            })
            break

        metrics = raw_metrics(physics, incumbent)
        objective_value = metrics[stage_index]

        if stage_best_bound is None:
            feasible_gap = None
        else:
            feasible_gap = max(
                0.0, objective_value - stage_best_bound
            ) / max(1.0, abs(objective_value))

        stage_rows.append({
            "stage": stage_index + 1,
            "objective": objective_name,
            "reason": stage_reason,
            "objective_value": objective_value,
            "best_bound": stage_best_bound,
            "full_feasible_gap": feasible_gap,
            "proven_optimal_current_stage": stage_proven_optimal,
            "runtime_s": elapsed,
            "solver_calls": calls,
            "full_feasible_solver_solutions":
                full_feasible_solver_solutions,
            "lock_tolerance": tolerance,
        })

        logger.info(
            "%s：第%d层结束，完整可行目标值=%.9g，"
            "本层已证最优=%s，原因=%s",
            label, stage_index + 1, objective_value,
            stage_proven_optimal, stage_reason,
        )

        # 按实际路线重算的指标锁定，而不是使用可能松弛的辅助变量值。
        cap = objective_value + tolerance
        model.constraint(objective_terms, upper=cap)
        locked_caps.append(cap)

        if stage_index == 3:
            four_stages_completed = True

    # --------------------------------------------------------
    # 3.6 输出求解过程与最终检查
    # --------------------------------------------------------

    write_csv(
        out / f"{label}_solver_iterations.csv",
        iteration_rows,
    )
    write_csv(
        out / f"{label}_solver_stages.csv",
        stage_rows,
    )
    write_csv(
        out / f"{label}_model_size.csv",
        [{
            "solver": "SciPy/HiGHS",
            "scipy_version": scipy.__version__,
            "candidate_count": count,
            "initial_variables": initial_nvars,
            "initial_constraints": initial_nrows,
            "final_variables": model.nvars,
            "final_constraints": model.nrows,
            "expanded_candidate_pairs": len(active_pairs),
            "horizon_s": horizon,
            "resource_big_m": resource_big_m,
        }],
    )

    if incumbent is not None:
        require(
            audit_schedule(
                physics,
                incumbent,
                out / f"{label}_highs_final_audit.csv",
            ),
            "最终保留方案未通过独立审计",
        )

    report = {
        "label": label,
        "solver": "SciPy/HiGHS",
        "candidate_count": count,
        "four_stages_completed": four_stages_completed,
        "all_stages_proven_optimal": (
            four_stages_completed
            and len(stage_rows) == 4
            and all(
                row["proven_optimal_current_stage"]
                for row in stage_rows
            )
        ),
        "expanded_candidate_pairs": len(active_pairs),
        "scope": (
            "当前候选集合；连续时间；资源互斥约束增量生成；"
            "只有完整审计通过的方案才返回；最优性以分层日志为准"
        ),
    }

    if incumbent is None:
        logger.warning(
            "%s：当前候选和预算内未得到完整可行方案；"
            "不得将未补齐资源约束的中间解作为结果",
            label,
        )

    return incumbent, report