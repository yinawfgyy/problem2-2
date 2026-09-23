# -*- coding: utf-8 -*-
"""第二问运行入口。统一配置见config.py；能耗分项仍须核验。"""
from __future__ import annotations

import json
import sys
import time
import traceback
from datetime import datetime

from config import (
    OUTPUT_ROOT, CODE_VERSION, DATA_ROOT, PROTOCOL_ROOT,
    ENERGY_FORMULA_CONFIRMED, ENERGY_FORMULA_SOURCE,
    CANDIDATE_ROUNDS, MAX_BOXES_PER_CANDIDATE, EXACT_ORDER_LIMIT,
    TIME_LIMIT_PER_STAGE_S, LEX_TOLERANCES, TRAJECTORY_STEP_S,
)
from utils import init_logging, write_csv, require, raw_metrics
from data_loader import InputData
from physics import Physics
from candidates import generate_pool, choose_candidates
from baseline import greedy_schedule
from optimizer import solve_joint_milp
from validation import audit_schedule
from reporting import candidate_records, export_manifest, export_results
from visualization import make_figures


def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = OUTPUT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=False)
    logger = init_logging(out)
    begin = time.perf_counter()

    configuration = {
        "code_version": CODE_VERSION,
        "run_id": run_id,
        "data_root": str(DATA_ROOT),
        "protocol_root": str(PROTOCOL_ROOT),
        "output_directory": str(out),
        "energy_formula_confirmed": ENERGY_FORMULA_CONFIRMED,
        "energy_formula_source": ENERGY_FORMULA_SOURCE,
        "candidate_rounds": CANDIDATE_ROUNDS,
        "max_boxes_per_candidate": MAX_BOXES_PER_CANDIDATE,
        "exact_order_limit": EXACT_ORDER_LIMIT,
        "time_limit_per_stage_s": TIME_LIMIT_PER_STAGE_S,
        "lex_tolerances": LEX_TOLERANCES,
        "battery_occupation_start": "takeoff",
        "trip_start_definition": "preparation_start",
        "delivery_definition": "handover_complete",
        "alns_enabled": False,
        "trajectory_step_s": TRAJECTORY_STEP_S,
        "command": " ".join(sys.argv),
    }
    write_csv(out / "run_config.csv", [
        {
            "parameter": key,
            "value": json.dumps(value, ensure_ascii=False)
            if isinstance(value, (list, dict)) else value,
        }
        for key, value in configuration.items()
    ])
    write_csv(out / "protocol_alignment.csv", [
        {
            "item": "符号和接口",
            "implementation": "沿用00和06中的变量语义、单位及强制输出字段",
        },
        {
            "item": "ALNS",
            "implementation": "按前文最终方案不启用；03文件该项属于旧方案",
        },
        {
            "item": "实体机与电池",
            "implementation": "分别互斥；电池从起飞到返航充满占用",
        },
        {
            "item": "能耗",
            "implementation": "分项待核验；未确认时不输出运输结果",
        },
        {
            "item": "单点基线",
            "implementation": "本程序重建单点贪心基线，不冒称Q1最优组批",
        },
        {
            "item": "最优性",
            "implementation": "仅在实际生成的候选集合与报告容差范围内",
        },
    ])

    try:
        logger.info("STAGE A：原始数据读取与一致性检查")
        data = InputData(DATA_ROOT, out, logger)

        logger.info("STAGE B：公共航段物理参数")
        physics = Physics(data, out, logger)
        export_manifest(data, physics, out)

        require(
            ENERGY_FORMULA_CONFIRMED and bool(ENERGY_FORMULA_SOURCE.strip()),
            "能耗分项公式尚未完成核验。请核对两个能耗函数，填写"
            "ENERGY_FORMULA_SOURCE，并确认ENERGY_FORMULA_CONFIRMED。"
            "本次仅保存数据检查与航段几何结果，不生成未经确认的运输解。",
        )

        best = None
        best_report = {}
        selected_previous = []
        pool = {}
        experiments = []

        for round_index, setting in enumerate(CANDIDATE_ROUNDS, 1):
            require(setting["max_stops"] <= EXACT_ORDER_LIMIT,
                    "候选服务区上限超过访问顺序枚举上限")
            label = f"round_{round_index}"
            logger.info("STAGE C：%s 候选生成", label)
            pool = generate_pool(
                physics, setting["max_stops"], pool, logger
            )

            protected = {c.candidate_id for c in selected_previous}
            if best:
                protected.update(x["candidate"].candidate_id for x in best)
            candidates = choose_candidates(
                pool, setting["budget"], protected
            )
            selected_previous = candidates
            write_csv(
                out / f"{label}_candidates.csv",
                candidate_records(candidates),
            )
            logger.info(
                "%s：原始池%d，进入联合模型%d", label, len(pool), len(candidates)
            )

            baselines = {}
            for name, single in [
                ("单点贪心基线", True),
                ("多点贪心基线", False),
            ]:
                plan = greedy_schedule(physics, candidates, single_only=single)
                feasible = False
                if plan is not None:
                    feasible = audit_schedule(
                        physics, plan,
                        out / f"{label}_{'single' if single else 'multi'}_audit.csv",
                    )
                if not feasible:
                    plan = None
                baselines[name] = plan
                metrics = raw_metrics(physics, plan)
                experiments.append({
                    "round": round_index,
                    "method": name,
                    "feasible": feasible,
                    "candidate_count": len(candidates),
                    "weighted_tardiness": metrics[0] if metrics else None,
                    "makespan_s": metrics[1] if metrics else None,
                    "energy_kwh": metrics[2] if metrics else None,
                    "sorties": metrics[3] if metrics else None,
                    "all_stages_proven_optimal": False,
                })

            starts = [p for p in [best, *baselines.values()] if p]
            initial = min(
                starts, key=lambda p: raw_metrics(physics, p)
            ) if starts else None

            logger.info("STAGE D：%s 联合资源与时间优化", label)
            plan, report = solve_joint_milp(
                physics, candidates, initial, out, label, logger
            )
            feasible = (
                plan is not None and audit_schedule(
                    physics, plan, out / f"{label}_constraint_audit.csv"
                )
            )
            metrics = raw_metrics(physics, plan) if feasible else None
            experiments.append({
                "round": round_index,
                "method": "HiGHS联合MILP＋资源冲突增量约束",
                "feasible": feasible,
                "candidate_count": len(candidates),
                "weighted_tardiness": metrics[0] if metrics else None,
                "makespan_s": metrics[1] if metrics else None,
                "energy_kwh": metrics[2] if metrics else None,
                "sorties": metrics[3] if metrics else None,
                "all_stages_proven_optimal":
                    report["all_stages_proven_optimal"],
            })
            if plan is not None and not feasible:
                raise RuntimeError(
                    "优化器返回方案未通过独立审计，停止正式输出，"
                    "请检查该轮constraint_audit.csv"
                )
            if feasible:
                # 新轮包含旧轮候选；优先输出最新轮已审计的求解结果，
                # 同时避免超时导致严格指标退化。
                if best is None or raw_metrics(
                    physics, plan
                ) <= raw_metrics(physics, best):
                    best = plan
                    best_report = report

            write_csv(out / "q2_experiment_comparison.csv", experiments)

        require(
            best is not None,
            "当前候选及时间预算内未取得完整可行解。"
            "请查看求解状态并扩充候选/增加时间；不能据此判定原题无解。",
        )
        require(
            audit_schedule(physics, best, out / "q2_constraint_audit.csv"),
            "最终方案未通过约束审计",
        )

        logger.info("STAGE E：导出运输、交付、资源及轨迹")
        exported = export_results(
            physics, best, out, run_id, best_report
        )
        metrics = raw_metrics(physics, best)
        logger.info(
            "最终结果：加权迟到=%.6f，最晚返航=%.6f s，"
            "能耗=%.6f kWh，架次数=%d",
            *metrics,
        )

        write_csv(out / "artifact_manifest.csv", [
            {
                "run_id": run_id,
                "filename": p.name,
                "bytes": p.stat().st_size,
                "code_version": CODE_VERSION,
            }
            for p in sorted(out.iterdir()) if p.is_file()
        ])

        logger.info("STAGE F：保存并显示图件")
        make_figures(physics, exported, out, logger)
        write_csv(out / "artifact_manifest.csv", [
            {
                "run_id": run_id,
                "filename": p.name,
                "bytes": p.stat().st_size,
                "code_version": CODE_VERSION,
            }
            for p in sorted(out.iterdir())
            if p.is_file() and p.name != "artifact_manifest.csv"
        ])
        logger.info(
            "完成，耗时%.2f s；输出目录：%s",
            time.perf_counter() - begin, out,
        )

    except Exception as exc:
        logger.error("停止：%s", exc)
        logger.error(traceback.format_exc())
        write_csv(out / "run_failure.csv", [{
            "run_id": run_id,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "elapsed_s": time.perf_counter() - begin,
        }])
        raise


if __name__ == "__main__":
    main()
