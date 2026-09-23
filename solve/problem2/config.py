# -*- coding: utf-8 -*-
"""第二问模块：config.py。由原单文件按职责拆分。"""
from __future__ import annotations

from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # Math
DATA_ROOT = PROJECT_ROOT / "problem" / "D题" / "数据"
PROTOCOL_ROOT = PROJECT_ROOT / "solve" / "2"
OUTPUT_ROOT = SCRIPT_DIR / "outputs"
# HiGHS资源冲突增量生成
MAX_CONFLICT_ITERATIONS = 40
HIGHS_SLICE_TIME_S = 30.0
HIGHS_DISP = True

# 每层目标的总预算，包含该层全部冲突补充迭代。
TIME_LIMIT_PER_STAGE_S = 180

CODE_VERSION = "Q2-HiGHS-incremental-resource-MILP-1.1"

# 能耗分项仍需与正式题面/已核验公共物理层核对。
ENERGY_FORMULA_CONFIRMED = True
ENERGY_FORMULA_SOURCE = ("建模补充假设：水平能耗采用可用能量与等效航程比例；"
                         "爬升能耗采用重力势能除以爬升效率。非已核验官方展开式。")

# 候选生成参数，仅限制搜索范围，不是题目物理约束。
# 第二轮继承第一轮全部候选，并允许更多服务区。
CANDIDATE_ROUNDS = [
    {"max_stops": 2, "budget": 500},
    {"max_stops": 3, "budget": 850},
]
MAX_BOXES_PER_CANDIDATE = 10
EXACT_ORDER_LIMIT = 3

# 每轮每层目标的求解时限。
TIME_LIMIT_PER_STAGE_S = 180
THREADS = 0
RANDOM_SEED = 20260923

# 字典序各层容差：
# 加权迟到、完工时间、能耗、架次数。
LEX_TOLERANCES = [1e-5, 1e-4, 1e-7, 0.0]

AUDIT_TIME_TOL_S = 1e-3
AUDIT_ENERGY_TOL_KWH = 1e-6
PHYSICAL_TOL = 1e-9

TRAJECTORY_STEP_S = 10.0
FIGURE_DPI = 300
SHOW_FIGURES = True

# 如自动找不到中文字体，可填写本机字体文件。
CHINESE_FONT_PATH = ""

CODE_VERSION = "Q2-candidate-joint-MILP-1.0"

PROTOCOL_FILES = [
    "00_全局符号表.md",
    "01_全局建模方案.md",
    "03_Q2_多点运输调度方案.md",
    "06_分工与接口清单.md",
]
