from __future__ import annotations

from dataclasses import dataclass


DELIVERED_FEATURES = [
    "fll_cwt_kf",
    "fsl_cwt_kf",
    "diff_ls_cwt_kf",
    "total_ls_cwt_kf",
    "risk_priority_number",
    "bin_index",
    "dominance",
    "diff_dom_ls_cwt_kf",
    "z_logTotalP",
    "z_sdom",
    "z_fll_cwt_kf",
    "z_fsl_cwt_kf",
]


@dataclass(frozen=True)
class FeatureDescriptor:
    name: str
    title: str
    category: str
    kind: str = "numeric"
    source: str = ""
    processing: str = ""
    meaning: str = ""
    notes: str = ""


FEATURE_DESCRIPTORS: dict[str, FeatureDescriptor] = {
    "fll_cwt_kf": FeatureDescriptor(
        name="fll_cwt_kf",
        title="fll_cwt_kf｜有效多头清算强度",
        category="Canonical liquidation family",
        source="fll_normal",
        processing="对多头清算侧做 trailing wavelet trend extraction、Kalman smoothing 与非负投影。",
        meaning="表达当前有效多头清算压力腿的强度。",
        notes="是 total、diff、risk_priority_number 和标准化特征的基础侧向序列。",
    ),
    "fsl_cwt_kf": FeatureDescriptor(
        name="fsl_cwt_kf",
        title="fsl_cwt_kf｜有效空头清算强度",
        category="Canonical liquidation family",
        source="fsl_normal",
        processing="对空头清算侧做 trailing wavelet trend extraction、Kalman smoothing 与非负投影。",
        meaning="表达当前有效空头清算压力腿的强度。",
        notes="是 total、diff、risk_priority_number 和标准化特征的基础侧向序列。",
    ),
    "diff_ls_cwt_kf": FeatureDescriptor(
        name="diff_ls_cwt_kf",
        title="diff_ls_cwt_kf｜有效净清算差",
        category="Canonical liquidation family",
        source="fll_cwt_kf - fsl_cwt_kf",
        processing="由两条有效侧向清算序列直接相减得到。",
        meaning="表达多头清算与空头清算之间的净方向压力。",
        notes="正值偏 FLL dominant，负值偏 FSL dominant。",
    ),
    "total_ls_cwt_kf": FeatureDescriptor(
        name="total_ls_cwt_kf",
        title="total_ls_cwt_kf｜有效总清算强度",
        category="Canonical liquidation family",
        source="fll_cwt_kf + fsl_cwt_kf",
        processing="由两条有效侧向清算序列直接相加得到。",
        meaning="表达不含方向的总清算压力强度。",
        notes="是 z_logTotalP 的原始有效总量来源。",
    ),
    "risk_priority_number": FeatureDescriptor(
        name="risk_priority_number",
        title="risk_priority_number｜多头清算占比",
        category="Canonical liquidation family",
        source="fll_cwt_kf / total_ls_cwt_kf",
        processing="由 fll_cwt_kf / total_ls_cwt_kf 直接得到，总清算为 0 时定义为 0.5。",
        meaning="表达有效强平中多头清算占总清算的比例。",
        notes="历史模型层别名为 RPN；当前合并输出只保留 risk_priority_number。",
    ),
    "diff_dom_ls_cwt_kf": FeatureDescriptor(
        name="diff_dom_ls_cwt_kf",
        title="diff_dom_ls_cwt_kf｜有效清算方向占优度",
        category="Canonical liquidation family",
        source="diff_ls_cwt_kf / total_ls_cwt_kf",
        processing="由有效净清算差除以有效总清算强度得到。",
        meaning="表达 scale-free 的净方向偏置。",
        notes="与 risk_priority_number 严格仿射等价：diff_dom_ls_cwt_kf = 2 * risk_priority_number - 1。",
    ),
    "z_logTotalP": FeatureDescriptor(
        name="z_logTotalP",
        title="z_logTotalP｜总清算压力标准化",
        category="Liquidation stress + direction (normalized)",
        source="log1p(total_ls_cwt_kf)",
        processing="由 side-first canonical total pressure 经 log1p 压缩后，再做 rolling median/MAD robust z-score。",
        meaning="衡量当前总清算压力相对近期基线有多异常。",
        notes="适合做 stress / regime 识别，不宜单独解释方向。",
    ),
    "z_sdom": FeatureDescriptor(
        name="z_sdom",
        title="z_sdom｜方向主导度标准化",
        category="Liquidation stress + direction (normalized)",
        source="diff_dom_ls_cwt_kf",
        processing="由 canonical sdom 直接做 rolling median/MAD robust z-score。",
        meaning="衡量当前哪一侧强平更占主导，以及该偏置相对近期是否异常。",
        notes="解释时必须结合 z_logTotalP。",
    ),
    "RPN": FeatureDescriptor(
        name="RPN",
        title="RPN｜多头清算占比（历史别名）",
        category="Liquidation stress + direction (normalized)",
        source="risk_priority_number",
        processing="由 fll_cwt_kf / total_ls_cwt_kf 直接得到，总清算为 0 时定义为 0.5。",
        meaning="表达有效强平中多头清算占总清算的比例。",
        notes="历史模型层别名；当前合并输出使用 risk_priority_number。",
    ),
    "bin_index": FeatureDescriptor(
        name="bin_index",
        title="bin_index｜risk_priority_number 有序分箱",
        category="Liquidation stress + direction (normalized)",
        kind="categorical",
        source="risk_priority_number",
        processing="基于 past-only expanding quantile binning 的 9 档有序标签。",
        meaning="离散表达当前 liquidation dominance 处于历史相对高低的哪个区间。",
        notes="它是 ordered label，不应解释为等距距离。",
    ),
    "z_fll_cwt_kf": FeatureDescriptor(
        name="z_fll_cwt_kf",
        title="z_fll_cwt_kf｜多头清算强度标准化",
        category="Liquidation stress + direction (normalized)",
        source="log1p(fll_cwt_kf)",
        processing="对有效多头清算强度做 log1p，再做 rolling median/MAD robust z-score。",
        meaning="衡量 downside deleveraging leg 当前是否异常强。",
        notes="更适合表达风险卸杠杆压力，不是单独方向信号。",
    ),
    "z_fsl_cwt_kf": FeatureDescriptor(
        name="z_fsl_cwt_kf",
        title="z_fsl_cwt_kf｜空头清算强度标准化",
        category="Liquidation stress + direction (normalized)",
        source="log1p(fsl_cwt_kf)",
        processing="对有效空头清算强度做 log1p，再做 rolling median/MAD robust z-score。",
        meaning="衡量 upside squeeze leg 当前是否异常强。",
        notes="常与 short squeeze / bear-market rally 同时出现。",
    ),
    "dominance": FeatureDescriptor(
        name="dominance",
        title="dominance｜清算方向主导状态",
        category="Dominance state (rule-derived)",
        kind="categorical",
        source="bin_index + risk_priority_number + diff_ls_cwt_kf + rolling quantile thresholds",
        processing="先由 past-only rolling quantile 构造 diff 阈值，再结合 bin_index 与 risk_priority_number 的 regime 条件，输出 {-1,0,1} 的离散状态。",
        meaning="表达当前是否处于 FLL dominant / FSL dominant / neutral congestion 的规则状态。",
        notes="当前编码为：1=FSL dominant / upward pressure，-1=FLL dominant / downward pressure，0=neutral。",
    ),
}


def get_feature_descriptor(name: str) -> FeatureDescriptor:
    return FEATURE_DESCRIPTORS.get(
        name,
        FeatureDescriptor(
            name=name,
            title=name,
            category="Other",
            source="",
            processing="",
            meaning=name,
            notes="",
        ),
    )
