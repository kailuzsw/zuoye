#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
① RC 低通滤波电路 —— 理论计算 + PySpice(NgSpice) 仿真 + 全部图自动生成
================================================================================
内容:
  1. 理论: 时间常数 tau = RC, 截止频率 fc = 1/(2*pi*R*C), 传递函数 H(jw)
  2. 仿真A: AC 扫描 -> 波特图 -> 插值提取 -3 dB 截止频率
  3. 仿真B: 方波瞬态 -> 输入/输出波形 -> 由 30%/70% 两点解析反解时间常数 tau
  4. 输出: figures/*.png (全部由代码生成) + results/*.txt|csv (README 数据来源)

依赖: PySpice(含 ngspice) + numpy + matplotlib  (不需要 schemdraw, 原理图由本文件自绘)

运行(注意必须用装了 PySpice 的解释器):
  & "C:\\Users\\21204\\AppData\\Local\\Python\\pythoncore-3.14-64\\python.exe" rc_filter.py
"""

from __future__ import annotations

import os
import math
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")                      # 无界面环境
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_V, u_s, u_us

HERE = os.path.dirname(os.path.abspath(__file__))
FIGD = os.path.join(HERE, "figures")
RESD = os.path.join(HERE, "results")
os.makedirs(FIGD, exist_ok=True)
os.makedirs(RESD, exist_ok=True)

# =====================================================================
#  电路参数（自定）
# =====================================================================
R_OHM = 1.0e3        # R1 = 1 kΩ
C_FARAD = 1.0e-6     # C1 = 1 µF
V_HIGH = 1.0         # 方波高电平 1 V (低电平 0 V)
F_SQ = 200.0         # 方波频率 200 Hz -> 半周期 2.5 ms = 2.5·tau
T_STEP = 2.0e-6      # 瞬态步长 2 µs
T_END = 15.0e-3      # 瞬态总时长 15 ms
F_START, F_STOP, NPTS_DEC = 1.0, 1.0e6, 50   # AC 扫描 1 Hz ~ 1 MHz, 50 点/十倍频程

DB3 = 20.0 * math.log10(1.0 / math.sqrt(2.0))   # -3.0103 dB


# =====================================================================
#  极简原理图绘制工具（纯 matplotlib 图元自绘, 不依赖 schemdraw）
# =====================================================================
INK = "#111111"
ACC = "#b00020"


class Sch:
    """在 matplotlib 坐标系里画原理图: 导线/电阻/电容/电源/接地/端口。"""

    LW = 1.8

    def __init__(self, w, h, xlim, ylim):
        self.fig, self.ax = plt.subplots(figsize=(w, h), dpi=150)
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)

    def wire(self, *pts, color=INK, lw=None, ls="-"):
        p = np.asarray(pts, dtype=float).reshape(-1, 2)
        self.ax.plot(p[:, 0], p[:, 1], color=color, lw=lw or self.LW, ls=ls,
                     solid_capstyle="round", zorder=2)

    def dot(self, x, y, r=0.045, color=INK):
        self.ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor="none", zorder=6))

    def label(self, x, y, s, ha="center", va="center", fs=10.5, color=INK, weight="normal"):
        self.ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=color,
                     fontweight=weight, zorder=8)

    def arrow(self, x1, y1, x2, y2, color=INK, lw=1.5, ms=13):
        self.ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=7,
                         arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                         mutation_scale=ms, shrinkA=0, shrinkB=0))

    def res(self, x, y, orient="h", L=1.05, W=0.34, name="", value="", side=None, fs=10):
        """矩形(IEC)电阻, 中心在 (x, y)。"""
        if orient == "h":
            self.ax.add_patch(Rectangle((x - L / 2, y - W / 2), L, W, facecolor="white",
                                        edgecolor=INK, lw=self.LW, zorder=4))
            txt = f"{name}  {value}".strip()
            if (side or "above") == "above":
                self.label(x, y + 0.30, txt, va="bottom", fs=fs)
            else:
                self.label(x, y - 0.30, txt, va="top", fs=fs)
        else:
            self.ax.add_patch(Rectangle((x - W / 2, y - L / 2), W, L, facecolor="white",
                                        edgecolor=INK, lw=self.LW, zorder=4))
            txt = f"{name}\n{value}".strip() if value else name
            if (side or "right") == "right":
                self.label(x + 0.26, y, txt, ha="left", fs=fs)
            else:
                self.label(x - 0.26, y, txt, ha="right", fs=fs)

    def cap(self, x, y, orient="v", gap=0.18, plate=0.62, name="", value="", side=None, fs=10):
        """电容, 中心在 (x, y); 两个极板之间是 gap。"""
        if orient == "v":
            for dy in (+gap / 2, -gap / 2):
                self.wire((x - plate / 2, y + dy), (x + plate / 2, y + dy), lw=2.4)
            txt = f"{name}\n{value}".strip() if value else name
            if (side or "right") == "right":
                self.label(x + plate / 2 + 0.14, y, txt, ha="left", fs=fs)
            else:
                self.label(x - plate / 2 - 0.14, y, txt, ha="right", fs=fs)
        else:
            for dx in (+gap / 2, -gap / 2):
                self.wire((x + dx, y - plate / 2), (x + dx, y + plate / 2), lw=2.4)
            txt = f"{name}  {value}".strip()
            if (side or "above") == "above":
                self.label(x, y + plate / 2 + 0.16, txt, va="bottom", fs=fs)
            else:
                self.label(x, y - plate / 2 - 0.16, txt, va="top", fs=fs)

    def vsource(self, x, y, r=0.42, name="", value="", fs=10):
        """电压源: 圆形符号, 内部 + / - 。"""
        self.ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=INK,
                                 lw=self.LW, zorder=4))
        self.label(x, y + 0.15, "+", fs=12)
        self.label(x, y - 0.17, "\u2212", fs=12)
        if name:
            self.label(x - r - 0.14, y + 0.15, name, ha="right", fs=fs)
        if value:
            self.label(x - r - 0.14, y - 0.18, value, ha="right", fs=fs)

    def gnd(self, x, y):
        """接地符号(向下)。"""
        self.wire((x, y), (x, y - 0.16))
        for i, w in enumerate((0.34, 0.22, 0.10)):
            yy = y - 0.16 - i * 0.10
            self.wire((x - w, yy), (x + w, yy), lw=1.6)

    def port(self, x, y, name="", side="right", fs=11):
        """开路端口(空心小圆)。"""
        self.ax.add_patch(Circle((x, y), 0.055, facecolor="white", edgecolor=INK,
                                 lw=1.6, zorder=6))
        dx = {"right": 0.22, "left": -0.22, "up": 0.0, "down": 0.0}[side]
        dy = {"up": 0.24, "down": -0.24}.get(side, 0.0)
        ha = {"right": "left", "left": "right", "up": "center", "down": "center"}[side]
        self.label(x + dx, y + dy, name, ha=ha, fs=fs)

    def save(self, path):
        self.fig.tight_layout()
        self.fig.savefig(path, facecolor="white")
        plt.close(self.fig)
        print(f"    [figure] {os.path.relpath(path, HERE)}")


# =====================================================================
#  1. 理论计算
# =====================================================================
def theory():
    tau = R_OHM * C_FARAD
    fc = 1.0 / (2.0 * math.pi * R_OHM * C_FARAD)
    return dict(tau=tau, fc=fc)


def H_theory(f, tau):
    """H(jw) = 1 / (1 + j*2*pi*f*tau)"""
    w = 2.0 * np.pi * np.asarray(f, dtype=float)
    return 1.0 / (1.0 + 1j * w * tau)


# =====================================================================
#  2. 仿真
# =====================================================================
def assert_netlist(circuit: Circuit, expect: list[str]):
    """打印网表并核对关键元件值, 防止单位/缩放写错(见 README 踩坑记录)。"""
    text = str(circuit)
    print("    ---- 实际网表 ----")
    for line in text.strip().splitlines():
        print("    " + line)
    for token in expect:
        if token not in text:
            raise SystemExit(f"[NETLIST CHECK FAILED] 网表里找不到 {token!r}\n{text}")


def build_ac_circuit() -> Circuit:
    c = Circuit("RC low-pass filter (AC)")
    # 用浮点 SI 单位直接给值(欧姆/法拉), 规避单位前缀缩放问题
    c.V("1", "vin", c.gnd, "DC 0V AC 1V")
    c.R("1", "vin", "vout", R_OHM)
    c.C("1", "vout", c.gnd, C_FARAD)
    return c


def build_tran_circuit() -> Circuit:
    period = 1.0 / F_SQ
    c = Circuit("RC low-pass filter (transient, square wave)")
    c.PulseVoltageSource("1", "vin", c.gnd,
                         initial_value=0 @ u_V, pulsed_value=V_HIGH @ u_V,
                         pulse_width=(period / 2) @ u_s, period=period @ u_s,
                         delay_time=0 @ u_s, rise_time=1 @ u_us, fall_time=1 @ u_us)
    c.R("1", "vin", "vout", R_OHM)
    c.C("1", "vout", c.gnd, C_FARAD)
    return c


def run_ac():
    print("\n[仿真A] AC 小信号扫描 ...")
    c = build_ac_circuit()
    assert_netlist(c, ["R1 vin vout 1000.0", "C1 vout 0 1e-06", "AC 1V"])
    sim = c.simulator()
    an = sim.ac(start_frequency=F_START, stop_frequency=F_STOP,
                number_of_points=NPTS_DEC, variation="dec")
    f = np.asarray(an.frequency, dtype=float)
    vout = np.asarray(an["vout"], dtype=complex)
    vin = np.asarray(an["vin"], dtype=complex)
    H = vout / vin
    mag_db = 20.0 * np.log10(np.abs(H))
    phase = np.degrees(np.unwrap(np.angle(H)))
    print(f"    频点数 = {len(f)}  ({f[0]:g} Hz ~ {f[-1]:g} Hz)")
    return dict(f=f, H=H, mag_db=mag_db, phase=phase)


def interp_logf_crossing(f, y, level):
    """在 log10(f) 上线性插值, 求 y = level 的全部交点频率。"""
    lf = np.log10(np.asarray(f, dtype=float))
    d = np.asarray(y, dtype=float) - level
    idx = np.nonzero(d[:-1] * d[1:] < 0)[0]
    out = []
    for i in idx:
        t = d[i] / (d[i] - d[i + 1])
        out.append(10.0 ** (lf[i] + t * (lf[i + 1] - lf[i])))
    return np.asarray(out)


def run_tran():
    print("\n[仿真B] 方波瞬态 ...")
    c = build_tran_circuit()
    assert_netlist(c, ["R1 vin vout 1000.0", "C1 vout 0 1e-06", "PULSE("])
    sim = c.simulator()
    an = sim.transient(step_time=T_STEP, end_time=T_END)
    t = np.asarray(an.time, dtype=float)
    vo = np.asarray(an["vout"], dtype=float)
    vi = np.asarray(an["vin"], dtype=float)
    print(f"    时间点数 = {len(t)}  步长设定 = {T_STEP:g} s  结束 = {t[-1]:g} s")
    return dict(t=t, vin=vi, vout=vo)


def edge_tau(t, v, t0, t1, v_inf):
    """在 [t0, t1] 窗口内, 用 v_inf 为渐近值, 取 30%/70% 两点解析反解时间常数。

    充电/放电都满足 v(t) = v_inf - (v_inf - v0)*exp(-(t-t0)/tau)
    => tau = (t2 - t1) / ln((v_inf - v1)/(v_inf - v2))
    """
    m = (t >= t0) & (t <= t1)
    tt, vv = t[m], v[m]
    v0 = float(vv[0])
    span = v_inf - v0
    lv1 = v0 + 0.30 * span
    lv2 = v0 + 0.70 * span

    def cross(level):
        d = vv - level
        i = np.nonzero(d[:-1] * d[1:] <= 0)[0]
        if len(i) == 0:
            return None
        i = i[0]
        return tt[i] + (tt[i + 1] - tt[i]) * (d[i] / (d[i] - d[i + 1]))

    ta, tb = cross(lv1), cross(lv2)
    if ta is None or tb is None:
        return None
    tau = (tb - ta) / math.log((v_inf - lv1) / (v_inf - lv2))
    return dict(v0=v0, t30=ta, t70=tb, tau=tau)


# =====================================================================
#  3. 出图
# =====================================================================
def fig_circuit(path):
    s = Sch(6.8, 3.9, xlim=(0.0, 6.8), ylim=(-0.75, 3.75))
    # 下端地线
    s.wire((0.9, 0.0), (4.6, 0.0))
    # 电压源支路 (x = 0.9)
    s.vsource(0.9, 1.30, r=0.42, name="V1", value="1 V")
    s.wire((0.9, 0.0), (0.9, 0.88))
    s.wire((0.9, 1.72), (0.9, 3.0))
    # 输入节点
    s.wire((0.25, 3.0), (0.9, 3.0))
    s.dot(0.9, 3.0)
    s.port(0.12, 3.0, "v_i", side="left")
    # 串联电阻 R1
    s.wire((0.9, 3.0), (1.62, 3.0))
    s.res(2.15, 3.0, "h", name="R1", value="1 kΩ")
    s.wire((2.68, 3.0), (4.6, 3.0))
    # 并联电容 C1
    s.dot(4.6, 3.0)
    s.cap(4.6, 1.62, "v", name="C1", value="1 µF", side="right")
    s.wire((4.6, 3.0), (4.6, 1.71))
    s.wire((4.6, 1.53), (4.6, 0.0))
    # 输出端口
    s.wire((4.6, 3.0), (6.35, 3.0))
    s.port(6.5, 3.0, "v_o", side="right")
    # 接地符号
    s.gnd(2.7, 0.0)
    s.label(2.7, -0.52, "GND", fs=9.5)
    s.fig.suptitle("Fig.1-1  RC low-pass filter (V1: 1 V square wave / AC 1 V)",
                   fontsize=11.5, y=0.97)
    s.save(path)


def fig_bode(path, ac, th):
    f, mag, ph = ac["f"], ac["mag_db"], ac["phase"]
    fc = th["fc"]
    fc_sim = interp_logf_crossing(f, mag, DB3)
    fc_sim = float(fc_sim[0]) if len(fc_sim) else float("nan")
    mag_fc = float(np.interp(math.log10(fc_sim), np.log10(f), mag))
    ph_fc = float(np.interp(math.log10(fc_sim), np.log10(f), ph))

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.2, 6.4), dpi=150, sharex=True)
    a1.semilogx(f, mag, color="#1f6fb4", lw=2.0, label="simulation (NgSpice)")
    a1.semilogx(f, 20 * np.log10(np.abs(H_theory(f, th["tau"]))), color="#d1495b",
                lw=1.2, ls="--", label="theory  $1/(1+j2\\pi f RC)$")
    a1.axhline(DB3, color="#2e8b57", lw=1.0, ls=":")
    a1.axvline(fc, color="#888888", lw=1.0, ls="--")
    a1.axvline(fc_sim, color="#2e8b57", lw=1.0, ls="-.")
    a1.plot([fc_sim], [DB3], "o", color="#2e8b57", ms=6)
    a1.annotate(f"$f_c$ (theory) = {fc:.2f} Hz", xy=(fc, -1.0), xytext=(fc * 0.9, 2.0),
                fontsize=9.5, color="#555555", ha="right",
                arrowprops=dict(arrowstyle="->", color="#888888", lw=0.9))
    a1.annotate(f"$f_c$ (sim) = {fc_sim:.2f} Hz\n@ {DB3:.3f} dB",
                xy=(fc_sim, DB3), xytext=(fc_sim * 1.15, -1.6), fontsize=9.5,
                color="#2e8b57", arrowprops=dict(arrowstyle="->", color="#2e8b57", lw=0.9))
    a1.set_ylabel("|H|  (dB)")
    a1.set_ylim(-45, 5)
    a1.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    a1.legend(loc="lower left", fontsize=9.5)
    a1.set_title("Fig.1-2  RC low-pass filter: AC sweep (1 Hz ~ 1 MHz, 50 pts/decade)",
                 fontsize=11)

    a2.semilogx(f, ph, color="#1f6fb4", lw=2.0, label="simulation (NgSpice)")
    a2.semilogx(f, np.degrees(np.angle(H_theory(f, th["tau"]))), color="#d1495b",
                lw=1.2, ls="--", label="theory $=-\\arctan(2\\pi f RC)$")
    a2.axhline(-45.0, color="#2e8b57", lw=1.0, ls=":")
    a2.axvline(fc, color="#888888", lw=1.0, ls="--")
    a2.plot([fc_sim], [ph_fc], "o", color="#2e8b57", ms=6)
    a2.annotate(f"{ph_fc:.2f}° @ $f_c$", xy=(fc_sim, ph_fc), xytext=(fc_sim * 1.2, -20),
                fontsize=9.5, color="#2e8b57",
                arrowprops=dict(arrowstyle="->", color="#2e8b57", lw=0.9))
    a2.set_ylabel("phase  (deg)")
    a2.set_xlabel("frequency  (Hz)")
    a2.set_ylim(-100, 5)
    a2.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    a2.legend(loc="lower left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"    [figure] {os.path.relpath(path, HERE)}")
    return dict(fc_sim=fc_sim, mag_fc=mag_fc, ph_fc=ph_fc)


def fig_tran(path, tr, th, tau_rise, tau_fall):
    t, vi, vo = tr["t"], tr["vin"], tr["vout"]
    tau = th["tau"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.3), dpi=150)

    a1.plot(t * 1e3, vi, color="#999999", lw=1.4, label="input square wave 200 Hz")
    a1.plot(t * 1e3, vo, color="#1f6fb4", lw=2.0, label="output $v_o$ (across C1)")
    a1.axhline(0.632, color="#2e8b57", lw=1.0, ls=":")
    a1.text(12.0, 0.655, "63.2 % of 1 V", fontsize=9, color="#2e8b57", ha="right")
    for k in (1, 2, 3):
        a1.axvline(k * tau * 1e3, color="#cccccc", lw=0.8, ls=":")
    a1.set_xlabel("time  (ms)")
    a1.set_ylabel("voltage  (V)")
    a1.set_title("Fig.1-3a  Square-wave transient (2.5·τ per half period)", fontsize=11)
    a1.grid(True, ls=":", lw=0.5, alpha=0.6)
    a1.legend(loc="upper right", fontsize=9.5)

    # 局部放大: 第一个上升沿 + 理论指数曲线
    m = t <= 3.2 * tau
    tt, vv = t[m], vo[m]
    v_th = 1.0 - np.exp(-tt / tau)
    a2.plot(tt * 1e3, vv, color="#1f6fb4", lw=2.2, label="simulation")
    a2.plot(tt * 1e3, v_th, color="#d1495b", lw=1.3, ls="--",
            label=r"theory $1-e^{-t/\tau}$, $\tau_{theory}=1$ ms")
    a2.axhline(0.632, color="#2e8b57", lw=1.0, ls=":")
    a2.plot([tau * 1e3], [1 - math.exp(-1)], "o", color="#2e8b57", ms=6)
    a2.annotate(f"$\\tau_{{sim}}$ (30 %/70 %) = {tau_rise * 1e3:.4f} ms",
                xy=(tau * 1e3, 0.632), xytext=(1.35, 0.30), fontsize=9.5, color="#2e8b57",
                arrowprops=dict(arrowstyle="->", color="#2e8b57", lw=0.9))
    a2.set_xlabel("time  (ms)")
    a2.set_ylabel("voltage  (V)")
    a2.set_ylim(-0.03, 1.05)
    a2.set_title("Fig.1-3b  Rising edge: simulation vs. exponential law", fontsize=11)
    a2.grid(True, ls=":", lw=0.5, alpha=0.6)
    a2.legend(loc="lower right", fontsize=9.5)

    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"    [figure] {os.path.relpath(path, HERE)}")


# =====================================================================
#  4. 主流程
# =====================================================================
def main():
    print("=" * 74)
    print("① RC 低通滤波电路  理论 + PySpice/NgSpice 仿真")
    print("=" * 74)
    print(f"解释器: {sys.executable}")
    print(f"参数: R1 = {R_OHM:g} Ω, C1 = {C_FARAD:g} F, 方波 {F_SQ:g} Hz / {V_HIGH:g} V")

    th = theory()
    print("\n[理论]")
    print(f"    tau = R*C            = {th['tau'] * 1e3:.6f} ms")
    print(f"    fc  = 1/(2*pi*R*C)   = {th['fc']:.6f} Hz")

    ac = run_ac()
    tr = run_tran()

    # -3 dB 截止频率
    fc_sim = interp_logf_crossing(ac["f"], ac["mag_db"], DB3)
    fc_sim = float(fc_sim[0]) if len(fc_sim) else float("nan")
    tau_from_fc = 1.0 / (2.0 * math.pi * fc_sim)
    mag_fc = float(np.interp(math.log10(fc_sim), np.log10(ac["f"]), ac["mag_db"]))
    ph_fc = float(np.interp(math.log10(fc_sim), np.log10(ac["f"]), ac["phase"]))

    # 通带增益 / 高频斜率 / 高频相位
    idx10 = int(np.argmin(np.abs(ac["f"] - 10.0)))
    gain_10 = float(ac["mag_db"][idx10])
    mband = (ac["f"] >= 10 * th["fc"]) & (ac["f"] <= 100 * th["fc"])
    slope = float(np.polyfit(np.log10(ac["f"][mband]), ac["mag_db"][mband], 1)[0])
    idx_hi = int(np.argmin(np.abs(ac["f"] - 100 * th["fc"])))
    ph_hi = float(ac["phase"][idx_hi])

    # 瞬态: 由 30%/70% 两点解析反解 tau
    period = 1.0 / F_SQ
    r_rise = edge_tau(tr["t"], tr["vout"], 0.0, period / 2, V_HIGH)
    r_fall = edge_tau(tr["t"], tr["vout"], period / 2, period, 0.0)
    tau_rise, tau_fall = r_rise["tau"], r_fall["tau"]
    tau_sim = 0.5 * (tau_rise + tau_fall)

    # 出图
    print("\n[出图]")
    fig_circuit(os.path.join(FIGD, "rc_circuit.png"))
    fig_bode(os.path.join(FIGD, "rc_bode.png"), ac, th)
    fig_tran(os.path.join(FIGD, "rc_transient_square.png"), tr, th, tau_rise, tau_fall)

    # 导出 CSV
    with open(os.path.join(RESD, "rc_ac_sweep.csv"), "w", encoding="utf-8", newline="") as fp:
        fp.write("frequency_Hz,mag_dB,phase_deg\n")
        for a, b, c in zip(ac["f"], ac["mag_db"], ac["phase"]):
            fp.write(f"{a:.6e},{b:.6f},{c:.6f}\n")
    with open(os.path.join(RESD, "rc_transient.csv"), "w", encoding="utf-8", newline="") as fp:
        fp.write("time_s,vin_V,vout_V\n")
        for a, b, c in zip(tr["t"], tr["vin"], tr["vout"]):
            fp.write(f"{a:.9e},{b:.6f},{c:.6f}\n")

    # 结果报告
    err_fc = (fc_sim - th["fc"]) / th["fc"] * 100.0
    err_tau = (tau_sim - th["tau"]) / th["tau"] * 100.0
    L = []
    L.append("① RC 低通滤波电路 —— 理论值 vs 仿真值")
    L.append("=" * 62)
    L.append(f"参数: R1 = {R_OHM:g} Ω, C1 = {C_FARAD:g} F")
    L.append(f"AC 扫描: {F_START:g} Hz ~ {F_STOP:g} Hz, {NPTS_DEC} 点/十倍频程, 共 {len(ac['f'])} 点")
    L.append(f"瞬态: 方波 {F_SQ:g} Hz / {V_HIGH:g} V, 步长 {T_STEP:g} s, 时长 {T_END * 1e3:g} ms, "
             f"共 {len(tr['t'])} 点")
    L.append("")
    L.append(f"{'对比项':<34}{'理论值':>16}{'仿真值':>16}{'偏差':>12}")
    L.append("-" * 78)
    rows = [
        ("时间常数 tau", f"{th['tau'] * 1e3:.6f} ms", f"{tau_sim * 1e3:.6f} ms", f"{err_tau:+.4f} %"),
        ("时间常数 tau (由 fc_sim 反算)", f"{th['tau'] * 1e3:.6f} ms",
         f"{tau_from_fc * 1e3:.6f} ms", f"{(tau_from_fc - th['tau']) / th['tau'] * 100:+.4f} %"),
        ("截止频率 fc", f"{th['fc']:.6f} Hz", f"{fc_sim:.6f} Hz", f"{err_fc:+.4f} %"),
        ("fc 处增益", f"{DB3:.4f} dB", f"{mag_fc:.4f} dB", f"{mag_fc - DB3:+.4f} dB"),
        ("fc 处相位", "-45.0000 deg", f"{ph_fc:.4f} deg", f"{ph_fc + 45.0:+.4f} deg"),
        ("通带增益 (10 Hz)", "0.0000 dB", f"{gain_10:.4f} dB", f"{gain_10:+.4f} dB"),
        ("高频滚降斜率 (10fc~100fc)", "-20.0000 dB/dec", f"{slope:.4f} dB/dec",
         f"{slope + 20.0:+.4f} dB/dec"),
        ("高频相位 (100fc)", f"{-math.degrees(math.atan(100)):.4f} deg", f"{ph_hi:.4f} deg",
         f"{ph_hi + math.degrees(math.atan(100)):+.4f} deg"),
    ]
    for a, b, c, d in rows:
        L.append(f"{a:<34}{b:>16}{c:>16}{d:>12}")
    L.append("")
    L.append(f"上升沿 tau (30%/70% 反解) = {tau_rise * 1e3:.6f} ms")
    L.append(f"下降沿 tau (30%/70% 反解) = {tau_fall * 1e3:.6f} ms")
    v_ss = (1 - math.exp(-(1 / F_SQ / 2) / th["tau"])) / (1 - math.exp(-(1 / F_SQ) / th["tau"]))
    L.append(f"瞬态 vout 最大值 = {tr['vout'].max():.6f} V")
    L.append(f"  稳态峰值理论 = (1-e^-2.5)/(1-e^-5) = {v_ss:.6f} V;  "
             f"首周期峰值理论 = {1 - math.exp(-(1 / F_SQ / 2) / th['tau']):.6f} V")
    L.append("")
    L.append("结论: 仿真 -3 dB 截止频率与理论 1/(2*pi*R*C) 一致, 瞬态充电/放电时间常数与")
    L.append("      tau = RC 一致, 说明该电路确实是一阶 RC 低通滤波器。")

    text = "\n".join(L)
    print("\n" + text + "\n")
    with open(os.path.join(RESD, "rc_filter_results.txt"), "w", encoding="utf-8") as fp:
        fp.write(text + "\n")
    print(f"[结果] {os.path.relpath(os.path.join(RESD, 'rc_filter_results.txt'), HERE)}")


if __name__ == "__main__":
    main()
