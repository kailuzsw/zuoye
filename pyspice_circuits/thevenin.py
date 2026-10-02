#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
② 验证戴维南定理 —— 理论计算 + PySpice(NgSpice) 仿真 + 全部图自动生成
================================================================================
含源线性二端网络(端口 a-b):
    V1 = 12 V 经 R1 = 4 kΩ 接节点 a
    V2 =  6 V 经 R2 = 6 kΩ 接节点 a
    R3 = 12 kΩ 由节点 a 到地(b)
理论: V_th = V_oc = 8 V, I_sc = 4 mA, R_th = V_oc/I_sc = 2 kΩ
仿真: (1) 开路电压  (2) 短路电流  (3) 负载扫描(原网络 vs 等效电路)
输出: figures/*.png (代码自动生成) + results/*.txt|csv

运行:
  & "C:\\Users\\21204\\AppData\\Local\\Python\\pythoncore-3.14-64\\python.exe" thevenin.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

from PySpice.Spice.Netlist import Circuit

HERE = os.path.dirname(os.path.abspath(__file__))
FIGD = os.path.join(HERE, "figures")
RESD = os.path.join(HERE, "results")
os.makedirs(FIGD, exist_ok=True)
os.makedirs(RESD, exist_ok=True)

# =====================================================================
#  电路参数（自定）
# =====================================================================
V1, R1 = 12.0, 4.0e3
V2, R2 = 6.0, 6.0e3
R3 = 12.0e3
RL_LIST = [0.5e3, 1.0e3, 2.0e3, 5.0e3, 10.0e3, 100.0e3]   # 负载扫描点
R_VOLTMETER = 1.0e9    # 仅用于“电压表”读数演示


# =====================================================================
#  极简原理图绘制工具（纯 matplotlib 图元自绘, 不依赖 schemdraw）
# =====================================================================
INK = "#111111"
ACC = "#b00020"
GREY = "#9a9a9a"


class Sch:
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

    def res(self, x, y, orient="h", L=1.0, W=0.34, name="", value="", side=None,
            fs=10, color=INK):
        if orient == "h":
            self.ax.add_patch(Rectangle((x - L / 2, y - W / 2), L, W, facecolor="white",
                                        edgecolor=color, lw=self.LW, zorder=4))
            txt = f"{name}  {value}".strip()
            if (side or "above") == "above":
                self.label(x, y + 0.28, txt, va="bottom", fs=fs, color=color)
            else:
                self.label(x, y - 0.28, txt, va="top", fs=fs, color=color)
        else:
            self.ax.add_patch(Rectangle((x - W / 2, y - L / 2), W, L, facecolor="white",
                                        edgecolor=color, lw=self.LW, zorder=4))
            txt = f"{name}\n{value}".strip() if value else name
            if (side or "right") == "right":
                self.label(x + 0.26, y, txt, ha="left", fs=fs, color=color)
            else:
                self.label(x - 0.26, y, txt, ha="right", fs=fs, color=color)

    def vsource(self, x, y, r=0.42, name="", value="", fs=10, color=INK):
        self.ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=color,
                                 lw=self.LW, zorder=4))
        self.label(x, y + 0.15, "+", fs=12, color=color)
        self.label(x, y - 0.17, "\u2212", fs=12, color=color)
        if name:
            self.label(x - r - 0.14, y + 0.15, name, ha="right", fs=fs, color=color)
        if value:
            self.label(x - r - 0.14, y - 0.18, value, ha="right", fs=fs, color=color)

    def gnd(self, x, y):
        self.wire((x, y), (x, y - 0.16))
        for i, w in enumerate((0.34, 0.22, 0.10)):
            yy = y - 0.16 - i * 0.10
            self.wire((x - w, yy), (x + w, yy), lw=1.6)

    def port(self, x, y, name="", side="right", fs=11, color=INK):
        self.ax.add_patch(Circle((x, y), 0.055, facecolor="white", edgecolor=color,
                                 lw=1.6, zorder=6))
        dx = {"right": 0.22, "left": -0.22, "up": 0.0, "down": 0.0}[side]
        dy = {"up": 0.24, "down": -0.24}.get(side, 0.0)
        ha = {"right": "left", "left": "right", "up": "center", "down": "center"}[side]
        self.label(x + dx, y + dy, name, ha=ha, fs=fs, color=color)

    def dashed_box(self, x0, y0, x1, y1, color=GREY):
        self.ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                    edgecolor=color, lw=1.3, ls=(0, (5, 4)), zorder=1))

    def save(self, path):
        self.fig.tight_layout()
        self.fig.savefig(path, facecolor="white")
        plt.close(self.fig)
        print(f"    [figure] {os.path.relpath(path, HERE)}")


# =====================================================================
#  1. 理论计算
# =====================================================================
def theory():
    # 节点法: (Va-V1)/R1 + (Va-V2)/R2 + Va/R3 = 0
    g1, g2, g3 = 1.0 / R1, 1.0 / R2, 1.0 / R3
    vth = (V1 * g1 + V2 * g2) / (g1 + g2 + g3)
    # 短路电流: a 接地 -> R3 无电流, Isc = V1/R1 + V2/R2
    isc = V1 / R1 + V2 / R2
    rth_from_isc = vth / isc
    rth_parallel = 1.0 / (g1 + g2 + g3)      # 独立源置零后 4k//6k//12k
    pmax = vth ** 2 / (4.0 * rth_parallel)   # 负载 = Rth 时最大功率
    return dict(vth=vth, isc=isc, rth_isc=rth_from_isc, rth=rth_parallel, pmax=pmax)


# =====================================================================
#  2. 仿真
# =====================================================================
def assert_netlist(circuit: Circuit, expect: list[str]):
    text = str(circuit)
    print("    ---- 实际网表 ----")
    for line in text.strip().splitlines():
        print("    " + line)
    for token in expect:
        if token not in text:
            raise SystemExit(f"[NETLIST CHECK FAILED] 网表里找不到 {token!r}\n{text}")


def build_original(load=None):
    """含源线性二端网络; load=None 表示端口开路。"""
    c = Circuit("Thevenin: original two-terminal network")
    c.V("1", "n1", c.gnd, V1)
    c.R("1", "n1", "a", R1)
    c.V("2", "n2", c.gnd, V2)
    c.R("2", "n2", "a", R2)
    c.R("3", "a", c.gnd, R3)
    if load is not None:
        c.R("L", "a", c.gnd, load)
    return c


def build_original_shorted():
    """端口 a-b 短路: 用 0 V 理想电压源当电流表。"""
    c = Circuit("Thevenin: original network, port shorted")
    c.V("1", "n1", c.gnd, V1)
    c.R("1", "n1", "a", R1)
    c.V("2", "n2", c.gnd, V2)
    c.R("2", "n2", "a", R2)
    c.R("3", "a", c.gnd, R3)
    c.V("sc", "a", c.gnd, 0.0)
    return c


def build_equivalent(load=None):
    th = theory()
    c = Circuit("Thevenin equivalent circuit")
    c.V("th", "n1", c.gnd, th["vth"])
    c.R("th", "n1", "a", th["rth"])
    if load is not None:
        c.R("L", "a", c.gnd, load)
    return c


def op_of(circuit: Circuit):
    sim = circuit.simulator()
    return sim.operating_point()


def measure_voc():
    print("\n[仿真1] 开路电压 V_oc (端口 a-b 开路) ...")
    c = build_original(load=None)
    assert_netlist(c, ["V1 n1 0 12.0", "R1 n1 a 4000.0", "V2 n2 0 6.0", "R2 n2 a 6000.0",
                       "R3 a 0 12000.0"])
    an = op_of(c)
    voc = float(np.asarray(an["a"]).ravel()[0])
    # 附: 用 1 GΩ“电压表”再测一次, 说明负载效应可忽略
    c2 = build_original(load=R_VOLTMETER)
    voc_vm = float(np.asarray(op_of(c2)["a"]).ravel()[0])
    print(f"    V_oc(理想开路) = {voc:.9f} V ;  V_oc(1 GΩ 电压表) = {voc_vm:.9f} V")
    return voc, voc_vm


def measure_isc():
    print("\n[仿真2] 短路电流 I_sc (端口 a-b 用 0 V 源短路) ...")
    c = build_original_shorted()
    assert_netlist(c, ["Vsc a 0 0.0"])
    an = op_of(c)
    isc = abs(float(np.asarray(an["vsc"]).ravel()[0]))
    print(f"    I_sc = {isc * 1e3:.9f} mA")
    return isc


def sweep_loads():
    print("\n[仿真3] 负载扫描: 原网络 vs 戴维南等效电路 ...")
    th = theory()
    rows = []
    for rl in RL_LIST:
        c_o = build_original(load=rl)
        vo = float(np.asarray(op_of(c_o)["a"]).ravel()[0])
        c_e = build_equivalent(load=rl)
        ve = float(np.asarray(op_of(c_e)["a"]).ravel()[0])
        i_theory = th["vth"] / (th["rth"] + rl)
        rows.append(dict(
            rl=rl,
            v_theory=th["vth"] * rl / (th["rth"] + rl),
            v_orig=vo, i_orig=vo / rl,
            v_equiv=ve, i_equiv=ve / rl,
            i_theory=i_theory,
        ))
        print(f"    R_L = {rl / 1e3:7.3f} kΩ : 原网络 V = {vo:.9f} V , "
              f"等效电路 V = {ve:.9f} V , 解析 V = {rows[-1]['v_theory']:.9f} V")
    for r in rows:
        r["p_theory"] = r["v_theory"] * r["i_theory"]
        r["p_orig"] = r["v_orig"] * r["i_orig"]
        r["p_equiv"] = r["v_equiv"] * r["i_equiv"]
    return rows


# =====================================================================
#  3. 出图
# =====================================================================
def fig_original(path):
    s = Sch(7.4, 5.4, xlim=(0.0, 7.4), ylim=(-0.9, 4.9))
    # 上下两条母线
    s.wire((1.5, 4.0), (6.6, 4.0))
    s.wire((1.5, 0.0), (6.6, 0.0))
    # 支路1: R1 + V1
    s.wire((1.5, 4.0), (1.5, 3.53))
    s.res(1.5, 3.0, "v", L=1.0, name="R1", value="4 kΩ", side="left")
    s.wire((1.5, 2.5), (1.5, 1.72))
    s.vsource(1.5, 1.30, r=0.42, name="V1", value="12 V")
    s.wire((1.5, 0.88), (1.5, 0.0))
    # 支路2: R3
    s.wire((3.3, 4.0), (3.3, 2.53))
    s.res(3.3, 2.0, "v", L=1.0, name="R3", value="12 kΩ", side="right")
    s.wire((3.3, 1.5), (3.3, 0.0))
    # 支路3: R2 + V2
    s.wire((5.1, 4.0), (5.1, 3.53))
    s.res(5.1, 3.0, "v", L=1.0, name="R2", value="6 kΩ", side="left")
    s.wire((5.1, 2.5), (5.1, 1.72))
    s.vsource(5.1, 1.30, r=0.42, name="V2", value="6 V")
    s.wire((5.1, 0.88), (5.1, 0.0))
    # 节点
    for x in (1.5, 3.3, 5.1):
        s.dot(x, 4.0)
        s.dot(x, 0.0)
    # 端口
    s.wire((6.6, 4.0), (7.0, 4.0))
    s.port(7.12, 4.0, "a (+)", side="right")
    s.port(7.12, 0.0, "b (\u2212)", side="right")
    s.arrow(6.85, 3.80, 6.85, 0.20, color=ACC, ms=12)
    s.label(6.72, 2.0, "v", ha="right", color=ACC)
    s.gnd(2.4, 0.0)
    s.label(2.4, -0.52, "GND", fs=9.5)
    s.dashed_box(0.95, -0.35, 6.05, 4.45)
    s.label(3.5, 4.70, "source-containing linear two-terminal network",
            fs=10.5, color=GREY)
    s.fig.suptitle("Fig.2-1  Original network (V1 = 12 V, V2 = 6 V, "
                   "R1 = 4 kΩ, R2 = 6 kΩ, R3 = 12 kΩ)", fontsize=11.5, y=0.98)
    s.save(path)


def fig_equivalent(path):
    th = theory()
    s = Sch(6.6, 4.4, xlim=(0.0, 6.8), ylim=(-0.9, 3.8))
    s.wire((2.0, 2.90), (4.6, 2.90))
    s.wire((2.0, 0.0), (4.6, 0.0))
    # Vth 串联 Rth
    s.vsource(2.0, 0.95, r=0.42, name="V_th", value=f"{th['vth']:.0f} V")
    s.wire((2.0, 0.0), (2.0, 0.53))
    s.wire((2.0, 1.37), (2.0, 1.95))
    s.res(2.0, 2.42, "v", L=0.95, name="R_th", value=f"{th['rth'] / 1e3:.0f} kΩ", side="left")
    s.wire((2.0, 2.90), (2.0, 2.90))
    s.dot(2.0, 2.90)
    # 负载
    s.wire((4.6, 2.90), (4.6, 2.30))
    s.res(4.6, 1.60, "v", L=1.1, name="R_L", value="(load)", side="left", color=GREY)
    s.wire((4.6, 1.05), (4.6, 0.0))
    s.dot(4.6, 2.90)
    s.dot(4.6, 0.0)
    # 端口
    s.wire((4.6, 2.90), (5.9, 2.90))
    s.wire((4.6, 0.0), (5.9, 0.0))
    s.port(6.02, 2.90, "a (+)", side="right")
    s.port(6.02, 0.0, "b (\u2212)", side="right")
    s.arrow(5.72, 2.70, 5.72, 0.20, color=ACC, ms=12)
    s.label(5.60, 1.45, "v", ha="right", color=ACC)
    s.gnd(3.3, 0.0)
    s.label(3.3, -0.52, "GND", fs=9.5)
    s.label(2.0, 3.42, f"V_th = {th['vth']:.0f} V,  R_th = {th['rth'] / 1e3:.0f} kΩ"
                       f"   (V_th/R_th = {th['isc'] * 1e3:.0f} mA)", fs=10.5)
    s.fig.suptitle("Fig.2-2  Thevenin equivalent, load R_L connected across port a-b",
                   fontsize=11.5, y=0.98)
    s.save(path)


def fig_load_sweep(path, rows, th):
    rl = np.array([r["rl"] for r in rows])
    rlf = np.logspace(math.log10(RL_LIST[0] * 0.4), math.log10(RL_LIST[-1] * 2.5), 400)
    v_th = th["vth"] * rlf / (th["rth"] + rlf)
    i_th = th["vth"] / (th["rth"] + rlf) * 1e3
    p_th = v_th * (th["vth"] / (th["rth"] + rlf)) * 1e3

    v_o = np.array([r["v_orig"] for r in rows])
    v_e = np.array([r["v_equiv"] for r in rows])
    i_o = np.array([r["i_orig"] for r in rows]) * 1e3
    i_e = np.array([r["i_equiv"] for r in rows]) * 1e3
    p_o = np.array([r["p_orig"] for r in rows]) * 1e3
    p_e = np.array([r["p_equiv"] for r in rows]) * 1e3

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.0), dpi=150)

    a = axes[0][0]
    a.semilogx(rlf, v_th, color="#d1495b", lw=1.4, ls="--",
               label=r"theory $V_{th}R_L/(R_{th}+R_L)$")
    a.semilogx(rl, v_o, "o", color="#1f6fb4", ms=7, label="original network (sim)")
    a.semilogx(rl, v_e, "x", color="#2e8b57", ms=8, mew=2, label="Thevenin equivalent (sim)")
    a.axvline(th["rth"], color="#aaaaaa", lw=1.0, ls=":")
    a.set_xlabel(r"load $R_L$  (Ω)")
    a.set_ylabel(r"$V_L$  (V)")
    a.set_title("(a) load voltage", fontsize=10.5)
    a.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    a.legend(fontsize=9, loc="lower right")

    a = axes[0][1]
    a.loglog(rlf, i_th, color="#d1495b", lw=1.4, ls="--",
             label=r"theory $V_{th}/(R_{th}+R_L)$")
    a.loglog(rl, i_o, "o", color="#1f6fb4", ms=7, label="original network (sim)")
    a.loglog(rl, i_e, "x", color="#2e8b57", ms=8, mew=2, label="Thevenin equivalent (sim)")
    a.axvline(th["rth"], color="#aaaaaa", lw=1.0, ls=":")
    a.set_xlabel(r"load $R_L$  (Ω)")
    a.set_ylabel(r"$I_L$  (mA)")
    a.set_title("(b) load current", fontsize=10.5)
    a.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    a.legend(fontsize=9)

    a = axes[1][0]
    a.semilogx(rlf, p_th, color="#d1495b", lw=1.4, ls="--", label="theory $V_L I_L$")
    a.semilogx(rl, p_o, "o", color="#1f6fb4", ms=7, label="original network (sim)")
    a.semilogx(rl, p_e, "x", color="#2e8b57", ms=8, mew=2, label="Thevenin equivalent (sim)")
    a.axvline(th["rth"], color="#aaaaaa", lw=1.0, ls=":")
    a.plot([th["rth"]], [th["pmax"] * 1e3], "*", color="#e08a1e", ms=15,
           label=f"max {th['pmax'] * 1e3:.3f} mW @ $R_L=R_{{th}}$")
    a.set_xlabel(r"load $R_L$  (Ω)")
    a.set_ylabel(r"$P_L$  (mW)")
    a.set_title("(c) load power", fontsize=10.5)
    a.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    a.legend(fontsize=9)

    # 端口伏安特性 (负载线): V = Vth - Rth*I
    a = axes[1][1]
    ii = np.linspace(0, th["vth"] / th["rth"] * 1e3 * 1.05, 100)
    a.plot(ii, th["vth"] - th["rth"] * ii * 1e-3, color="#d1495b", lw=1.4, ls="--",
           label=r"theory $V=V_{th}-R_{th}I$")
    a.plot(i_o, v_o, "o", color="#1f6fb4", ms=7, label="original network (sim)")
    a.plot(i_e, v_e, "x", color="#2e8b57", ms=8, mew=2, label="Thevenin equivalent (sim)")
    a.set_xlabel(r"$I_L$  (mA)")
    a.set_ylabel(r"$V_L$  (V)")
    a.set_title("(d) port I-V characteristic", fontsize=10.5)
    a.grid(True, ls=":", lw=0.5, alpha=0.6)
    a.legend(fontsize=9)

    fig.suptitle("Fig.2-3  Thevenin verification by load sweep "
                 "(original network vs equivalent circuit)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"    [figure] {os.path.relpath(path, HERE)}")


# =====================================================================
#  4. 主流程
# =====================================================================
def main():
    print("=" * 74)
    print("② 验证戴维南定理  理论 + PySpice/NgSpice 仿真")
    print("=" * 74)
    print(f"解释器: {sys.executable}")

    th = theory()
    print("\n[理论]")
    print(f"    V_th = V_oc = {th['vth']:.6f} V   (节点法)")
    print(f"    I_sc        = {th['isc'] * 1e3:.6f} mA  (V1/R1 + V2/R2)")
    print(f"    R_th = V_oc/I_sc        = {th['rth_isc']:.6f} Ω")
    print(f"    R_th = R1//R2//R3       = {th['rth']:.6f} Ω   (两法互证)")
    print(f"    P_max = V_th^2/(4R_th)  = {th['pmax'] * 1e3:.6f} mW")

    voc, voc_vm = measure_voc()
    isc = measure_isc()
    rows = sweep_loads()

    print("\n[出图]")
    fig_original(os.path.join(FIGD, "thevenin_original.png"))
    fig_equivalent(os.path.join(FIGD, "thevenin_equivalent.png"))
    fig_load_sweep(os.path.join(FIGD, "thevenin_load_sweep.png"), rows, th)

    with open(os.path.join(RESD, "thevenin_load_sweep.csv"), "w",
              encoding="utf-8", newline="") as fp:
        fp.write("RL_ohm,V_theory_V,V_original_V,V_equivalent_V,I_theory_mA,"
                 "I_original_mA,I_equivalent_mA,P_theory_mW,P_original_mW,P_equivalent_mW\n")
        for r in rows:
            fp.write(f"{r['rl']:.6f},{r['v_theory']:.9f},{r['v_orig']:.9f},"
                     f"{r['v_equiv']:.9f},{r['i_theory'] * 1e3:.9f},{r['i_orig'] * 1e3:.9f},"
                     f"{r['i_equiv'] * 1e3:.9f},{r['p_theory'] * 1e3:.9f},"
                     f"{r['p_orig'] * 1e3:.9f},{r['p_equiv'] * 1e3:.9f}\n")

    L = []
    L.append("② 戴维南定理 —— 理论值 vs 仿真值")
    L.append("=" * 62)
    L.append(f"网络: V1 = {V1:g} V (R1 = {R1 / 1e3:g} kΩ), V2 = {V2:g} V "
             f"(R2 = {R2 / 1e3:g} kΩ), R3 = {R3 / 1e3:g} kΩ; 端口 a-b")
    L.append("")
    L.append("表1  开路电压 / 短路电流 / 等效内阻")
    L.append(f"{'对比项':<30}{'理论值':>18}{'仿真值':>18}{'偏差':>12}")
    L.append("-" * 78)
    rth_sim = voc / isc
    t1 = [
        ("开路电压 V_oc", f"{th['vth']:.9f} V", f"{voc:.9f} V",
         f"{(voc - th['vth']) / th['vth'] * 100:+.6f} %"),
        ("短路电流 I_sc", f"{th['isc'] * 1e3:.9f} mA", f"{isc * 1e3:.9f} mA",
         f"{(isc - th['isc']) / th['isc'] * 100:+.6f} %"),
        ("等效内阻 R_th = V_oc/I_sc", f"{th['rth']:.6f} Ω", f"{rth_sim:.6f} Ω",
         f"{(rth_sim - th['rth']) / th['rth'] * 100:+.6f} %"),
        ("V_oc (1 GΩ 电压表)", f"{th['vth']:.9f} V", f"{voc_vm:.9f} V",
         f"{(voc_vm - th['vth']) / th['vth'] * 100:+.6f} %"),
    ]
    for a, b, c, d in t1:
        L.append(f"{a:<30}{b:>18}{c:>18}{d:>12}")
    L.append("")
    L.append("表2  等效电路替换后接负载的电压/电流验证")
    L.append(f"{'R_L':>9}{'V理论':>11}{'V原网络':>11}{'V等效':>11}"
             f"{'I理论':>10}{'I原网络':>10}{'I等效':>10}{'P理论':>10}{'P等效':>10}{'':>3}")
    L.append("      单位: R_L Ω, V V, I mA, P mW")
    L.append("-" * 96)
    for r in rows:
        L.append(f"{r['rl']:>9.0f}{r['v_theory']:>11.6f}{r['v_orig']:>11.6f}"
                 f"{r['v_equiv']:>11.6f}{r['i_theory'] * 1e3:>10.6f}"
                 f"{r['i_orig'] * 1e3:>10.6f}{r['i_equiv'] * 1e3:>10.6f}"
                 f"{r['p_theory'] * 1e3:>10.6f}{r['p_equiv'] * 1e3:>10.6f}")
    dv = max(abs(r["v_orig"] - r["v_equiv"]) for r in rows)
    di = max(abs(r["i_orig"] - r["i_equiv"]) for r in rows) * 1e3
    dp = max(abs(r["p_orig"] - r["p_equiv"]) for r in rows) * 1e3
    L.append("")
    L.append(f"原网络与等效电路的最大差异: |ΔV| = {dv:.3e} V, |ΔI| = {di:.3e} mA, "
             f"|ΔP| = {dp:.3e} mW")
    L.append(f"最大功率传输: R_L = R_th = {th['rth'] / 1e3:.3f} kΩ 时 P_max = "
             f"{th['pmax'] * 1e3:.6f} mW (理论)")
    L.append("")
    L.append("结论: 开路电压 8 V、短路电流 4 mA、由两者反算的内阻 2 kΩ 与理论完全一致;")
    L.append("     在 6 个负载点上, 原网络与戴维南等效电路的端电压/电流/功率逐点相同,")
    L.append("     端口伏安特性为同一条直线 V = 8 - 2000·I, 戴维南定理得到验证。")

    text = "\n".join(L)
    print("\n" + text + "\n")
    with open(os.path.join(RESD, "thevenin_results.txt"), "w", encoding="utf-8") as fp:
        fp.write(text + "\n")
    print(f"[结果] {os.path.relpath(os.path.join(RESD, 'thevenin_results.txt'), HERE)}")


if __name__ == "__main__":
    main()
