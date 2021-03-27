#!/usr/bin/env python
import fire
from plotlib import *
from viznet import *
import numpy as np
from scipy import optimize, special
import json

class PLT(object):
    def fig1(self, tp='pdf'):
        T = 32.0
        with DataPlt(filename="fig1.%s"%tp, figsize=(7,4)) as dp:
            for (i,nsteps) in enumerate(10**np.arange(2, 7, 2)):
                time_overhead = np.arange(2.0, T)
                ns = np.array([optimize.root_scalar(lambda x: (2.0-1.0/(nsteps**(1.0/x)))**(x-1)-t, bracket=[1.0, nsteps+1.0]).root for t in time_overhead])
                ds = np.array([optimize.root_scalar(lambda x: special.binom(t+x, x)-nsteps, bracket=[1.0, nsteps+1.0]).root for t in time_overhead])
                ks = nsteps**(1.0/ns)
                ss = ns*(ks-1)

                nstr = "$N = 10^{%d}$"%np.log10(nsteps)

                ax1 = plt.subplot(121)
                plt.plot(time_overhead, ss, label="Bennett (%s)"%nstr, color="C%d"%i)
                plt.plot(time_overhead, ds, label="Treeverse (%s)"%nstr, ls="--", color="C%d"%i)
                plt.ylabel("$S'/S$")
                plt.xlabel("$T'/T$")
                plt.yscale("log")
                plt.ylim(1,nsteps)
                plt.xlim(1,T)
                plt.legend(fontsize=11)

                plt.subplot(122, sharex=ax1)
                plt.plot(time_overhead, ss/ds, label=nstr, color="C%d"%i)
                plt.ylabel("$S_b'/S_t'$")
                plt.xlabel("$T'/T$")
                plt.xlim(1,T)
                plt.legend(fontsize=11)
            ax1.axhline(y=50, color="C1", ls=':')
            plt.tight_layout()

    def fig2(self, tp='pdf'):
        from matplotlib.font_manager import FontProperties
        ChineseFont2 = FontProperties(fname='/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
        with DataPlt(filename="fig2.%s"%tp, figsize=(7,4)) as dp:
            ax1 = plt.subplot(121)
            errors = np.loadtxt("lorentz/data/errors.dat")
            xs = np.arange(len(errors))
            plt.plot(xs, errors, lw=1.5)
            plt.xlabel(u"积分步数", fontproperties = ChineseFont2, fontsize=14)
            plt.ylabel(u"相对误差", fontproperties = ChineseFont2, fontsize=14)
            plt.yscale("log")
            plt.xlim(0,1000)
            ax2 = plt.subplot(122)
            data = np.loadtxt("lorentz/data/neuralode_checkpoint.dat")
            xs = data[:,0]
            ys = data[:,1]
            plt.plot(xs, ys, lw=1.5, marker="o")
            plt.xlabel(u"检查点步长", fontproperties = ChineseFont2, fontsize=14)
            plt.ylabel(u"相对误差", fontproperties = ChineseFont2, fontsize=14)
            plt.xlim(0,500)
            plt.yscale("log")

            plt.tight_layout()

    def fig3(self, tp="pdf"):
        FONTSIZE = 16
        LW = 1.3
        node = NodeBrush("basic", color='none', lw=LW, size=0.2)
        node2 = NodeBrush("basic", color='none', lw=LW, edgecolor='none', size=0.001)
        edge = EdgeBrush('->-', lw=LW)
        edge2 = EdgeBrush('->.', lw=LW)
        dashed = EdgeBrush('.>.', lw=LW)
        node >> (2.0, 0.0)
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"],
            'text.latex.preamble': r'\usepackage{dsfont}'
            })
        with DynamicShow((6,3), 'fig3.%s'%tp) as ds:
            y = 0.0
            plt.text(-0.5, y+0.5, "(a)", fontsize=18)
            a = node >> (0.0, y)
            b = node >> (1.0, y)
            c = node >> (2.0, y)
            d = node >> (3.0, y)
            e = node >> (4.0, y)
            dashed >> (a, b)
            e1 = edge >> (b, c)
            dashed >> (c, d)
            e2 = edge >> (d, e)
            e1.text(r"ODEStep", "top", fontsize=12)
            e2.text(r"loss", "top", fontsize=12)
            a.text(r"$\vec s_0$", fontsize=FONTSIZE)
            b.text(r"$\vec s_{i}$", fontsize=FONTSIZE)
            c.text(r"$\vec s_{i+1}$", fontsize=FONTSIZE)
            d.text(r"$\vec s_{n}$", fontsize=FONTSIZE)
            e.text(r"$\mathcal{L}$", fontsize=FONTSIZE)
            a.text(r"$\mathds{1}$", "top", fontsize=FONTSIZE, text_offset=0.1, color='r')
            b.text(r"$\frac{\partial \vec{s}_{i}}{\partial \vec s_0}$", "top", fontsize=FONTSIZE, color='r', text_offset=0.1)
            c.text(r"$\frac{\partial \vec{s}_{i+1}}{\partial \vec s_0} = {\frac{\partial\vec{s}_{i+1}}{\partial \vec s_i}}\frac{\partial\vec{s}_{i}}{\partial \vec s_0}$", "top", fontsize=FONTSIZE, color='r', text_offset=0.1)
            d.text(r"$\frac{\partial \vec{s}_{n}}{\partial \vec s_0}$", "top", fontsize=FONTSIZE, color='r', text_offset=0.1)
            e.text(r"$\frac{\partial \mathcal{L}}{\partial \vec s_0}$", "top", fontsize=FONTSIZE, color='r', text_offset=0.1)

            y = -1.1
            plt.text(-0.5, y+0.5, "(b)", fontsize=18)
            a = node >> (0.0, y)
            b = node >> (1.0, y)
            c = node >> (2.0, y)
            d = node >> (3.0, y)
            e = node >> (4.0, y)
            dashed >> (b, a)
            e1 = edge >> (c, b)
            dashed >> (d, c)
            e2 = edge >> (d, e)
            e1.text(r"ODEStep", "top", fontsize=12)
            e2.text(r"loss", "top", fontsize=12)
            a.text(r"$\vec s_0$", fontsize=FONTSIZE)
            b.text(r"$\vec s_{i}$", fontsize=FONTSIZE)
            c.text(r"$\vec s_{i+1}$", fontsize=FONTSIZE)
            d.text(r"$\vec s_{n}$", fontsize=FONTSIZE)
            e.text(r"$\mathcal{L}$", fontsize=FONTSIZE)
            a.text(r"$\frac{\partial \mathcal{L}}{\partial \vec s_0}$", "top", fontsize=FONTSIZE, color='r', text_offset=0.1)
            #b.text(r"$\frac{\partial \mathcal{L}}{\partial \vec s_i}$", "top", fontsize=FONTSIZE, color='r', text_offset=0.1)
            b.text(r"$ \frac{\partial \mathcal{L}}{\partial \vec s_i} = \frac{\partial \mathcal{L}}{\partial \vec s_{i+1}}{\frac{\partial\vec{s}_{i+1}}{\partial \vec s_i}}$", "top", fontsize=FONTSIZE, color='r', text_offset=0.1)
            c.text(r"$\frac{\partial \mathcal{L}}{\partial \vec s_{i+1}}$", "top", fontsize=FONTSIZE, color='r', text_offset=0.1)
            d.text(r"$\frac{\partial \mathcal{L}}{\partial \vec s_{n}}$", "top", fontsize=FONTSIZE, color='r', text_offset=0.1)
            e.text(r"$1$", "top", fontsize=FONTSIZE, text_offset=0.1, color='r')

fire.Fire(PLT())
