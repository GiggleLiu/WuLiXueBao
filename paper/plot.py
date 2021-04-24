#!/usr/bin/env python
import fire
from plotlib import *
from viznet import *
import numpy as np
from scipy import optimize, special
import json
from matplotlib.font_manager import FontProperties
from fontTools.ttLib import TTFont
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
ChineseFont2 = FontProperties(fname='/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
font = TTFont("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", fontNumber=0)
font.save("wqy-microhei.ttc")
ChineseFont2 = FontProperties(fname='wqy-microhei.ttc')

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
        with DataPlt(filename="fig2.%s"%tp, figsize=(7,4)) as dp:
            ax1 = plt.subplot(121)
            errors = np.loadtxt("lorenz/data/errors.dat")
            xs = np.arange(len(errors))
            plt.plot(xs, errors, lw=1.5)
            plt.xlabel(u"积分步数", fontproperties = ChineseFont2, fontsize=14)
            plt.ylabel(u"相对误差", fontproperties = ChineseFont2, fontsize=14)
            plt.yscale("log")
            plt.xlim(0,1000)
            ax2 = plt.subplot(122)
            data = np.loadtxt("lorenz/data/neuralode_checkpoint.dat")
            xs = data[:,0]
            ys = data[:,1]
            plt.plot(xs, ys, lw=1.5, marker="o")
            plt.xlabel(u"检查点步长", fontproperties = ChineseFont2, fontsize=14)
            plt.ylabel(u"相对误差", fontproperties = ChineseFont2, fontsize=14)
            plt.xlim(0,500)
            plt.yscale("log")

            plt.tight_layout()

    def fig3(self, tp="pdf"):
        FONTSIZE = 20
        LW = 1.3
        node = NodeBrush("basic", color='none', lw=0, size=0.25)
        sq = NodeBrush("tn.mps", color='black', lw=0, size=0.1)
        node2 = NodeBrush("basic", color='none', lw=LW, edgecolor='none', size=0.001)
        edge = EdgeBrush('->-', lw=LW)
        edge_ = EdgeBrush('-<-', lw=LW)
        edge2 = EdgeBrush('->.', lw=LW)
        dashed = EdgeBrush('.>.', lw=LW)
        node >> (2.0, 0.0)
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"],
            'text.latex.preamble': r'\usepackage{dsfont}'
            })
        with DynamicShow((10,3), 'fig3.%s'%tp) as ds:
            def seq(x, y, edge, labels, color, n):
                nodes = [node >> (x+i, y) for i in range(n)]
                for i in range(n-1):
                    edge >> (nodes[i], nodes[i+1])
                for nd, label in zip(nodes, labels):
                    nd.text(label, fontsize=FONTSIZE, color=color)
                return nodes

            x = 5.5
            y = -1.5
            labels = [r"$s_%d$"%i for i in range(4)] + [r"$\mathcal{L}$"]
            plt.text(-0.5, 0.5, "(a)", fontsize=18)
            n1s = seq(0, 0, edge, labels, 'k', 5)
            labels_fd = [r"$\mathds{1}$"] + [r"$\frac{d {s}_{%d}}{d s_0}$"%i for i in range(1,4)] + [r"$\frac{d \mathcal{L}}{d s_0}$"]
            n2s = seq(0, y, edge, labels_fd, 'k', 5)
            for i in range(4):
                edge >> (n1s[i], n2s[i+1])
            #c.text(r"$\frac{d {s}_{i+1}}{d s_0} = {\frac{d{s}_{i+1}}{d s_i}}\frac{d{s}_{i}}{d s_0}$", "top", fontsize=FONTSIZE, color='r', text_offset=0.1)

            plt.text(x-0.5, 0.5, "(b)", fontsize=18)
            n3s = seq(x, 0, edge, labels, 'k', 5)

            labels_bp = [r"$\frac{d \mathcal{L}}{d s_0}$"] + [r"$\frac{d \mathcal{L}}{d s_{%d}}$"%(i+1) for i in range(3)] + [r"$1$", "top"]
            #b.text(r"$ \frac{d \mathcal{L}}{d s_i} = \frac{d \mathcal{L}}{d s_{i+1}}{\frac{d{s}_{i+1}}{d s_i}}$", "top", fontsize=FONTSIZE, color='r', text_offset=0.1)
            n4s = seq(x, y, edge_, labels_bp, 'k', 5)
            for i in range(4):
                si = sq >> [x+i, y/2]
                edge >> (n3s[i], si)
                edge >> (si, n4s[i])

    def fig4(self, tp='pdf'):
        fname_line="./lorenz_line.dat"
        fname="./lorenz_grad"
        with DataPlt(filename="fig4.%s"%tp, figsize=(8,4)) as dp:
            gs = gridspec.GridSpec(ncols=22, nrows=10)
            ax = plt.subplot(gs[1:9,:10])
            cornertex("(a)", ax, offset=(0,0.18))
            mg = np.loadtxt(fname + "_heatmap.dat")
            #vmin, vmax = -5, 15
            #mg[mg<10**vmin] = 10**vmin
            #mg[mg>10**vmax] = 10**vmax
            curve = np.loadtxt(fname + "_curve.dat")
            σs = np.linspace(0,20,mg.shape[0])
            rhos = np.linspace(0,50,mg.shape[1])

            #ax = plt.pcolormesh(σs, rhos, np.log10(mg).T, shading="auto", vmin=vmin-0.1, vmax=vmax+0.1, cmap='inferno')
            ax = plt.pcolormesh(σs, rhos, mg.T, shading="auto", cmap='inferno', vmin=1e-5, vmax=1e15, norm=matplotlib.colors.LogNorm())
            plt.colorbar()
            plt.scatter([10.0], [27.0], color="C0", s=20, lw=1, edgecolor='none')
            plt.scatter([10.0], [15.0], color="C1", s=20, lw=1, edgecolor='none')
            ax.set_edgecolor('face')
            plt.ylim(0,50)
            plt.xlim(0,20)
            curve[σs < 4] = 51
            plt.plot(σs, curve, color="black", lw=2, label="theoretical")
            plt.xlabel(r"$\sigma$")
            plt.ylabel(r"$\rho$")
            ax = plt.subplot(gs[:,11:], projection='3d')
            mg = np.loadtxt(fname_line)
            ax.plot(mg[0,:], mg[1,:], mg[2,:], label="not stable", lw=1, color="C0")
            ax.plot(mg[3,:], mg[4,:], mg[5,:], label="stable", lw=1, color="C1")
            ax.text2D(0.05, 0.95, "(b)", transform=ax.transAxes, fontsize=16)
            plt.legend()
            plt.tight_layout()

    def fig5(self, tp="pdf", lang="CN"):
        fname1="../data/cuda-gradient-bennett.dat"
        fname2="../data/cuda-gradient-treeverse.dat"
        with DataPlt(filename="fig5.%s"%tp, figsize=(6,4)) as dp:
            mg1 = np.loadtxt(fname1)
            mg2 = np.loadtxt(fname2)
            mem1 = mg1[0,:]
            mem2 = mg2[0,:]
            plt.plot(mem1, mg1[1,:], marker="o", label="Bennett")
            plt.plot(mem2, mg2[1,:], marker="x", label="Treeverse + NiLang")
            plt.ylim([0, 1500])
            plt.xlim([4, 400])
            for (x, y, t) in zip(mem1, mg1[1,:], mg1[2,:]):
                plt.annotate("%.2f"%(t/1e4), (x, y+30), va="bottom", ha="center")
            for (x, y, t) in zip(mem2, mg2[1,:], mg2[2,:]):
                plt.annotate("%.2f"%(t/1e4-1), (x, y+30), va="bottom", ha="center")
            plt.xlabel(r"theorerical peak memory/GB" if lang == "EN" else r"理论内存峰值/32MB", fontproperties = ChineseFont2, fontsize=14)
            plt.ylabel(r"time/s" if lang == "EN" else r"时间/秒", fontproperties = ChineseFont2, fontsize=14)
            plt.xscale("log")
            plt.legend()
            plt.tight_layout()

 
fire.Fire(PLT())
