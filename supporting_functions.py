#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ---------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------
import os
import gc
import pickle
import numpy as np
import pandas as pd
import pytablewriter
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import gridspec
import seaborn as sns
# ---------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------
pd.set_option('precision', 2)
pd.set_option('float_format', lambda x: '%.2f' % x)
col_rename = {'wlsServer':'server', 'wlsDomain':'domain',  'chttt':'click_time',  'chrnt':'render_time'}
feature_names = ['domain', 'server']
time_names = ['click_time', 'render_time']
# ---------------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------------
def format_names(names):

    if isinstance(names, pd.Index): names = list(names)
    if isinstance(names, str): return names.replace('_', ' ').strip().title()
    names = [s.replace('_', ' ').strip().title() for s in names]
    return names

def format_categories(names):
    if isinstance(names, pd.Index): names = list(names)
    if isinstance(names, str): return names.replace('_', ' ').strip().title()
    names = [s.replace('Server_1', '').replace('Domain', '').strip() for s in names]
    return names

def save_plot(figh=plt.gcf(), bbox='tight', outname='myfig.png', pad=2, tight=True, dpi=300, transparent=False):

    if tight:
        plt.tight_layout(pad=pad)

    figh.savefig(outname, dpi=dpi, bbox_inches='tight',
                 transparent=transparent)
    plt.close()
    print('\nFIGURE SAVED TO: ' + outname)

def df2md(df, precision=2, tablename='table_name', ncol2bold=1):


    dft = df.reset_index().round(precision).copy()
    for i in range(ncol2bold):
        dft.iloc[:,i] = dft.iloc[:,i].apply(lambda x: '**' + x + '**')
    writer = pytablewriter.MarkdownTableWriter()
    writer.header_list = list(dft.columns.values)
    writer.from_dataframe(dft)
    writer.write_table()

def force_dataframe(col):
    if ~isinstance(col, pd.DataFrame):
        col = pd.DataFrame(col)
    return col

def format_yticks(ax):

    tickfmt = ticker.FuncFormatter(lambda x, p: format(int(x), ','))
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    if ax.get_ylim()[1] >= 1000:
        ax.yaxis.set_major_formatter(tickfmt)

def format_xticks(ax):

    tickfmt = ticker.FuncFormatter(lambda x, p: format(int(x), ','))
    ax.xaxis.set_major_locator(ticker.AutoLocator())
    if ax.get_xlim()[1] >= 1000:
        ax.xaxis.set_major_formatter(tickfmt)

def get_color_palette(N=12, as_color_palette=False):

    if N <= 2:
        pal =['27567B', '484D76']
        pal = ['#' + p for p in pal]
    else:
        pal = ['27567B', '484D76', 'CCCCCC', 'BBBBBB', 'AAAAAA']
        pal = ['#' + p for p in pal]

    if as_color_palette:
        return sns.palettes.color_palette(pal[:N])
    else:
        return pal[:N]
    muted_nautical= ['E3CB6B', '484D76', 'CCCCCC', 'BBBBBB', 'AAAAAA']
    muted_neutral = ['27567B', 'B63B3B', 'F9C414', 'F7EBD8', '505050']

def set_sns(scale=1.5):

    sns.set(font_scale=scale, style='ticks', color_codes=True)
    sns.set_palette(get_color_palette())

def plot_hists(y, saveplot=False, bins='auto', normed=True, ylabel='Count', xlabel=None, title=None, xlim=None, percentile=100, footnote=None, cumulative=False, describe_inset=False):

    y = force_dataframe(y)
    ncol = y.shape[1]
    ax = []
    pal = get_color_palette()
    for c in y.columns:
        if percentile < 100:
            footpad = -0.04
            cidx = y[c] > y[c].quantile(percentile/100)
            xlabel = c + \
                '\n(Upper {}% of distribution omitted)'.format(100-percentile)
            s = y.loc[~cidx, c]
        else:
            footpad = -0.01
            s = y[c]
        figh, ax_hist = plt.subplots(figsize=(8, 4))
        ax_hist.hist(s, bins=bins, density=normed,
                     cumulative=cumulative, color=pal[1])
        ax_hist.set(ylabel=ylabel)
        ax_hist.yaxis.labelpad = 1
        if xlabel:
            ax_hist.set(xlabel=xlabel)
        xmax = np.max(ax_hist.get_xlim())
        ymax = np.max(ax_hist.get_ylim())
        if ymax > 1000:
            format_yticks(ax_hist)
        if xmax > 1000:
            format_xticks(ax_hist)
        if describe_inset:
            ax_hist.text(xmax*.99, ymax*.975, describe_str(s),
                         ha='right', va='top', fontsize=16)
        if footnote:
            figh.tight_layout()
            figh.text(.99, footpad, footnote, ha='right', va='bottom',
                      fontsize='small', fontstyle='italic', color='#555555')
        ax.append(ax_hist)
        if saveplot:
            save_plot(figh=figh, outname='DistPlot_{}.png'.format(
                c.replace(' ', '_')))
    if ncol == 1:
        ax = ax[0]
    return ax

def print_update(idx, description='True values in index'):

    print('{}: {:,}/{:,} ({:.2f}%)'.format(description,
                                           np.sum(idx), len(idx), 100 * (np.sum(idx) / len(idx))))

def load_data(data_file = 'clickLog.json'):

    df = pd.read_json(data_file, lines=True).rename(col_rename, axis='columns')
    for n in feature_names:
        df[n] = format_categories(df[n])
        df[n] = pd.Categorical(df[n])

    for n in time_names:
        lab1 = n + '_sign'
        df[lab1] = 'Positive'
        df.loc[df[n] == 0, lab1] = 'Zero'
        df.loc[df[n] < 0, lab1] = 'Negative'
        df[lab1] = pd.Categorical(df[lab1])
        lab2 = n + '_(+)'
        df[lab2] = df[n]
        df.loc[df[n] <= 0, lab2] = np.NaN
    df.columns = format_names(df.columns)

    # - Descriptives
    d = df.describe().sort_index(axis=1)

    return df, d

