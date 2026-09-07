"""Colormaps and shared plot furniture.

``colormap`` is moved verbatim from ``main/myfun.py``.  ``MSEadjust/myfun.py``
carries a second body under the same name, which has not been diffed against
this one.
"""

from __future__ import annotations

import numpy as np

def colormap(color,*args, **kwargs):
    from matplotlib.colors import ListedColormap

    if color == "PurOra":
        color = ['#828de7', '#979ae7', '#aba7e8', '#bcb5ea', '#ccc3ec', '#dbd1ef', '#e9e0f3', '#f5eff8', '#ffffff', '#ffffff','#fff2d2', '#ffe3bd', '#ffd3ad', '#ffc3a0', '#ffb295', '#ffa08b', '#ff8d82', '#ff7979']
        cmap  = ListedColormap(color)
        cmap.set_under(color = '#6981e6' )
        cmap.set_over(color = '#ff6171')
    elif color == "BluWhiRed":
        color = ['#6281d7', '#6d90da', '#79a0dc', '#85afdf', '#92bee1', '#a1cde3', '#b1dce4', '#c5eae6', '#ddf7ea', '#ffffff', '#fff0cd', '#ffe0bb', '#ffd0a9', '#ffc096', '#ffaf84', '#ff9e73', '#ff8b61', '#fd7852', '#fa6345']
        cmap  = ListedColormap(color)
        cmap.set_under(color = '#5671d4' )
        cmap.set_over(color = '#ff3f46')
    elif color == "WhiRed":
        color = ['#ffffff', '#ffffff','#fff7e7', '#ffeed1', '#ffe5be', '#ffdcab', '#ffd29b', '#ffc88c', '#ffbe7f', '#ffb373', '#ffa869', '#ff9d5f', '#ff9157', '#ff854f', '#ff7749', '#ff6942', '#fc5b3c', '#fa4c36', '#f83a2e', '#f42423']
        cmap  = ListedColormap(color)
        cmap.set_under(color = '#ffffff' )
        cmap.set_over(color = '#ef0000')
    elif color == "WhGreBlu":
        color = ['#eaf4df', '#d3e8c4', '#bcddac', '#a3d199', '#89c68a', '#6fb5b7', '#66a4ba', '#5a95b8', '#4c86b4']
        cmap  = ListedColormap(color)
        cmap.set_under(color = '#ffffff' )
        cmap.set_over(color = '#3c78af')
    elif color == "PurGre":
        color = ['#808ee7', '#949be9', '#a6a8ec', '#b7b6ee', '#c7c4f1', '#d6d2f4', '#e4e1f7', '#f2f0fb', '#ffffff', '#ffffff', '#eafbf3', '#d5f6e7', '#c0f1db', '#abecce', '#97e6c2', '#83e1b5', '#70daa9', '#61d39b']
        cmap  = ListedColormap(color)
        cmap.set_under(color = '#6981e6')
        cmap.set_over(color = '#5dcb8c')
    elif color == "BluPurOra":
        color = ['#4267df', '#4d69de', '#576bdd', '#5f6ddc', '#686fdb', '#6f71da', '#7673d9', '#7d75d8', '#8477d7', '#8a79d6', '#907bd5', '#967dd4', '#9c7fd3', '#a281d1', '#a783d0', '#ad85cf', '#b287cd', '#b889cc', '#bd8bca', '#c28dc8', '#c790c6', '#cd92c4', '#d294c2', '#d796c0', '#dc98bd', '#e29aba', '#e79cb7', '#ed9eb3', '#f2a0ae']
        cmap  = ListedColormap(color)
        cmap.set_under(color = '#ffffff')
        cmap.set_over(color = '#ffa194')
    return cmap
