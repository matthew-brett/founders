# From
# https://www.blog.pythonlibrary.org/2018/04/12/adding-svg-files-in-reportlab/
# With thanks

from io import BytesIO
from pathlib import Path
from collections.abc import Sequence

import numpy as np
import matplotlib.pyplot as plt

from reportlab.graphics import renderPDF
from reportlab.pdfgen import canvas
from svglib.svglib import svg2rlg


IMAGE_PATH = Path(__file__).parent / '..' / 'images'


def get_icon(froot, scales=None):
    pth = IMAGE_PATH / f'{froot}.svg'
    icon = svg2rlg(pth)
    if scales:
        if not isinstance(scales, Sequence):
            scales = [scales, scales]
        icon.scale(*scales)
    return icon


def get_bw(scales):
    return {f: get_icon(f, scales) for f in ('black', 'white')}


def make_pop_panel(my_canvas, pop_bb, n,
                   icon_scale, bound_scale=0.8,
                   retries=1000):
    pop_bb = np.array(pop_bb)
    icons = get_bw(icon_scale)
    icon_size = np.array(icons['black'].getBounds()[2:])
    bound = icon_size * bound_scale
    maxes = pop_bb[2:] - icon_size
    rng = np.random.default_rng()
    samples = rng.choice(['black', 'white'], size=n, p=[0.26, 0.74])
    positions = np.zeros((n, 2))
    for i, juror in enumerate(samples):
        for j in range(retries):
            xy = np.round([
                rng.uniform(pop_bb[0], maxes[0] + 1),
                rng.uniform(pop_bb[1], maxes[1] + 1)])
            diffs = np.abs(positions - xy)
            in_bound = np.all(diffs < bound, axis=1)
            if not np.any(in_bound):
                break
        else:
            print(f'Bailed after placing {i} icons')
            break
        renderPDF.draw(icons[juror], my_canvas, *xy)
        positions[i, :] = xy
    centers = positions + icon_size / 2
    return np.array(list(zip(samples, centers)), dtype=object)


def make_chairs(my_canvas, chair_bb, members=(),
               icon_scale=0.1):
    chair = get_icon('chair', icon_scale)
    chair_height = chair.getBounds()[-1]
    bw = get_bw(icon_scale)
    if len(members) < 12:
        members = tuple(members) + (None,) * (12 - len(members))
    chair_size = np.array(chair.getBounds()[2:])
    bb_width = chair_bb[2] - chair_bb[0]
    chair_gap = (bb_width - chair_size[0]) / 11
    chair_x = (np.arange(12) * chair_gap) + chair_bb[0]
    member_corners = np.zeros((12, 2))
    for i, (x, member) in enumerate(zip(chair_x, members)):
        renderPDF.draw(chair, my_canvas, x, chair_bb[1])
        m_y = chair_bb[1] + chair_height * 1.2
        if member:
            renderPDF.draw(bw[member], my_canvas, x, m_y)
        member_corners[i, :] = (x, m_y)
    return member_corners


def make_hist(counts, max_count=40):
    plt.figure(figsize=(12, 5.75))
    plt.hist(counts[~np.isnan(counts)], bins=np.arange(13))
    plt.axis([0, 12, 0, max_count])
    plt.xticks(np.arange(13))
    plt.xlabel('Number of Black jurors')
    plt.ylabel('Count')
    fig_fo = BytesIO()
    plt.savefig(fig_fo, dpi=600, format='svg')
    fig_fo.seek(0)
    return svg2rlg(fig_fo)


def main():
    n_iters = 100
    page_height = 1000
    page_width = page_height * 16 / 9  # Widescreen slides
    margins = [50, 50, 50, 50]
    my_canvas = canvas.Canvas('population.pdf',
                              pagesize=(page_width, page_height))
    pop_right = round(page_width / 2.5)
    pop_bb = margins[:2] + [pop_right, page_height - margins[-1] - 50]
    icon_scale = 0.05
    samples = make_pop_panel(my_canvas, pop_bb, n=500, icon_scale=icon_scale)
    jury_left = pop_right + margins[0] * 2
    jury_bb = [jury_left, 700,
               page_width - margins[-2], page_height - margins[-1]]
    rng = np.random.default_rng()
    counts = np.zeros(n_iters) + np.nan
    sample = rng.choice(samples, size=12, replace=True)
    jurors, centers = zip(*sample)
    icons = get_bw(icon_scale)
    r = icons['black'].getBounds()[-1] * 0.75
    my_canvas.setStrokeColorRGB(1, 0, 0)
    my_canvas.setLineWidth(5)
    center = centers[0]
    my_canvas.circle(center[0], center[1], r, stroke=1, fill=0)
    my_canvas.setStrokeColorRGB(0, 0, 0)
    n_black = len([j for j in jurors if j == 'black'])
    counts[0] = n_black
    make_chairs(my_canvas, jury_bb, jurors)
    my_canvas.setFont('Helvetica', 60)
    my_canvas.drawString(200, 925, 'Population')
    my_canvas.drawString(pop_right + 500, 925, 'Jury')
    my_canvas.drawString(pop_right + 420, 575, f'Count = {n_black}')
    plt_svg = make_hist(counts)
    renderPDF.draw(plt_svg, my_canvas, pop_right, margins[1])
    my_canvas.save()


if __name__ == "__main__":
    main()
