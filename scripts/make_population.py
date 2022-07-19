# From
# https://www.blog.pythonlibrary.org/2018/04/12/adding-svg-files-in-reportlab/
# With thanks

from pathlib import Path
import random
from collections.abc import Sequence

import numpy as np

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


def make_pop_panel(my_canvas, pop_bb, n,
                   icon_scale, bound_scale=0.8,
                   retries=1000):
    pop_bb = np.array(pop_bb)
    black, white = [get_icon(f, icon_scale) for f in ('black', 'white')]
    icon_size = np.array(black.getBounds()[2:])
    bound = icon_size * bound_scale
    maxes = pop_bb[2:] - icon_size
    rng = np.random.default_rng()
    samples = rng.choice([black, white], size=n, p=[0.26, 0.74])
    positions = np.zeros((n, 2))
    for i, icon in enumerate(samples):
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
        renderPDF.draw(icon, my_canvas, *xy)
        positions[i, :] = xy
    return positions


def make_chairs(my_canvas, chair_bb, members=()):
    if members:
        assert len(members) == 12
    chair = get_icon('chair', 0.1)
    chair_size = np.array(chair.getBounds()[2:])
    bb_width = chair_bb[2] - chair_bb[0]
    chair_gap = (bb_width - chair_size[0]) / 11
    chair_x = (np.arange(12) * chair_gap) + chair_bb[0]
    for i, x in enumerate(chair_x):
        renderPDF.draw(chair, my_canvas, x, chair_bb[1])
    for i, x in enumerate(chair):
        renderPDF.draw(chair, my_canvas, x, chair_bb[1])


def main():
    page_height = 1000
    page_width = page_height * 16 / 9  # Widescreen slides
    margins = [50, 50, 50, 50]
    my_canvas = canvas.Canvas('population.pdf',
                              pagesize=(page_width, page_height))
    pop_right = round(page_width / 2.5)
    pop_bb = margins[:2] + [pop_right, page_height - margins[-1] - 50]
    make_pop_panel(my_canvas, pop_bb, n=500, icon_scale=0.05)
    jury_bb = [pop_right + margins[0] * 2, 700,
               page_width - margins[-2], page_height - margins[-1]]
    make_chairs(my_canvas, jury_bb)
    my_canvas.setFont('Helvetica', 60)
    my_canvas.drawString(200, 925, 'Population')
    my_canvas.save()


if __name__ == "__main__":
    main()
