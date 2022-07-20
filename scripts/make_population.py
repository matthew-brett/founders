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


class JuryPage:

    def __init__(self,
                 out_path,
                 image_path=Path(__file__).parent / '..' / 'images',
                 n_iters=100,
                 max_count=40,
                 page_height=1000,
                 page_width=None,
                 margins=(50, 50, 50, 50),
                 pop_icon_scale=0.05,
                 chair_icon_scale=0.1,
                 n_pop=500,
                 pop_bound_scale=0.8,
                ):
        self._canvas = None
        self.out_path = out_path
        self.image_path = Path(image_path)
        if page_width is None:
            page_width = page_height * 16 / 9  # Widescreen slides
        self.n_iters = n_iters
        self.max_count = max_count
        self.page_height = page_height
        self.page_width = page_width
        self.margins = margins
        self.rng = np.random.default_rng()
        self.icons = {f: self._get_icon(f)
                      for f in ('black', 'white', 'chair')}
        self.pop_icons = self._scaled_icons(self.icons, pop_icon_scale)
        self.chair_icons = self._scaled_icons(self.icons, chair_icon_scale)
        self.pop_icon_scale = pop_icon_scale
        self.chair_icon_scale = chair_icon_scale
        self.n_pop = n_pop
        self.pop_bound_scale = pop_bound_scale
        self.counts = np.zeros(n_iters) + np.nan
        self.pop_right = round(page_width / 2.5)
        self.jury_left = self.pop_right + margins[0] * 2
        self.pop_bb = np.array(
            margins[:2] + [self.pop_right, page_height - margins[-1] - 50])
        self.jury_bb = np.array(
            [self.jury_left,
             700,
             page_width - margins[-2],
             page_height - margins[-1]])
        self.calc_pop_panel()
        self.calc_chairs()
        self.reset(out_path)

    def reset_counts(self):
        self.jury = None
        self.current_count = np.nan
        self.counts = np.zeros(self.n_iters) + np.nan
        self.n_samples = 0

    def reset_canvas(self, out_path, draw=True):
        self.out_path = str(out_path)
        self._canvas = canvas.Canvas(
            self.out_path,
            pagesize=(self.page_width, self.page_height))
        if draw:
            self.draw_background()

    def reset(self, out_path, draw=True):
        self.reset_canvas(out_path, draw)
        self.reset_counts()
        self._current_hist = None

    @property
    def canvas(self):
        if self._canvas is None:
            self.reset_canvas(self.out_path)
        return self._canvas

    def _get_icon(self, froot):
        pth = self.image_path / f'{froot}.svg'
        return svg2rlg(pth)

    def _scaled_icons(self, icons, scales=None):
        new_icons = {k: v.copy() for k, v in self.icons.items()}
        if scales:
            if not isinstance(scales, Sequence):
                scales = [scales, scales]
            for k, v in new_icons.items():
                v.scale(*scales)
        return new_icons

    def calc_pop_panel(self, retries=1000):
        pop_bb = np.array(self.pop_bb)
        icons = self.pop_icons
        icon_size = np.array(icons['black'].getBounds()[2:])
        bound = icon_size * self.pop_bound_scale
        maxes = pop_bb[2:] - icon_size
        samples = self.rng.choice(['black', 'white'],
                             size=self.n_pop,
                             p=[0.26, 0.74])
        positions = np.zeros((self.n_pop, 2))
        for i, juror in enumerate(samples):
            for j in range(retries):
                xy = np.round([
                    self.rng.uniform(pop_bb[0], maxes[0] + 1),
                    self.rng.uniform(pop_bb[1], maxes[1] + 1)])
                diffs = np.abs(positions - xy)
                in_bound = np.all(diffs < bound, axis=1)
                if not np.any(in_bound):
                    break
            else:
                raise RuntimeError(f'Bailed after placing {i} icons')
            positions[i, :] = xy
        centers = positions + icon_size / 2
        self.population = np.array(list(zip(samples, centers)), dtype=object)
        self.pop_icon_size = icon_size

    def draw_pop(self):
        self.canvas.setFont('Helvetica', 60)
        self.canvas.drawString(200, 925, 'Population')
        icons = self.pop_icons
        for juror, xy in self.population:
            x, y = xy - self.pop_icon_size / 2
            renderPDF.draw(icons[juror], self.canvas, *xy)

    def calc_chairs(self):
        chair = self.chair_icons['chair']
        chair_size = np.array(chair.getBounds()[2:])
        chair_height = chair_size[1]
        bb_width = self.jury_bb[2] - self.jury_bb[0]
        chair_gap = (bb_width - chair_size[0]) / 11
        self.chair_x = (np.arange(12) * chair_gap) + self.jury_bb[0]
        self.member_corners = np.zeros((12, 2))
        for i, x in enumerate(self.chair_x):
            m_y = self.jury_bb[1] + chair_height * 1.2
            self.member_corners[i, :] = (x, m_y)

    def draw_chairs(self):
        chair = self.chair_icons['chair']
        for i, x in enumerate(self.chair_x):
            renderPDF.draw(chair, self.canvas, x,
                           self.jury_bb[1])
        self.canvas.drawString(self.pop_right + 500, 925, 'Jury')

    def draw_members(self, members):
        if hasattr(members, 'shape') and len(members.shape) == 2:
            members = members[:, 0]
        if len(members) < 12:
            members = tuple(members) + (None,) * (12 - len(members))
        for i, ((x, y), member) in enumerate(zip(self.member_corners, members)):
            if member is None:
                continue
            renderPDF.draw(self.chair_icons[member], self.canvas, x, y)

    def draw_background(self):
        self.draw_pop()
        self.draw_chairs()

    def calc_hist(self):
        fig = plt.figure(figsize=(12, 5.75))
        plt.hist(self.counts[~np.isnan(self.counts)], bins=np.arange(13))
        plt.axis([0, 12, 0, self.max_count])
        plt.xticks(np.arange(13))
        plt.xlabel('Number of Black jurors')
        plt.ylabel('Count')
        fig_fo = BytesIO()
        plt.savefig(fig_fo, dpi=600, format='svg')
        plt.close(fig)
        fig_fo.seek(0)
        self.current_hist = svg2rlg(fig_fo)

    def draw_hist(self):
        renderPDF.draw(self.current_hist,
                       self.canvas,
                       self.pop_right,
                       self.margins[1])

    def draw_circle_pop(self, element):
        name, center = element
        r = self.pop_icons['black'].getBounds()[-1] * 0.75
        c = self.canvas
        c.saveState()
        c.setStrokeColorRGB(1, 0, 0)
        c.setLineWidth(5)
        c.circle(center[0], center[1], r, stroke=1, fill=0)
        c.restoreState()

    def sample_pop(self):
        self.jury = self.rng.choice(self.population, size=12)
        self.current_count = len([s[0] for s in self.jury if s[0] == 'black'])
        self.counts[self.n_samples] = self.current_count
        self.n_samples += 1
        return self.jury

    def draw_jury(self):
        self.draw_members(self.jury[:, 0])

    def draw_count(self, count=None):
        c = self.canvas
        c.saveState()
        c.setFont('Helvetica', 60)
        if count is None:
            count = '?'
            if not np.isnan(self.current_count):
                count = self.current_count
        c.drawString(self.pop_right + 420, 575, f'Count = {count}')
        c.restoreState()

    def save(self):
        self.canvas.save()

def main():
    opp = Path() / 'out_pdfs'
    jp = JuryPage(
        opp / 'hypo_at_first.pdf',
        n_iters = 100,
        page_height = 1000,
        margins = [50, 50, 50, 50])
    jp.save()
    n_slow = 3
    for i in range(n_slow):
        jury = jp.sample_pop()
        count = jp.current_count
        prefix = f'hypo_jury{i}'
        for j, juror in enumerate(jury):
            jp.reset_canvas(opp / f'{prefix}_{j:02d}_chair_no.pdf')
            jp.draw_circle_pop(juror)
            jp.draw_members(jury[:j])
            if i > 0:
                jp.draw_hist()
            jp.save()
            jp.reset_canvas(opp / f'{prefix}_{j:02d}_chair_yes.pdf')
            jp.draw_circle_pop(juror)
            jp.draw_members(jury[:j + 1])
            if i > 0:
                jp.draw_hist()
            jp.save()
        jp.reset_canvas(opp / f'{prefix}_seated_at_first.pdf')
        jp.draw_members(jury)
        if i > 0:
            jp.draw_hist()
        jp.save()
        jp.reset_canvas(opp / f'{prefix}_seated_counted.pdf')
        jp.draw_members(jury)
        jp.draw_count(count)
        if i > 0:
            jp.draw_hist()
        jp.save()
        jp.reset_canvas(opp / f'{prefix}_seated_hist.pdf')
        jp.draw_members(jury)
        jp.draw_count(count)
        jp.calc_hist()
        jp.draw_hist()
        jp.save()
    prefix = 'hypo_zall'
    jp.reset_canvas(opp / f'{prefix}.pdf')
    for i in range(jp.n_iters - n_slow):
        jp.sample_pop()
    jp.calc_hist()
    jp.draw_hist()
    jp.save()


if __name__ == "__main__":
    main()
