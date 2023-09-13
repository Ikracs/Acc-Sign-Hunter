import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages

FONT_SIZE  = 16
LINE_WIDTH = 2.5

LOG_ROOT = 'logs'

SAVE_ROOT = 'figs'
SAVE_NAME = 'prop.pdf'

LOG_FILES = [

]

if __name__ == '__main__':
    if not os.path.exists(SAVE_ROOT): os.mkdir(SAVE_ROOT)
    pdf = PdfPages(os.path.join(SAVE_ROOT, SAVE_NAME))

    xlim = 0
    for fname in LOG_FILES:
        with open(os.path.join(LOG_ROOT, fname), 'r') as f:
            info = json.load(f)
        plt.plot(
            info['Corr bits Prop'],
            label=f"$\epsilon={info['epsilon']:.1e}$",
            linewidth=LINE_WIDTH
        )
        xlim = max(xlim, len(info['Corr bits Prop']))
    plt.xlabel('Iteration', fontsize=FONT_SIZE)
    plt.ylabel('Proportion of Correct bits', fontsize=FONT_SIZE)
    plt.xticks(
        np.linspace(0, xlim, 11).astype(np.uint16),
        rotation=45, fontsize=FONT_SIZE
    )
    plt.yticks(fontsize=FONT_SIZE)
    plt.gca().yaxis.set_major_formatter(
        ticker.FormatStrFormatter('%.3f')
    )
    plt.legend(fontsize=FONT_SIZE)
    plt.grid(); plt.tight_layout()
    pdf.savefig()
    pdf.close()
    plt.close()
