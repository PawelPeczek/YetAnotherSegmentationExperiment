from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def make_column_plot(title: str,
                     column_names: List[str],
                     content: List[Tuple[np.ndarray]]
                     ) -> plt.Figure:
    columns_number = len(column_names)
    rows_number = len(content)
    figure_size = (10, len(content) * 4)
    fig, ax = plt.subplots(rows_number, columns_number, figsize=figure_size)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(title, fontsize=22)
    for row_idx, row in enumerate(content):
        for column_id, element in enumerate(row[:columns_number]):
            ax[row_idx, column_id].imshow(content[row_idx][column_id])
            ax[row_idx, column_id].set_title(
                column_names[column_id], fontsize='medium'
            )
    return fig
