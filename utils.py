import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt

def show_bboxes(ax, boxes, labels, colors):
    for i in range(len(boxes)):
        # plot bboxes
        bbox = boxes[i]
        xy = (bbox[0], bbox[1])
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        rect = mlp.patches.Rectangle(xy, w, h, lw=1, 
                            facecolor='none', edgecolor=colors[i])
        ax.add_patch(rect)

        text = labels[i]

        bbox_props = dict(boxstyle='square', facecolor=colors[i],
                          edgecolor='none', pad=0)
        ax.text(xy[0], xy[1], text,style='italic',
                              color='k',horizontalalignment='left',
                              verticalalignment='bottom',
                              bbox=bbox_props)

def show_masks(ax, masks, colors, segments=None):
    cmap = mlp.colors.LinearSegmentedColormap.from_list('masks_cmap', colors)
    for i in range(len(masks)):
        mask = masks[i] 
        if mask.ndim == 3:
            mask = mask.squeeze(0)
        masked = np.ma.masked_where(mask<0.5, mask)
        ax.imshow(masked, cmap, alpha=.3)

        if segments:
            seg = segments[i]
            ax.plot(seg[:, 0], seg[:, 1], lw=1,
                    color=colors[i])