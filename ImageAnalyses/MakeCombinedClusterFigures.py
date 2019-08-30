"""
combine cluster heatmaps and slices
for use as supplementary figures
"""

import os
from PIL import Image


def append_images(imgfiles, direction='vertical',
                  bg_color=(255, 255, 255),
                  aligment='center',
                  nudge=100):
    """
    Appends images in horizontal/vertical direction.

    Args:
        images: List of PIL images
        direction: direction of concatenation, 'horizontal' or 'vertical'
        bg_color: Background color (default: white)
        aligment: alignment mode if images need padding;
           'left', 'right', 'top', 'bottom', or 'center'

    Returns:
        Concatenated image as a new PIL image object.
    """
    images = map(Image.open, imgfiles)
    images = [i.convert('RGBA') for i in images]
    widths, heights = zip(*(i.size for i in images))

    if direction == 'horizontal':
        new_width = sum(widths)
        new_height = max(heights)
    else:
        new_width = max(widths)
        new_height = sum(heights)

    new_im = Image.new('RGBA', (new_width, new_height), color=bg_color)

    offset = 0
    for i, im in enumerate(images):
        if direction == 'horizontal':
            y = 0
            if aligment == 'center':
                y = int((new_height - im.size[1])/2)
            elif aligment == 'bottom':
                y = new_height - im.size[1]
            new_im.paste(im, (offset, y))
            offset += im.size[0]
        else:
            x = 0
            if i == 0:
                xnudge = 0
            else:
                xnudge = nudge

            if aligment == 'center':
                x = int((new_width - im.size[0])/2 + xnudge)
            elif aligment == 'right':
                x = new_width - im.size[0]
            new_im.paste(im, (x, offset))
            offset += im.size[1]

    return new_im

# after https://stackoverflow.com/questions/30227466/
#  combine-several-images-horizontally-with-python


def make_combined_cluster_figures(basedir):
    figdir = os.path.join(basedir, 'figures')
    for hyp in range(1, 10):
        if hyp in [3, 4]:
            continue
        heatmap_file = os.path.join(
            figdir,
            'hyp%d_spearman_map_unthresh.png' % hyp
        )
        slices_file = os.path.join(
            figdir,
            'hyp%d_cluster_means.png' % hyp
        )
        outfile = os.path.join(
            figdir,
            'hyp%d_combined_clusters.png' % hyp
        )
        new_im = append_images([heatmap_file, slices_file])
        new_im.save(outfile)


if __name__ == "__main__":

    basedir = os.environ['NARPS_BASEDIR']
    make_combined_cluster_figures(basedir)
