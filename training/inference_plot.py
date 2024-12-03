import matplotlib.pyplot as plt
import cv2
import numpy as np


def extract_rgb(image):
    return np.dstack([image[2, :, :], image[1, :, :], image[0, :, :]])

def stretch_rgb(img,rgb_dict):
    stacked_img = np.hstack([rgb_dict['l8'],rgb_dict['s2']])
    normalized = []
    for i in range(3):
        normalized.append(img[:, :, i] * (255.0 / stacked_img[:, :, i].max()))

    normalized = np.dstack(normalized)
    return normalized.astype(np.uint8)


def make_fig(image_dict,epoch):
    fig, ax = plt.subplots(1, 4, figsize=(20, 5), dpi=80, sharex=True, sharey=True)
    fig.suptitle(f"EPOCH-{epoch} RGB-Composite")

    rgb_dict = {'s2':extract_rgb(image_dict['s2']),
                'l8':extract_rgb(image_dict['l8']),
                'l8_pan':image_dict['l8_pan'],
                'pred':extract_rgb(image_dict['pred'])}
    # rgb_dict['l8_zoom'] = cv2.resize(rgb_dict['l8'],[256,256],interpolation=cv2.INTER_NEAREST)

    ax[0].imshow(stretch_rgb(rgb_dict['s2'],rgb_dict),origin = 'lower')
    ax[0].set_title("Sentinel-2 (10m)")
    ax[0].set_axis_off()

    # ax[1].imshow(stretch_rgb(rgb_dict['l8'],rgb_dict),origin = 'lower')
    # ax[1].set_title("Landsat-8 (30m)")
    # ax[1].set_axis_off()

    ax[1].imshow(rgb_dict['l8_pan'],origin = 'lower',cmap = 'gray')
    ax[1].set_title("Landsat-8 PAN (15m)")
    ax[1].set_axis_off()

    ax[2].imshow(stretch_rgb(rgb_dict['l8'],rgb_dict),origin = 'lower')
    ax[2].set_title("Landsat-8 Bilinear")
    ax[2].set_axis_off()

    ax[3].imshow(stretch_rgb(rgb_dict['pred'],rgb_dict),origin = 'lower')
    ax[3].set_title("model prediction")
    ax[3].set_axis_off()

    return fig

def make_channel_wise_fig(img_dict):
    fig, axs = plt.subplots(6, 1, dpi=80, sharex=True, sharey=True)
    fig.suptitle('Bandwise visualization')

    for ax in axs:
        ax.remove()

    gridspec = axs[0].get_subplotspec().get_gridspec()
    subfigs = [fig.add_subfigure(gs) for gs in gridspec]

    l8_img,s2_img,pred = img_dict['l8'],img_dict['s2'],img_dict['pred']
    band_names = {
        0:"BLUE",
        1:"GREEN",
        2:"RED",
        3:"NIR",
        4:"SWIR1",
        5:"SWIR2"
    }

    for i,subfig in enumerate(subfigs):
        l8_ch,s2_ch,pred_ch = np.squeeze(l8_img[i,:,:]),np.squeeze(s2_img[i,:,:]),np.squeeze(pred[i,:,:])
        l8_zoom_ch = cv2.resize(l8_ch,[256,256],interpolation=cv2.INTER_NEAREST)
        min,max = 0,np.array([s2_ch,l8_zoom_ch]).max()

        subfig.suptitle(band_names[i])
        axs = subfig.subplots(nrows = 1,ncols = 4,sharex = True,sharey = True)


        axs[0].imshow(s2_ch,vmin = min,vmax = max, origin = 'lower', cmap = "gray")
        axs[0].set_axis_off()

        axs[1].imshow(l8_ch,vmin = min,vmax = max, origin = 'lower', cmap = "gray")
        axs[1].set_axis_off()

        axs[2].imshow(l8_zoom_ch,vmin = min,vmax = max, origin = 'lower', cmap = "gray")
        axs[2].set_axis_off()

        axs[3].imshow(pred_ch,vmin = min,vmax = max, origin = 'lower', cmap = "gray")
        axs[3].set_axis_off()

        if i==0:
            axs[0].set_title(f"Sentinel-2 {band_names[i]}")
            axs[1].set_title(f"Landsat-8 {band_names[i]}")
            axs[2].set_title(f"Landsat-8 NN {band_names[i]}")
            axs[3].set_title(f"Prediction {band_names[i]}")
    plt.show()


if __name__ == '__main__':
    img_dict = {
        's2' : np.random.rand(6,256,256),
        'l8' : np.random.rand(6,86,86),
        'pred': np.random.rand(6,256,256)
    }
    make_channel_wise_fig(img_dict)