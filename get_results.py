import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.gridspec as gridspec
from mpl_toolkits import axes_grid1
import csv

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

def get_metric(outputs, data, beta, latent=20):
    dic_np = np.load(f'./outputs{outputs}/{data}{latent}_b{beta}/figs_data/figs_data.npy')
    dic = dic_np.item()
    return np.max(dic['disentangled_metric']), dic['disentangled_metric'][-1]

Data = [
    '2dbw_1pos_bvae_lat',
    '2dbw_pos_bvae_lat',
    '2dbw_pos_scl_bvae_lat',
    '2dbw_pos_scl_shp_bvae_lat',
    '2d_bw_bvae_lat',
    '2d_1col_bvae_lat',
    '3d_1col_2rot_bvae_lat'
]

Generative_factors = range(1, 8)

Betas = [
    1,
    2,
    4,
    8
]

Latents = [
    5,
    10,
    15,
    20
]

Latents_graph = [
    100,
    200,
    300,
    400
]

Outputs = [
    1,
    2,
    # 3
]

Complexity_results = np.zeros((len(Outputs), len(Data), len(Betas), 2)) # Last dimension is whether or not the result is best(0) or last(1)

for o in range(len(Outputs)):
    for d in range(len(Data)):
        for b in range(len(Betas)):
            Complexity_results[o, d, b] = get_metric(outputs=Outputs[o], data=Data[d], beta=Betas[b], latent=20)

Latent_results = np.zeros((len(Outputs), len(Latents), len(Betas), 2)) # Last dimension is whether or not the result is best(0) or last(1)

for o in range(len(Outputs)):
    for l in range(len(Latents)):
        for b in range(len(Betas)):
            Latent_results[o, l, b] = get_metric(outputs=Outputs[o], data='2d_bw_bvae_lat', beta=Betas[b], latent=Latents[l])

def get_image(data, xlabel, xticks, title, file_name, best=0):

    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    data_means_std = np.std(data_mean, axis=1)

    csv_columns = [f'Beta\\{xlabel}']
    for ind_col in range(data_mean.shape[0]): # Complexity/Latent size
        csv_columns.append(f'{xticks[ind_col]}')

    dict_data = []
    for ind_row in range(data_mean.shape[1]): # beta
        dict_data.append({csv_columns[0]: Betas[ind_row]})
        for ind_col in range(data_mean.shape[1]):
            dict_data[ind_row][csv_columns[ind_col+1]] = f'{data_mean[ind_col, ind_row, best]:.1f}({data_std[ind_col, ind_row, best]:.1f})'
            
    csv_file = f'{file_name}.csv'

    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for d in dict_data:
            writer.writerow(d)

    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2)
    
    plt.subplot(gs[0, 0])
    for b in range(len(Betas)):
        plt.plot(data_mean[:, b, best], label=rf'$\beta = {Betas[b]}$')
        plt.fill_between(range(data_mean.shape[0]), data_mean[:, b, best]+data_std[:, b, best], data_mean[:, b, best]-data_std[:, b, best], alpha=0.5)
    plt.legend()
    plt.xlabel(xlabel)
    plt.xticks(np.arange(len(xticks)), xticks)
    plt.ylabel('Disentanglement(%)')

    plt.subplot(gs[0, 1])
    plt.plot(data_means_std[:, best], linestyle='--', color='gray')
    plt.xlabel(xlabel)
    plt.xticks(np.arange(len(xticks)), xticks)
    plt.ylabel(rf'Standard deviation of the mean values across $\beta$\'s')

    plt.subplot(gs[1, 0])
    im = plt.imshow(data_mean[:, :, best].T, cmap='autumn', aspect='auto')
    cbar = add_colorbar(im)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Disentanglement(%): mean', rotation=270)
    plt.ylabel(r'$\beta$ value')
    plt.yticks(np.arange(len(Betas)), Betas)
    plt.xlabel(xlabel)
    plt.xticks(np.arange(len(xticks)), xticks)
    plt.gca().invert_yaxis()

    threshold_mean = np.percentile(data_mean[:, :, best], 50)
    for i in range(data_mean.shape[0]):
        for j in range(data_mean.shape[1]):
            mean = data_mean[i, j, best]
            color = 'black' if mean > threshold_mean else 'white'
            text = plt.text(i, j, f'{data_mean[i, j, best]:.1f}', ha='center', va='center', color=color)

    plt.subplot(gs[1, 1])
    im = plt.imshow(data_std[:, :, best].T, cmap='winter', aspect='auto')
    cbar = add_colorbar(im)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Disentanglement(%): standard deviation', rotation=270)
    plt.ylabel(r'$\beta$ value')
    plt.yticks(np.arange(len(Betas)), Betas)
    plt.xlabel(xlabel)
    plt.xticks(np.arange(len(xticks)), xticks)
    plt.gca().invert_yaxis()

    threshold_std = np.percentile(data_std[:, :, best], 90)
    for i in range(data_std.shape[0]):
        for j in range(data_std.shape[1]):
            std = data_std[i, j, best]
            color = 'black' if std > threshold_std else 'white'
            text = plt.text(i, j, f'{std:.1f}', ha='center', va='center', color=color)

    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    # fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(file_name)

get_image(
    data=Complexity_results,
    xlabel='Number of generative factors',
    xticks=Generative_factors,
    title=r'Disentanglement over varying data complexity and $\beta$ values',
    file_name='complexity_best',
    best=0
)

get_image(
    data=Complexity_results,
    xlabel='Number of generative factors',
    xticks=Generative_factors,
    title=r'Disentanglement over varying data complexity and $\beta$ values',
    file_name='complexity_last',
    best=1
)

get_image(
    data=Latent_results,
    xlabel='Latent space dimensionality',
    xticks=Latents,
    title=r'Disentanglement over varying dimensions of bottleneck and $\beta$ values',
    file_name='latent_best',
    best=0
)

get_image(
    data=Latent_results,
    xlabel='Latent space dimensionality',
    xticks=Latents,
    title=r'Disentanglement over varying dimensions of bottleneck and $\beta$ values',
    file_name='latent_last',
    best=1
)
