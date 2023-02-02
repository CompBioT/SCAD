import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def scale_to_01_range(x):
    ''' scale and move the coordinates so they fit [0; 1] range '''
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range

def plot_tsne(feature_cb,label_cb,save_tsne_plot_to,it):
    tsne = TSNE(n_components=2, random_state=42).fit_transform(feature_cb)
    tx = scale_to_01_range(tsne[:, 0])
    ty = scale_to_01_range(tsne[:, 1])
    #define color
    classes = ['S_sens','S_res','T_sens','T_res']
    class2idx = {c:i for i, c in enumerate(classes)}
    colors = ['darkorange', 'cornflowerblue', 'tomato', 'forestgreen'] #rebeccapurple,forestgreen
    colors_per_class = {label: colors[i] for i, label in enumerate(classes)}

    markers = ['>','>',',',',']
    markers_per_class = {label: markers[i] for i, label in enumerate(classes)}

    fig = plt.figure(figsize=(4, 4))
    plt.rcParams.update({'font.size': 10})
    ax = fig.add_subplot(111)
    for label in colors_per_class:
        indices = [i for i, l in enumerate(label_cb) if l == classes[class2idx[label]]]
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        color = colors_per_class[label]
        mrk = markers_per_class[label]
        ax.scatter(current_tx, current_ty, c=color, label=label, alpha=0.5, marker = mrk, s = 25)

    #ax.legend(loc='upper right')
    ax.legend(loc='lower right', bbox_to_anchor=(1.1, -0.05))
    it=it+1
    plt.title("TSNE of extracted feature, epoch = {}".format(it))
    plt.savefig(save_tsne_plot_to)
    plt.close()