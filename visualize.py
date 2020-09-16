from sklearn.manifold import TSNE
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle
def plot_embedding(X, y, source_num, file_name):
    pp = PdfPages(file_name)
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure(figsize=(5, 5))
    #ax = Axes3D(fig)
    xs = X[:source_num]
    ys = y[:source_num]

    index_neg_s = np.where(ys == 0)[0]##for Xbully, change it to -1
    index_pos_s = np.where(ys == 1)[0]
    xs_pos = xs[index_pos_s]
    xs_neg = xs[index_neg_s]
    plt.scatter(xs_neg[:, 0], xs_neg[:, 1], color='blue', alpha=0.7, s=5, label='xs negative samples')
    plt.scatter(xs_pos[:, 0], xs_pos[:, 1], color='red', alpha=0.7, s=5, label='xs positive samples')

    pp.savefig()
    pp.close()

with open('HANCD_Tem_results.pickle', 'rb') as handle:
    store_data = pickle.load(handle)

#with open('label_test') as f:
 #   lines = f.read().splitlines()
#y_test=[int(l) for l in lines]
hs = store_data['representations']
ys=np.asarray(store_data['labels'])
#ys=np.asarray(y_test)
source_num = hs.shape[0]

tsne = TSNE(perplexity=30, n_components=2, n_iter=3000)
source_only_tsne = tsne.fit_transform(hs)
plot_embedding(source_only_tsne, ys, source_num, 'Temp.pdf')
