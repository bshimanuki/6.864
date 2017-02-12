from sklearn.manifold import TSNE
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

files = np.load("mid_stretch.npz")
num_vectors = 200
niv = files["niv"][:num_vectors,:]
nkjv = files["nkjv"][:num_vectors,:]

together = np.concatenate((niv, nkjv), axis=0)
model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
together = model.fit_transform(together)
nivx = together[:num_vectors,0]
nivy = together[:num_vectors,1]
nkjvx = together[num_vectors:,0]
nkjvy = together[num_vectors:,1]

plt.plot(nivx, nivy, 'ro')
plt.plot(nkjvx, nkjvy, 'bo')
plt.axis('off')
plt.savefig('plot', bbox_inches='tight')

files.close()
