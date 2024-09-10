from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train_flat = x_train.reshape(-1, 28*28).astype('float32') / 255.0
x_test_flat = x_test.reshape(-1, 28*28).astype('float32') / 255.0

pca = PCA(n_components=128)
x_train_pca = pca.fit_transform(x_train_flat)

x_train_reconstructed = pca.inverse_transform(x_train_pca)

x_test_pca = pca.transform(x_test_flat)
x_test_reconstructed = pca.inverse_transform(x_test_pca)

idx = np.random.randint(0, len(x_test))  # pick a random index
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(x_test_flat[idx].reshape(28, 28), cmap='gray')
axes[0].set_title('Original Image')

axes[1].imshow(x_test_reconstructed[idx].reshape(28, 28), cmap='gray')
axes[1].set_title('Reconstructed Image')

plt.show()
