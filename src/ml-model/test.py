from sklearn.decomposition import PCA

arr = [[1, 2, 1, 4, 5], [1, 2, 5, 6, 7]]
pca = PCA(n_components=1)
pca.fit(arr)
print(pca.singular_values_)