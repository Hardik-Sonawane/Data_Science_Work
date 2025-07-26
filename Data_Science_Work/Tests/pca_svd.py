####################################  Test 11   ######################################################
################################################################################################
######################################################################################################
"""
Que.1. Use PCA to compress images from the sklearn digits dataset. Reconstruct 
the images using fewer principal components and compare the results. 
Goals: 
 Apply PCA to reduce the dimensionality of the digit images. 
 Reconstruct the compressed images using a selected number of components. 
 Compare the visual quality of the reconstructed images with the original ones using a side-by-side plot.
 """
# Solution ::>

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

digits = load_digits()
images = digits.images  
data = digits.data
n_components = 10
pca = PCA(n_components=n_components)
compressed_data = pca.fit_transform(data)

rcstrd_data = pca.inverse_transform(compressed_data)
rcstrd_images = rcstrd_data.reshape(-1, 8, 8)

n_images = 10  
fig, axes = plt.subplots(nrows=2, ncols=n_images, figsize=(10, 3))

for i in range(n_images):
    axes[0, i].imshow(images[i], cmap='gray')
    axes[0, i].axis('off')
    axes[0, i].set_title(f'Original {i}')
    axes[1, i].imshow(rcstrd_images[i], cmap='gray')
    axes[1, i].axis('off')
    axes[1, i].set_title(f'Reconstructed {i}')

plt.suptitle(f'PCA Image Compression with {n_components} Components')
plt.tight_layout()
plt.show()
################################################3#######################

"""
Que.2. Perform PCA on the iris dataset to understand how much variance is 
explained by each principal component. Additionally, you must determine 
how many principal components are required to capture at least 95% of 
the total variance in the data. 
Goals: 
Apply PCA to the standardized Iris dataset. 
 Calculate and plot the cumulative explained variance for each principal component. 
 Identify the minimum number of components needed to explain 95% of the variance. 
 Visualize this with a plot that shows the cumulative explained variance. 
 """
 #Solution::>
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris


iris = load_iris()
X = iris.data  
    #########  Data Standardizing   #######
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

       ### Applying PCA #######
pca = PCA()
pca.fit(X_std)

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

########### Ploting  cumulative explained variance   ######
plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--', color='b')
plt.axhline(y=0.95, color='r', linestyle='-')
plt.text(0.5, 0.96, '95% cut-off threshold', color='red', fontsize=12)
plt.title('Cumulative Explained Variance by the Principal Components')
plt.xlabel('Number of The Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()

n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f'Number of principal components required to capture at least 95% variance: {n_components_95}')
################################################################################################################
 
""" 
Que.3. Perform Singular Value Decomposition (SVD) on a randomly generated 
matrix and verify that the original matrix can be reconstructed using the 
product of the decomposed matrices. 
Goals: 
Generate a random matrix of size 5x5. 
 Perform SVD on this matrix to obtain the U, Σ (singular values), and Vᵀ matrices. 
 Reconstruct the original matrix using the decomposed matrices. 
 Compare the original and reconstructed matrices and compute the difference.
 """
##Solution::>
import numpy as np
######### Step 1: generating  matrix #######
np.random.seed(42)
A = np.random.rand(5, 5)

############step 2: SVD performing  ###########
U, S, Vt = np.linalg.svd(A)

####### Step 3:  #################
Sigma = np.diag(S)
## reconstruct matrix ##
A_reconstructed = U @ Sigma @ Vt

### Step 4: comapring original one and reconstructed matrix #### 
print("Original Matrix A:\n", A)
print("\nReconstructed Matrix :\n", A_reconstructed)

##### Computing difference between them #####
difference = np.abs(A - A_reconstructed)
print("\nDifference between Original and Reconstructed Matrix:\n", difference)

#### sum of differnce ideally it should be zero ####
total_difference = np.sum(difference)
print(f"\nTotal difference between original and reconstructed matrix: {total_difference}")

###############################################################################################
"""
QUE.5. You are given a small collection of text documents. Use TruncatedSVD on 
a TF-IDF matrix derived from a small set of text documents to reduce the 
dimensionality of the matrix and reconstruct it. 
Goals: 
 Convert a set of text documents into a TF-IDF matrix using TfidfVectorizer. 
 Apply TruncatedSVD to reduce the dimensionality of the TF-IDF matrix to 2 components. 
 Reconstruct the original TF-IDF matrix from the reduced representation. 
 Compare the reconstructed matrix with the original one to assess how much information was 
retained.
Sample text document

doc = ["The quick brown fox jumps over the lazy dog",
       "Never jump over the lazy dog quickly",
       "Brown foxes are quick and dogs are lazy",
       "The dog is quick and brown"]
"""
##Solution 


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

### Sample text doc  #########3
doc = ["The quick brown fox jumps over the lazy dog",
       "Never jump over the lazy dog quickly",
       "Brown foxes are quick and dogs are lazy",
       "The dog is quick and brown"]

# Step 1: Converting  text doc into a TF-IDF matrix #########
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(doc)

###### original TF-IDF matrix ########
print("Original TF-IDF Matrix:\n", tfidf_matrix.toarray())

# Step 2: Apply TruncatedSVD
svd = TruncatedSVD(n_components=2)
reduced_matrix = svd.fit_transform(tfidf_matrix)

# Step 3: Reconstruct the original TF-IDF matrix from the reduced representation
reconstructed_matrix = svd.inverse_transform(reduced_matrix)

##### Step 4: Comparing original and reconstructed matrices   ########
print("\nReconstructed TF-IDF Matrix:\n", reconstructed_matrix)

###### Computing difference between the original and reconstructed matrix  #####
difference = np.abs(tfidf_matrix.toarray() - reconstructed_matrix)
print("\nDifference between Original and Reconstructed Matrix:\n", difference)

######### Sum of difference  ########
total_difference = np.sum(difference)
print(f"\nTotal difference between original and reconstructed matrix: {total_difference}")

###############################################
##############################################