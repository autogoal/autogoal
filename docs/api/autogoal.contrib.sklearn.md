# `autogoal.contrib.sklearn`

This module contains wrappers for several estimators and transformers
from `scikit-learn`.

!!! warning
    Importing this module requires `sklearn` with a version equal or greater
    than `0.22`. You can either install it manually or with `pip install autogoal[sklearn]`.

Most of the classes and functions inside this module deal with the automatic
generation of wrappers and thus are considered private API.

The main public functionality exposed by this module is the function
[find_classes](/api/autogoal.contrib.sklearn/#find_classes), which allows to
enumerate the wrappers implemented in this module applying some filters.

!!! note
    You can manually import any wrapper class directly from `autogoal.contrib.sklearn._generated`
    buy beware that namespace changes wildly from version to version and classes in it
    might disappear or change their signature anytime.

## Classes

### [`ARDRegression`](../autogoal.contrib.sklearn.ARDRegression)
> Bayesian ARD regression.

### [`AffinityPropagation`](../autogoal.contrib.sklearn.AffinityPropagation)
> Perform Affinity Propagation Clustering of data.

### [`BayesianRidge`](../autogoal.contrib.sklearn.BayesianRidge)
> Bayesian ridge regression.

### [`BernoulliNB`](../autogoal.contrib.sklearn.BernoulliNB)
> Naive Bayes classifier for multivariate Bernoulli models.

### [`Birch`](../autogoal.contrib.sklearn.Birch)
> Implements the Birch clustering algorithm.

### [`CategoricalNB`](../autogoal.contrib.sklearn.CategoricalNB)
> Naive Bayes classifier for categorical features

### [`ComplementNB`](../autogoal.contrib.sklearn.ComplementNB)
> The Complement Naive Bayes classifier described in Rennie et al. (2003).

### [`CountVectorizer`](../autogoal.contrib.sklearn.CountVectorizer)
> Convert a collection of text documents to a matrix of token counts

### [`CountVectorizerNoTokenize`](../autogoal.contrib.sklearn.CountVectorizerNoTokenize)
> Convert a collection of text documents to a matrix of token counts

### [`DecisionTreeClassifier`](../autogoal.contrib.sklearn.DecisionTreeClassifier)
> A decision tree classifier.

### [`DecisionTreeRegressor`](../autogoal.contrib.sklearn.DecisionTreeRegressor)
> A decision tree regressor.

### [`ElasticNet`](../autogoal.contrib.sklearn.ElasticNet)
> Linear regression with combined L1 and L2 priors as regularizer.

### [`ExtraTreeClassifier`](../autogoal.contrib.sklearn.ExtraTreeClassifier)
> An extremely randomized tree classifier.

### [`ExtraTreeRegressor`](../autogoal.contrib.sklearn.ExtraTreeRegressor)
> An extremely randomized tree regressor.

### [`FactorAnalysis`](../autogoal.contrib.sklearn.FactorAnalysis)
> Factor Analysis (FA)

### [`FastICA`](../autogoal.contrib.sklearn.FastICA)
> FastICA: a fast algorithm for Independent Component Analysis.

### [`FeatureAgglomeration`](../autogoal.contrib.sklearn.FeatureAgglomeration)
> Agglomerate features.

### [`FlagsDenseVectorizer`](../autogoal.contrib.sklearn.FlagsDenseVectorizer)
### [`FlagsSparseVectorizer`](../autogoal.contrib.sklearn.FlagsSparseVectorizer)
### [`GaussianNB`](../autogoal.contrib.sklearn.GaussianNB)
> Gaussian Naive Bayes (GaussianNB)

### [`HashingVectorizer`](../autogoal.contrib.sklearn.HashingVectorizer)
> Convert a collection of text documents to a matrix of token occurrences

### [`HuberRegressor`](../autogoal.contrib.sklearn.HuberRegressor)
> Linear regression model that is robust to outliers.

### [`IncrementalPCA`](../autogoal.contrib.sklearn.IncrementalPCA)
> Incremental principal components analysis (IPCA).

### [`Isomap`](../autogoal.contrib.sklearn.Isomap)
> Isomap Embedding

### [`KBinsDiscretizer`](../autogoal.contrib.sklearn.KBinsDiscretizer)
> Bin continuous data into intervals.

### [`KMeans`](../autogoal.contrib.sklearn.KMeans)
> K-Means clustering.

### [`KNNImputer`](../autogoal.contrib.sklearn.KNNImputer)
> Imputation for completing missing values using k-Nearest Neighbors.

### [`KNeighborsClassifier`](../autogoal.contrib.sklearn.KNeighborsClassifier)
> Classifier implementing the k-nearest neighbors vote.

### [`KNeighborsRegressor`](../autogoal.contrib.sklearn.KNeighborsRegressor)
> Regression based on k-nearest neighbors.

### [`KNeighborsTransformer`](../autogoal.contrib.sklearn.KNeighborsTransformer)
> Transform X into a (weighted) graph of k nearest neighbors

### [`KernelCenterer`](../autogoal.contrib.sklearn.KernelCenterer)
> Center a kernel matrix

### [`KernelPCA`](../autogoal.contrib.sklearn.KernelPCA)
> Kernel Principal component analysis (KPCA)

### [`LabelBinarizer`](../autogoal.contrib.sklearn.LabelBinarizer)
> Binarize labels in a one-vs-all fashion

### [`Lars`](../autogoal.contrib.sklearn.Lars)
> Least Angle Regression model a.k.a. LAR

### [`Lasso`](../autogoal.contrib.sklearn.Lasso)
> Linear Model trained with L1 prior as regularizer (aka the Lasso)

### [`LassoLars`](../autogoal.contrib.sklearn.LassoLars)
> Lasso model fit with Least Angle Regression a.k.a. Lars

### [`LassoLarsIC`](../autogoal.contrib.sklearn.LassoLarsIC)
> Lasso model fit with Lars using BIC or AIC for model selection

### [`LatentDirichletAllocation`](../autogoal.contrib.sklearn.LatentDirichletAllocation)
> Latent Dirichlet Allocation with online variational Bayes algorithm

### [`LinearRegression`](../autogoal.contrib.sklearn.LinearRegression)
> Ordinary least squares Linear Regression.

### [`LinearSVC`](../autogoal.contrib.sklearn.LinearSVC)
> Linear Support Vector Classification.

### [`LinearSVR`](../autogoal.contrib.sklearn.LinearSVR)
> Linear Support Vector Regression.

### [`LocalOutlierFactor`](../autogoal.contrib.sklearn.LocalOutlierFactor)
> Unsupervised Outlier Detection using Local Outlier Factor (LOF)

### [`LocallyLinearEmbedding`](../autogoal.contrib.sklearn.LocallyLinearEmbedding)
> Locally Linear Embedding

### [`LogisticRegression`](../autogoal.contrib.sklearn.LogisticRegression)
> Logistic Regression (aka logit, MaxEnt) classifier.

### [`MeanShift`](../autogoal.contrib.sklearn.MeanShift)
> Mean shift clustering using a flat kernel.

### [`MinMaxScaler`](../autogoal.contrib.sklearn.MinMaxScaler)
> Transform features by scaling each feature to a given range.

### [`MiniBatchKMeans`](../autogoal.contrib.sklearn.MiniBatchKMeans)
> Mini-Batch K-Means clustering.

### [`MiniBatchSparsePCA`](../autogoal.contrib.sklearn.MiniBatchSparsePCA)
> Mini-batch Sparse Principal Components Analysis

### [`MultinomialNB`](../autogoal.contrib.sklearn.MultinomialNB)
> Naive Bayes classifier for multinomial models

### [`NMF`](../autogoal.contrib.sklearn.NMF)
> Non-Negative Matrix Factorization (NMF)

### [`NearestCentroid`](../autogoal.contrib.sklearn.NearestCentroid)
> Nearest centroid classifier.

### [`NuSVC`](../autogoal.contrib.sklearn.NuSVC)
> Nu-Support Vector Classification.

### [`NuSVR`](../autogoal.contrib.sklearn.NuSVR)
> Nu Support Vector Regression.

### [`OneClassSVM`](../autogoal.contrib.sklearn.OneClassSVM)
> Unsupervised Outlier Detection.

### [`OneHotEncoder`](../autogoal.contrib.sklearn.OneHotEncoder)
> Encode categorical features as a one-hot numeric array.

### [`OrdinalEncoder`](../autogoal.contrib.sklearn.OrdinalEncoder)
> Encode categorical features as an integer array.

### [`OrthogonalMatchingPursuit`](../autogoal.contrib.sklearn.OrthogonalMatchingPursuit)
> Orthogonal Matching Pursuit model (OMP)

### [`PCA`](../autogoal.contrib.sklearn.PCA)
> Principal component analysis (PCA).

### [`PassiveAggressiveClassifier`](../autogoal.contrib.sklearn.PassiveAggressiveClassifier)
> Passive Aggressive Classifier

### [`PassiveAggressiveRegressor`](../autogoal.contrib.sklearn.PassiveAggressiveRegressor)
> Passive Aggressive Regressor

### [`Perceptron`](../autogoal.contrib.sklearn.Perceptron)
> Perceptron

### [`PowerTransformer`](../autogoal.contrib.sklearn.PowerTransformer)
> Apply a power transform featurewise to make data more Gaussian-like.

### [`RadiusNeighborsRegressor`](../autogoal.contrib.sklearn.RadiusNeighborsRegressor)
> Regression based on neighbors within a fixed radius.

### [`RadiusNeighborsTransformer`](../autogoal.contrib.sklearn.RadiusNeighborsTransformer)
> Transform X into a (weighted) graph of neighbors nearer than a radius

### [`Ridge`](../autogoal.contrib.sklearn.Ridge)
> Linear least squares with l2 regularization.

### [`RidgeClassifier`](../autogoal.contrib.sklearn.RidgeClassifier)
> Classifier using Ridge regression.

### [`RobustScaler`](../autogoal.contrib.sklearn.RobustScaler)
> Scale features using statistics that are robust to outliers.

### [`SGDClassifier`](../autogoal.contrib.sklearn.SGDClassifier)
> Linear classifiers (SVM, logistic regression, a.o.) with SGD training.

### [`SGDRegressor`](../autogoal.contrib.sklearn.SGDRegressor)
> Linear model fitted by minimizing a regularized empirical loss with SGD

### [`SVC`](../autogoal.contrib.sklearn.SVC)
> C-Support Vector Classification.

### [`SVR`](../autogoal.contrib.sklearn.SVR)
> Epsilon-Support Vector Regression.

### [`SparsePCA`](../autogoal.contrib.sklearn.SparsePCA)
> Sparse Principal Components Analysis (SparsePCA)

### [`StandardScaler`](../autogoal.contrib.sklearn.StandardScaler)
> Standardize features by removing the mean and scaling to unit variance

### [`TfidfTransformer`](../autogoal.contrib.sklearn.TfidfTransformer)
> Transform a count matrix to a normalized tf or tf-idf representation

### [`TfidfVectorizer`](../autogoal.contrib.sklearn.TfidfVectorizer)
> Convert a collection of raw documents to a matrix of TF-IDF features.

### [`TheilSenRegressor`](../autogoal.contrib.sklearn.TheilSenRegressor)
> Theil-Sen Estimator: robust multivariate regression model.

### [`TruncatedSVD`](../autogoal.contrib.sklearn.TruncatedSVD)
> Dimensionality reduction using truncated SVD (aka LSA).

