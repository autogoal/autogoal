# from sklearn.pipeline import Pipeline as _Pipeline

# from . import _generated as sk
# from autogoal.grammar import Union


# class Noop:
#     def __repr__(self):
#         return "Noop()"

#     def fit(self, X, y=None):
#         pass

#     def transform(self, X, y=None):
#         return X

#     def fit_transform(self, X, y=None):
#         return X


# class DataPreprocessing(_Pipeline):
#     def __init__(
#         self,
#         encoding: Union("Encoder", Noop, sk.OneHotEncoder),
#         rescaling: Union("Rescaler", Noop, sk.MinMaxScaler, sk.StandardScaler), #, sk.QuantileTransformer),
#         imputation: Union("Imputer", Noop, sk.KNNImputer)# , sk.SimpleImputer),
#     ):
#         self.encoding = encoding
#         self.rescaling = rescaling
#         self.imputation = imputation

#         super().__init__(steps=[
#             ('encoder', self.encoding),
#             ('rescaler', self.rescaling),
#             ('imputer', self.imputation),
#         ])


# Decomposition = Union("Decomposition", sk.FastICA, sk.PCA, sk.TruncatedSVD, sk.KernelPCA)
# FeatureSelection = Union("FeatureSelection", sk.FeatureAgglomeration) #, sk.PolynomialFeatures) # TODO: Nystrom
# FeaturePreprocessing = Union("FeaturePreprocessing", Noop, Decomposition, FeatureSelection)

# BayesClassifier = Union("BayesClassifier", sk.GaussianNB, sk.MultinomialNB, sk.ComplementNB, sk.BernoulliNB)
# LinearClassifier = Union("LinearClassifier", sk.SGDClassifier, sk.RidgeClassifier, sk.PassiveAggressiveClassifier, sk.LogisticRegression, sk.Lasso, sk.Perceptron)
# Classification = Union("Classification", BayesClassifier, LinearClassifier, sk.SVC, sk.DecisionTreeClassifier, sk.KNeighborsClassifier)


# class SklearnClassifier(_Pipeline):
#     def __init__(
#         self,
#         data_preprocessing: DataPreprocessing,
#         feature_preprocessing: FeaturePreprocessing,
#         classification: Classification
#     ):
#         self.data_preprocessing = data_preprocessing
#         self.feature_preprocessing = feature_preprocessing
#         self.classification = classification

#         super().__init__(steps = [
#             ('data_preprocesor', self.data_preprocessing),
#             ('feature_preprocesor', self.feature_preprocessing),
#             ('classifier', self.classification)
#         ])
