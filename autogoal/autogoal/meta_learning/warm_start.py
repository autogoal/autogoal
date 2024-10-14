from typing import Dict, List, Optional
from autogoal.meta_learning._experience import Experience, ExperienceStore
from autogoal.meta_learning.feature_extraction.text_classification import (
    TextClassificationFeatureExtractor,
)
from autogoal.meta_learning.normalization import Normalizer
from autogoal.meta_learning.distance import DistanceMetric, EuclideanDistance
from autogoal.meta_learning.feature_extraction.system_feature_extractor import (
    SystemFeatureExtractor,
)
from autogoal.meta_learning.sampling import ExperienceReplayModelSampler
from autogoal.meta_learning import FeatureExtractor
from autogoal.sampling import (
    DistributionParam,
    MeanDevParam,
    ModelSampler,
    WeightParam,
    update_model,
)
import numpy as np


class WarmStart:
    """
    A class to adjust the internal probabilistic model of an AutoML process when starting
    based on relevant past experiences (warm starting).

    This class "warm-starting" method follows 5 steps to adjust the internal
    the AutoML process. Namely:

    1. Extracting meta-features from the current dataset/system.
    2. Computing distances between the current dataset/system and past experiences.
    3. Selecting the most relevant experiences based on distance and accuracy threshold.
    4. Computing learning rates (alphas) for adjusting the sampler.
    5. Adjusting the internal probabilistic model accordingly for each experience in their order of relevance (alpha).

    Parameters:
        threshold (float, optional): Minimum accuracy threshold for considering an experience.
            Experiences with accuracy below this threshold will be ignored. Default is `0.2`.
        k (int, optional): The maximum number of past experiences to consider.
            Default is `20`.
        max_alpha (float, optional): The maximum learning rate (alpha) used when adjusting
            the model. Default is `0.5`.
        normalizers (Optional[List[Normalizer]], optional): A list of normalizer instances
            to apply to the features before computing distances. Default is an empty list.
        distance (DistanceMetric, optional): The distance metric class to use when computing
            distances between feature vectors. Default is `EuclideanDistance`.
        dataset_feature_extractor (Optional[FeatureExtractor], optional): The feature extractor
            class to use for extracting dataset features. Default is `TextClassificationFeatureExtractor`.
        system_feature_extractor (Optional[FeatureExtractor], optional): The feature extractor
            class to use for extracting system features. Default is `SystemFeatureExtractor`.

    Attributes:
        _model (Dict): The internal probabilistic model that will be adjusted.
        generator_fn (callable): The function used to generate configurations during the warm-up.
        threshold (float): The accuracy threshold.
        k (int): The maximum number of experiences to consider.
        max_alpha (float): The maximum learning rate.
        normalizers (List[Normalizer]): List of normalizers for feature normalization.
        distance (DistanceMetric): The distance metric instance.
        dataset_feature_extractor_class (FeatureExtractor): Class for dataset feature extraction.
        system_feature_extractor_class (FeatureExtractor): Class for system feature extraction.
        X_train: Training data features of the current dataset.
        y_train: Training data labels of the current dataset.
    """

    def __init__(
        self,
        positive_min_threshold=0.2,
        k_pos=20,
        k_neg=20,
        max_alpha=0.03,
        min_alpha=-0.01,
        normalizers: Optional[List[Normalizer]] = None,
        distance: DistanceMetric = EuclideanDistance,
        dataset_feature_extractor: Optional[
            FeatureExtractor
        ] = TextClassificationFeatureExtractor,
        system_feature_extractor: Optional[FeatureExtractor] = SystemFeatureExtractor,
    ):
        self._model: Dict = {}
        self.generator_fn = None
        self.positive_min_threshold = positive_min_threshold
        self.k_pos = k_pos
        self.k_neg = k_neg
        self.max_alpha = max_alpha
        self.min_alpha = min_alpha
        self.normalizers = normalizers or []
        self.distance = distance() if distance else EuclideanDistance()
        self.dataset_feature_extractor_class = dataset_feature_extractor
        self.system_feature_extractor_class = system_feature_extractor

    def pre_warm_up(self, X_train, y_train):
        """
        Stores the training data for later use during warm-up.

        Parameters:
            X_train: Training data features of the current dataset.
            y_train: Training data labels of the current dataset.

        Returns:
            None
        """
        self.X_train = X_train
        self.y_train = y_train

        # Step 1: Extract meta-features of the current dataset and current system
        self.current_dataset_features = self._extract_meta_features(
            self.X_train, self.y_train
        )
        self.current_system_features = self._extract_system_features()

    def warm_up(self, generator_fn):
        """
        Adjusts the internal probabilistic model based on relevant past experiences.

        This method performs the following steps:
        1. Extracts meta-features from the current dataset.
        2. Computes distances between the current dataset/system and past experiences.
        3. Selects the most relevant experiences based on distance and accuracy threshold.
        4. Computes learning rates (alphas) for adjusting the sampler.
        5. Adjusts the internal probabilistic model accordingly.

        Parameters:
            generator_fn (callable): A function that, given a sampler, generates configurations
                (e.g., the function that defines the search space).

        Returns:
            Dict: The updated internal probabilistic model.
        """
        self.generator_fn = generator_fn

        # Step 2: Load experiences
        experiences = ExperienceStore.load_all_experiences()

        # Step 2.1: Filter experiences based on feature extractors
        experiences = self.filter_experiences_by_feature_extractors(experiences)

        if not experiences:
            # No relevant experiences found
            return  # No need to adjust the model_sampler

        # Step 3: Compute distances and select relevant experiences
        distances = self.compute_distances(
            self.current_dataset_features, self.current_system_features, experiences
        )

        (
            selected_positive_experiences,
            positive_distances,
            selected_negative_experiences,
            negative_distances,
        ) = self.select_experiences(experiences, distances)

        if not selected_positive_experiences and not selected_negative_experiences:
            # No experiences to adjust with
            return

        # Step 4: Compute learning rates (alphas)
        alpha_experiences = self.compute_learning_rates(
            selected_positive_experiences,
            positive_distances,
            selected_negative_experiences,
            negative_distances,
        )

        # Step 5: Adjust the internal probabilistic model
        self.adjust_model(alpha_experiences)
        return self._model

    def _normalize_features(
        self, feature_vectors_list: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Applies the sequence of normalizers to a list of feature vectors.

        Parameters:
            feature_vectors_list (List[np.ndarray]): A list of feature vectors (numpy arrays).

        Returns:
            List[np.ndarray]: A list of normalized feature vectors.
        """
        # Stack feature vectors for fitting
        feature_matrix = np.vstack(feature_vectors_list)

        # Apply each normalizer sequentially
        for normalizer in self.normalizers:
            feature_matrix = normalizer.fit_transform(feature_matrix)

        # Split back into individual feature vectors
        num_vectors = len(feature_vectors_list)
        normalized_features_list = np.vsplit(feature_matrix, num_vectors)

        # Flatten each array in the list
        normalized_features_list = [vec.flatten() for vec in normalized_features_list]

        return normalized_features_list

    def _extract_meta_features(self, X_train, y_train):
        """
        Extracts meta-features from the current dataset using the specified feature extractor.

        Parameters:
            X_train: Training data features.
            y_train: Training data labels.

        Returns:
            np.ndarray: Extracted dataset meta-features.
        """
        extractor = self.dataset_feature_extractor_class()
        return extractor.extract_features(X_train, y_train)

    def _extract_system_features(self):
        """
        Extracts system features using the specified system feature extractor.

        Returns:
            np.ndarray: Extracted system features.
        """
        extractor = self.system_feature_extractor_class()
        return extractor.extract_features()

    def compute_distances(
        self,
        current_dataset_features,
        current_system_features,
        experiences: List[Experience],
    ):
        """
        Computes the total distances between the current dataset/system and each past experience,
        using the specified distance metric.

        Parameters:
            current_dataset_features (np.ndarray): Meta-features of the current dataset.
            current_system_features (np.ndarray): System features of the current system.
            experiences (List[Experience]): List of past experiences.

        Returns:
            List[float]: Distances corresponding to each experience.
        """
        # Collect all features for normalization
        all_dataset_features = [exp.dataset_features for exp in experiences]
        all_dataset_features.append(current_dataset_features)
        all_system_features = [exp.system_features for exp in experiences]
        all_system_features.append(current_system_features)

        # Normalize features
        normalized_dataset_features = self._normalize_features(all_dataset_features)
        normalized_system_features = self._normalize_features(all_system_features)

        # Update experiences with normalized features
        for i, exp in enumerate(experiences):
            exp.dataset_features = normalized_dataset_features[i]
            exp.system_features = normalized_system_features[i]

        # Get normalized current features
        current_dataset_features = normalized_dataset_features[-1]
        current_system_features = normalized_system_features[-1]

        distances = []
        for exp in experiences:
            exp_features = np.concatenate((exp.dataset_features, exp.system_features))
            current_features = np.concatenate(
                (current_dataset_features, current_system_features)
            )

            distance = self.distance.compute(current_features, exp_features)

            distances.append(distance)
        return distances

    def select_experiences(self, experiences: List[Experience], distances):
        # Pair each experience with its distance
        experience_distance_pairs = list(zip(experiences, distances))

        # Filter positive experiences based on F1 score threshold
        positive_experiences = [
            (exp, dist)
            for exp, dist in experience_distance_pairs
            if exp.f1 is not None
            and exp.f1 > -np.Infinity
            and exp.evaluation_time is not None
            and exp.evaluation_time < np.Infinity
            and exp.f1 >= self.positive_min_threshold
        ]

        # Filter negative experiences where F1 is None or evaluation_time is None
        negative_experiences = [
            (exp, dist)
            for exp, dist in experience_distance_pairs
            if exp.f1 is None
            or exp.f1 == -np.Infinity
            or exp.evaluation_time is None
            or exp.evaluation_time == np.Infinity
        ]

        # Sort positive and negative experiences by distance
        sorted_positive_experiences = sorted(positive_experiences, key=lambda x: x[1])
        sorted_negative_experiences = sorted(negative_experiences, key=lambda x: x[1])

        # Select top-k positive and negative experiences
        selected_positive = sorted_positive_experiences[: self.k_pos]
        selected_negative = sorted_negative_experiences[: self.k_neg]

        # Separate experiences and distances
        selected_positive_experiences = [exp for exp, dist in selected_positive]
        positive_distances = [dist for exp, dist in selected_positive]

        selected_negative_experiences = [exp for exp, dist in selected_negative]
        negative_distances = [dist for exp, dist in selected_negative]

        return (
            selected_positive_experiences,
            positive_distances,
            selected_negative_experiences,
            negative_distances,
        )

    def filter_experiences_by_feature_extractors(
        self, experiences: List[Experience]
    ) -> List[Experience]:
        """
        Filters experiences to include only those that used the same feature extractors.

        Parameters:
            experiences (List[Experience]): A list of past experiences.

        Returns:
            List[Experience]: A list of experiences that used the same feature extractors.
        """
        dataset_extractor_name = self.dataset_feature_extractor_class.__name__
        system_extractor_name = self.system_feature_extractor_class.__name__

        filtered_experiences = [
            exp
            for exp in experiences
            if exp.dataset_feature_extractor_name == dataset_extractor_name
            and exp.system_feature_extractor_name == system_extractor_name
        ]

        return filtered_experiences

    def compute_learning_rates(
        self,
        selected_positive_experiences: List[Experience],
        positive_distances,
        selected_negative_experiences: List[Experience],
        negative_distances,
    ):
        experience_alphas = {}

        # Handle positive experiences
        if selected_positive_experiences:
            f1_scores = [exp.f1 for exp in selected_positive_experiences]
            eval_times = [exp.evaluation_time for exp in selected_positive_experiences]

            # Normalize F1 scores
            max_f1 = max(f1_scores) or 1.0  # Prevent division by zero
            normalized_f1 = [f1 / max_f1 for f1 in f1_scores]

            # Normalize evaluation times (lower is better)
            min_time = min(eval_times)
            max_time = max(eval_times)
            time_range = max_time - min_time if max_time != min_time else 1.0
            normalized_time = [(time - min_time) / time_range for time in eval_times]

            # Invert normalized evaluation times
            normalized_time_inv = [1 - t for t in normalized_time]

            # Combine normalized F1 and inverted evaluation time into a utility score
            w1 = 0.5  # Weight for F1 score
            w2 = 0.5  # Weight for evaluation time
            utility_scores = [
                w1 * f1 + w2 * t_inv
                for f1, t_inv in zip(normalized_f1, normalized_time_inv)
            ]

            # Normalize distances
            dist_min = min(positive_distances)
            dist_max = max(positive_distances)
            dist_range = dist_max - dist_min if dist_max != dist_min else 1.0
            normalized_distances = [
                (dist - dist_min) / dist_range for dist in positive_distances
            ]

            # Compute learning rates (alphas) for positive experiences
            alphas = [
                self.max_alpha * utility * (1 - dist_norm)
                for utility, dist_norm in zip(utility_scores, normalized_distances)
            ]

            # Map positive experiences to their learning rates
            for exp, alpha in zip(selected_positive_experiences, alphas):
                experience_alphas[exp] = alpha

        # Handle negative experiences
        if selected_negative_experiences:
            # Normalize distances
            dist_min = min(negative_distances)
            dist_max = max(negative_distances)
            dist_range = dist_max - dist_min if dist_max != dist_min else 1.0
            normalized_distances = [
                (dist - dist_min) / dist_range for dist in negative_distances
            ]

            # Compute learning rates (alphas) for negative experiences
            alphas = [
                self.min_alpha * (1 - dist_norm) for dist_norm in normalized_distances
            ]

            # Map negative experiences to their learning rates
            for exp, alpha in zip(selected_negative_experiences, alphas):
                experience_alphas[exp] = alpha

        return experience_alphas

    def handle_error_experiences(self, experiences: List[Experience], alphas):
        """
        Assigns negative learning rates to experiences with errors (missing accuracy).

        The negative learning rate is equal in magnitude to the smallest positive learning rate
        among the successful experiences.

        Parameters:
        - experiences: A list of experiences (some may have 'accuracy' as None).
        - alphas: A list of computed learning rates for the experiences.

        Returns:
        - error_experience_alphas: A dictionary mapping error experiences to their negative learning rates.
        """
        # Find the minimum positive learning rate
        min_positive_alpha = min([alpha for alpha in alphas if alpha > 0], default=0)

        # Handle experiences with errors (missing accuracy)
        error_experience_alphas = {}
        for exp in experiences:
            if exp.accuracy is None:
                # Assign negative learning rate equal to the smallest positive alpha
                error_experience_alphas[exp] = -min_positive_alpha

        return error_experience_alphas

    def adjust_model(self, alpha_experiences: Dict[Experience, float]):
        """
        Adjusts the internal probabilistic model based on external experiences.

        For each experience, it uses its learning rate (alpha) to update the model parameters.

        Parameters:
            alpha_experiences (Dict[Experience, float]): A dictionary mapping experiences to their learning rates.

        Returns:
            None
        """

        # Extract experiences and alphas
        experience_alpha_pairs = list(alpha_experiences.items())
        experience_alpha_pairs.sort(key=lambda x: x[1], reverse=True)

        for experience, alpha in experience_alpha_pairs:
            # intialize the model sampler with the experience
            sampler = ExperienceReplayModelSampler(self._model)
            sampler.set_replicate_mode(experience)

            # generate the probabilistic model for the experience
            self.generator_fn(sampler)

            # update the warmstart model with the experience model
            self._model = update_model(self._model, sampler.updates, alpha)
