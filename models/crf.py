"""Course-provided skeleton for Part A: CRF slot filling.

The constructor, `fit()`, and `predict()` are fully implemented — they
wire the model together and call into `sklearn-crfsuite`. The one thing
left to implement is `extract_features()`: turning a tokenized sentence
into a per-token feature dict. Per the README, feature engineering *is*
the Part A assignment — look for the `# TODO: implement` block below.
"""

from __future__ import annotations
from typing import Any, cast
import sklearn_crfsuite


class CRFModel:
    def __init__(self) -> None:
        """Initialize underlying `sklearn-crfsuite` CRF estimator.

        Reasonable defaults are set here; override them by editing this
        constructor or by reconstructing `self.model` with different
        hyperparameters as you experiment.
        """
        self.model = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True,
        )

    def extract_features(self, sentence: list[str]) -> list[dict[str, Any]]:
        """Convert a tokenized sentence into one feature dict per token.

        This is the part you implement.

        Args:
            sentence: A list of token strings, e.g. ["show", "me", "flights", "to", "boston"].

        Returns:
            A list the same length as `sentence`, where each element is a
            dict of features for the token at that position. Typical
            features to consider: word identity, casing, prefixes/suffixes,
            surrounding-token context window, gazetteers if you build them.
        """
        # ------------------------------------------------------------------
        # TODO: implement
        features = []
        
        for index, word in enumerate(sentence):
            first_char = word[0] if word else ""
            last_char = word[-1] if word else ""
            token_features = {
                "word": word,
                "is_first": index == 0,
                "is_last": index == len(sentence) - 1,
                "is_capitalized": first_char.isupper(),
                "is_all_caps": word.isupper(),
                "is_all_lower": word.islower(),
                "prefix-1": first_char,
                "prefix-2": word[:2],
                "prefix-3": word[:3],
                "suffix-1": last_char,
                "suffix-2": word[-2:],
                "suffix-3": word[-3:],
            }
            if index > 0:
                previous_word = sentence[index - 1]
                previous_first_char = previous_word[0] if previous_word else ""
                token_features.update({
                    "prev_word": previous_word,
                    "prev_is_capitalized": previous_first_char.isupper(),
                })
                
            if index < len(sentence) - 1:
                next_word = sentence[index + 1]
                next_first_char = next_word[0] if next_word else ""
                token_features.update({
                    "next_word": next_word,
                    "next_is_capitalized": next_first_char.isupper(),
                })
            if index > 1:
                previous_word_2 = sentence[index - 2]
                previous_first_char = previous_word_2[0] if previous_word_2 else ""
                token_features.update({
                    "prev_word_2": previous_word_2,
                    "prev_is_capitalized_2": previous_first_char.isupper(),
                })
                
            if index < len(sentence) - 2:
                next_word_2 = sentence[index + 2]
                next_first_char = next_word_2[0] if next_word_2 else ""
                token_features.update({
                    "next_word_2": next_word_2,
                    "next_is_capitalized_2": next_first_char.isupper(),
                })
            
            features.append(token_features)
        
        return features
        # ------------------------------------------------------------------
        raise NotImplementedError("extract_features() is left for you to implement.")

    def fit(self, sentences: list[list[str]], labels: list[list[str]]) -> None:
        """Train CRF on tokenized sentences and their BIO label sequences.

        Args:
            sentences: A list of tokenized sentences.
            labels: A list of BIO label sequences, one per sentence, aligned
                token-for-token with `sentences`.
        """
        features = [self.extract_features(sentence) for sentence in sentences]
        self.model.fit(features, labels)

    def predict(self, sentences: list[list[str]], vocab: Any = None) -> list[list[str]]:
        """Predict BIO label sequences for tokenized sentences.

        Args:
            sentences: A list of tokenized sentences.
            vocab: Unused. Accepted only so `CRFModel.predict()` and
                `BiLSTMModel.predict()` share the exact same call signature —
                sklearn-crfsuite works on raw token features, not vocab ids,
                so there's nothing for CRF to look up here. This is
                intentional, not a bug.

        Returns:
            A list of predicted BIO label sequences, one per sentence,
            aligned token-for-token with `sentences`.
        """
        features = [self.extract_features(sentence) for sentence in sentences]
        return cast(list[list[str]], self.model.predict(features))
