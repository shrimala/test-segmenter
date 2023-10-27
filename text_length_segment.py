import numpy as np


class TextLengthSegment:
    def __init__(
        self,
        distribution_n: int,
        distribution_p: float,
        distribution_units: str = "sentences",
        beam_width: int = 5,
        max_iteractions: int = 3,
    ) -> None:
        self.distribution_n = distribution_n
        self.distribution_p = distribution_p
        self.distribution_units = distribution_units

    def segment(self, text_data, scores, **kwargs):
        # Calculate the length of each segment based on the 'distribution_units'
        pass

    def calculate_negative_binomial_prob(self, segmentation):
        return (
            np.math.comb(segmentation + self.distribution_n - 1, segmentation)
            * (self.distribution_p**segmentation)
            * ((1 - self.distribution_p) ** self.distribution_n)
        )

    def calculate_section_length(self, segmentation, segments):
        pass

    def beam_search_segmentation(self, texts, scores, beam_width, max_iterations):
        pass
