import numpy as np


class TextLengthSegment:
    def __init__(
        self,
        distribution_n: int,
        distribution_p: float,
        beam_width: int = 5,
    ) -> None:
        self.distribution_n = distribution_n
        self.distribution_p = distribution_p
        self.beam_width = beam_width

    def segment(self, text_data, scores, **kwargs):
        segments = self.beam_search_segmentation(text_data, scores, self.beam_width)
        return self.get_segmented_text(text_data, segments)

    def calculate_negative_binomial_prob(self, segmentation):
        return (
            np.math.comb(segmentation + self.distribution_n - 1, segmentation)
            * (self.distribution_p**segmentation)
            * ((1 - self.distribution_p) ** self.distribution_n)
        )

    def beam_search_segmentation(self, texts, scores, beam_width):
        num_segments = len(texts)

        initial_beam = [([0], 0)]  # (segment_indices, score)

        best_segmentation = None
        best_score = -float("inf")

        for _ in range(num_segments - 1):
            new_beam = []

            for segment_indices, score in initial_beam:
                last_segment_index = segment_indices[-1]

                for i in range(last_segment_index + 1, num_segments):
                    section_length = i - last_segment_index
                    section_length = max(
                        section_length, 1
                    )  # Ensure at least one segment per section

                    new_score = score + scores[
                        i - 1
                    ] * self.calculate_negative_binomial_prob(section_length)

                    new_segment_indices = segment_indices + [i]
                    new_beam.append((new_segment_indices, new_score))

                    # Check if this is the best segmentation so far
                    if new_score > best_score:
                        best_segmentation = new_segment_indices
                        best_score = new_score

            # Prune the beam to keep the top beam_width candidates
            new_beam.sort(key=lambda x: x[1], reverse=True)
            initial_beam = new_beam[:beam_width]

        return best_segmentation

    def get_segmented_text(self, text_data, best_segmentation):
        segmented_text = []
        start = 0
        for end in best_segmentation[1:]:
            segmented_text.append(" ".join(text_data[start:end]))
            start = end
        return segmented_text
