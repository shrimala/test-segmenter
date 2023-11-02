from DeepTiling.models.DeepTilingModels import DeepTiling
import pandas as pd


class DeepTilingSegmentScorer(DeepTiling):
    def __init__(
        self,
        parameters,
        text_data,
        encoding_model="paraphrase-xlm-r-multilingual-v1",
        nxt_sentence_prediction=False,
    ):
        super().__init__(encoding_model, nxt_sentence_prediction)
        self.text_data = text_data
        self.parameters = parameters

    def get_scores(self):
        embs = self.encoder.encode(self.text_data)
        embs = pd.DataFrame(embs)
        scores, depth_scores = self.compute_depth_score(
            embs,
            window=self.parameters.get("window", 10),
            clip=self.parameters.get("clip", 2),
            single=self.parameters.get("multi_encode", False),
            smooth=self.parameters.get("smooth", False),
        )
        return [0.0]+scores
