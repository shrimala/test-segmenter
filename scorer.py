from deeptiling import DeepTiling
class DeepTilingSegmentScorer:
    def __init__(self, text_data, encoding_model, **kwargs) -> None:
        self.deep_tiling = DeepTiling(
            encoding_model=encoding_model,
            nxt_sentence_prediction=kwargs.get("nxt_sentence_prediction", False),
        )
        self.results = self.predict(text_data, **kwargs)

    def predict(self, text_data, **kwargs):
        output = self.deep_tiling.predict(
            data=text_data,
            parameters={
                "window": kwargs.get("window", 10),
                "threshold": kwargs.get("threshold", 1),
            }
        )
        return output

    def get_scores(self):
        return self.results["scores"]
