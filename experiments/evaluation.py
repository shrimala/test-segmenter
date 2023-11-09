from scorer import DeepTilingSegmentScorer
from text_length_segment import TextLengthSegment
from utils import read_file, divide_into_sentences
from glob import glob
from nltk.metrics.segmentation import windowdiff
import json
from tqdm import tqdm
files = glob("./experiments/books/**/*.txt")


def get_paragraphs(file):
    out = read_file(file)
    out = out.replace("\n========", "\n\n").split("\n\n")
    output = [x.replace("\n", "") for x in out if len(x.split()) > 10]
    return output


def paragraph_segmenter(input_text: list, distribution_n=10, distribution_p=0.2):
    # this is the top text embedding model from huggingface leaderboard, https://huggingface.co/spaces/mteb/leaderboard
    segment_scorer = DeepTilingSegmentScorer(
        parameters={}, text_data=input_text, encoding_model="BAAI/bge-large-en-v1.5"
    )
    text_length_segment = TextLengthSegment(
        distribution_n=distribution_n, distribution_p=distribution_p
    )
    output = text_length_segment.segment(
        text_data=segment_scorer.text_data, scores=segment_scorer.get_scores()
    )
    return output

def tokenize(data:list):
    output = []
    for para in data:
        k = [0]*len(para.split())
        k[-1] = 1
        output.extend(k)
    return "".join(map(str,output))

all_output = []
for file in tqdm(files):
    base_paragraphs = get_paragraphs(file)
    base_paragraphs = divide_into_sentences(" ".join(base_paragraphs))
    n_params = [5, 10, 15, 20]
    p_params = [0.1, 0.2, 0.5, 0.7]
    output = {}
    for dist_n in tqdm(n_params, leave=False):
        for dist_p in p_params:
            output_paragraphs = paragraph_segmenter(base_paragraphs, dist_n, dist_p)
            output[str((dist_n, dist_p))] = windowdiff(tokenize(base_paragraphs[:-1]), tokenize(output_paragraphs), 3)
    all_output.append(output)

with open("experiments/output.json", "w") as f:
    json.dump({"output":all_output}, f)


