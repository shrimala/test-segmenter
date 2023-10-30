from utils import divide_into_sentences
from scorer import DeepTilingSegmentScorer
from text_length_segment import TextLengthSegment
from utils import read_file, write_docx
from pathlib import Path




def paragraph_segmenter(input_text, distribution_n=10, distribution_p=0.2):
    sentences = divide_into_sentences(input_text)
    #this is the top text embedding model from huggingface leaderboard, https://huggingface.co/spaces/mteb/leaderboard
    segment_scorer = DeepTilingSegmentScorer(
        sentences, encoding_model="BAAI/bge-large-en-v1.5"
    )
    text_length_segment = TextLengthSegment(distribution_n=distribution_n, distribution_p=distribution_p)
    output = text_length_segment.segment(
        text_data=sentences, scores=segment_scorer.get_scores()
    )
    return output


if __name__ == "__main__":
    # read the docx file
    file = Path("data/textsamples/8211.docx")
    input_text = read_file(file)
    output_text = paragraph_segmenter(input_text)
    print("Output:\n", output_text)
    write_docx("results/"+file.parent.name + "/" + file.name, output_text)
    



