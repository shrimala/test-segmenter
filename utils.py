import spacy
import docx2txt
import docx

def divide_into_sentences(input_text):
    nlp = spacy.load("cache/en_core_web_sm/en_core_web_sm-3.7.0")
    doc = nlp(input_text)
    sentences = [sent.text for sent in doc.sents]
    return sentences


def read_file(path:str):
    if path.endswith("docx"):
        with open(path, "rb") as f:
            txt = docx2txt.process(f)
    else:
        with open(path, "r") as f:
            txt = f.read()
    return txt

def write_docx(file_path_name:str, data:str):
    doc = docx.Document()
    doc.add_paragraph(data)
    doc.save(file_path_name)
    print("Saved :", file_path_name)