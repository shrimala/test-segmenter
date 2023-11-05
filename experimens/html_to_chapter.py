from langchain.document_loaders import UnstructuredHTMLLoader
from tqdm import tqdm
from collections import OrderedDict

book = UnstructuredHTMLLoader(
    "experimens/books/The Project Gutenberg eBook of The Book of Life, by Upton Sinclair..html"
)
docs = book.load()
print("Docs Count :", len(docs))
docs_splits = docs[0].page_content.replace("\n \n", "\n\n").split("\n\n")
chapters = OrderedDict()
book_content_title_end_idx = None


def remove_new_line(x):
    if "." in x:
        out = " ".join(map(lambda x: x.strip(), x.split("\n"))).split(".")[0].strip()
    else:
        out = " ".join(map(lambda x: x.strip(), x.split("\n"))).split(" ")
        out = " ".join(out[:2])
    return out


patience = 4
counts = 0


def chapter_name_capitalize(x):
    out = x.split(" ")
    if len(out) == 2:
        return out[0].capitalize() + " " + out[1].upper()
    else:
        return x


for doc_idx in range(len(docs_splits)):
    doc = docs_splits[doc_idx]
    doc = remove_new_line(doc)
    if doc.lower().startswith("chapter ") or doc.lower().startswith("section "):
        if doc not in chapters:
            chapters[chapter_name_capitalize(doc)] = ""
            book_content_title_end_idx = doc_idx + 1
    else:
        if len(chapters):
            if counts == patience:
                break
            else:
                counts += 1

if len(chapters) < 3:
    book_content_title_end_idx = 0

docs_splits = docs_splits[book_content_title_end_idx:]

book_end_1 = "APPENDIX"
book_end_2 = "index"
book_end_3 = "*** END OF THE PROJECT GUTENBERG EBOOK"
book_end_4 = "FOOTNOTES"


doc_idx = 0
chapter_name = None
for doc_idx in range(len(docs_splits)):
    doc = docs_splits[doc_idx]

    chap = remove_new_line(doc)

    if chap.lower().startswith("epilogue"):
        chapter_name = chapter_name_capitalize(chap)
        chapters[chapter_name] = doc
        continue
    if (
        chap.lower().startswith(book_end_2)
        or chap.lower().startswith(book_end_1.lower())
        or chap.lower().startswith(book_end_4.lower())
        or book_end_3 in doc
    ) and chapter_name is not None:
        break
    if chap.lower().startswith("chapter ") or doc.lower().startswith("section "):
        chapter_name = chapter_name_capitalize(chap)
        if book_content_title_end_idx == 0:
            chapters[chapter_name] = ""

    if chapter_name is not None:
        try:
            chapters[chapter_name] += f"\n\n{doc}"
        except KeyError:
            chapters[chapter_name] += f"\n\n{doc}"

print()