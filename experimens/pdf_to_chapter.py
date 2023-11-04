from langchain.document_loaders import PDFMinerPDFasHTMLLoader
from bs4 import BeautifulSoup
from glob import glob
from tqdm import tqdm
import json
import re
from langchain.docstore.document import Document


def get_chapter_docs(path_dir):
    files = glob(path_dir + "/*.pdf")
    print("files :", files, sep="\n")
    chapters = []
    for file in tqdm(files):
        loader = PDFMinerPDFasHTMLLoader(file)
        data = loader.load()[0]  # entire PDF is loaded as a single Document
        soup = BeautifulSoup(data.page_content, "html.parser")
        content = soup.find_all("div")
        cur_fs = None
        cur_text = ""
        snippets = []  # first collect all snippets that have the same font size
        for c in tqdm(content, total=len(content), leave=False):
            sp = c.find("span")
            if not sp:
                continue
            st = sp.get("style")
            if not st:
                continue
            fs = re.findall("font-size:(\d+)px", st)
            if not fs:
                continue
            fs = int(fs[0])
            if not cur_fs:
                cur_fs = fs
            if fs == cur_fs:
                cur_text += c.text
            else:
                snippets.append((cur_text, cur_fs))
                cur_fs = fs
                cur_text = c.text
        snippets.append((cur_text, cur_fs))
        # Note: The above logic is very straightforward. One can also add more strategies such as removing duplicate snippets (as
        # headers/footers in a PDF appear on multiple pages so if we find duplicates it's safe to assume that it is redundant info)

        cur_idx = -1
        semantic_snippets = []
        # Assumption: headings have higher font size than their respective content
        for s in tqdm(snippets, total=len(snippets), leave=False):
            # if current snippet's font size > previous section's heading => it is a new heading
            if (
                not semantic_snippets
                or s[1] > semantic_snippets[cur_idx].metadata["heading_font"]
            ):
                metadata = {"heading": s[0], "content_font": 0, "heading_font": s[1]}
                metadata.update(data.metadata)
                semantic_snippets.append(Document(page_content="", metadata=metadata))
                cur_idx += 1
                continue

            # if current snippet's font size <= previous section's content => content belongs to the same section (one can also create
            # a tree like structure for sub sections if needed but that may require some more thinking and may be data specific)
            if (
                not semantic_snippets[cur_idx].metadata["content_font"]
                or s[1] <= semantic_snippets[cur_idx].metadata["content_font"]
            ):
                semantic_snippets[cur_idx].page_content += s[0]
                semantic_snippets[cur_idx].metadata["content_font"] = max(
                    s[1], semantic_snippets[cur_idx].metadata["content_font"]
                )
                continue

            # if current snippet's font size > previous section's content but less than previous section's heading than also make a new
            # section (e.g. title of a PDF will have the highest font size but we don't want it to subsume all sections)
            metadata = {"heading": s[0], "content_font": 0, "heading_font": s[1]}
            metadata.update(data.metadata)
            semantic_snippets.append(Document(page_content="", metadata=metadata))
            cur_idx += 1

        assert (
            len(semantic_snippets) > 5
        ), f"number of docs {len(semantic_snippets)}, are less than 5"
        chapter_docs = []
        for doc in semantic_snippets:
            heading = doc.metadata["heading"]
            content = doc.page_content
            if "chapter" in heading.lower().split() and len(content) > 100:
                chapter_docs.append(doc.__dict__)
        chapters.extend(chapter_docs)
    return chapters


if __name__ == "__main__":
    chapters = get_chapter_docs("experimens/books")
    json.dump({"chapters": chapters}, open("experimens/books/chapters.json", "w"))
