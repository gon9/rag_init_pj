import os
import pdfplumber
import logging

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path, start_page=391, end_page=469):
    """
    指定した範囲のPDFページからテキストを抽出します。

    Args:
        pdf_path (str): PDFファイルのパス。
        start_page (int): 抽出を開始するページ番号（0始まり）。
        end_page (int): 抽出を終了するページ番号（0始まり）。

    Returns:
        str: 抽出されたテキスト。
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        # ページ範囲を設定
        pages = pdf.pages[start_page:end_page]
        logger.info("pdfファイルを読み込んでいます...")
        for page in pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
                # print(page_text)
    logger.info(f"pdfファイルを読み込み完了。{len(text)}文字。)")
    return text


script_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(script_dir, '..', 'data', 'aqua_202005.pdf')

from split_text import split_text

pdf_text = extract_text_from_pdf(pdf_path, start_page=391, end_page=469)
texts = split_text(pdf_text)

print(texts[0])
print(texts[1])
print(len(texts))
