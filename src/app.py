import os
import pickle
from dotenv import load_dotenv

import gradio as gr
from extract_text import extract_text_from_pdf
from split_text import split_text
from create_embeddings import create_vectorstore
from qa_chain import create_qa_chain, answer_question
import logging

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# .envファイルから環境変数を読み込む
load_dotenv()

# 環境変数からOpenAI APIキーを取得
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    raise ValueError("OpenAI APIキーが設定されていません。環境変数 'OPENAI_API_KEY' を設定してください。")


# 現在のスクリプトのディレクトリを取得
script_dir = os.path.dirname(os.path.abspath(__file__))
# キャッシュファイルのパス
VECTORSTORE_PATH = os.path.join(script_dir, '..', 'data', 'faiss_index')

# PDFからテキストを抽出
# PDFファイルへのパスを構築
pdf_path = os.path.join(script_dir, '..', 'data', 'aqua_202005.pdf')

# ベクトルストアを作成
if os.path.exists(VECTORSTORE_PATH):
    logger.info("ベクトルストアを読み込んでいます...")
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True # ValueError: The de-serialization relies loading a pickle file
        )
else:
    # PDFからテキストを抽出
    pdf_text = extract_text_from_pdf(pdf_path, start_page=391, end_page=469)

    # テキストを分割
    texts = split_text(pdf_text)

    # ベクトルストアを作成
    logger.info("ベクトルストアを作成しています...")
    vectorstore = create_vectorstore(texts, OPENAI_API_KEY)

    # ベクトルストアを保存
    vectorstore.save_local(VECTORSTORE_PATH)
    
# 質問応答チェーンを作成
logger.info("質問応答チェーンを作成")
qa_chain = create_qa_chain(vectorstore, OPENAI_API_KEY)

# Gradioインターフェースの定義
def qa_interface(question):
    return answer_question(qa_chain, question)

iface = gr.Interface(
    fn=qa_interface,
    inputs="text",
    outputs="text",
    title="ToyotaマニュアルQAシステム"
)

def main():
    iface.launch()

if __name__ == "__main__":
    main()
