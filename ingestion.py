import requests
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

# 환경 변수 로드
load_dotenv()

# 환경 변수 받아오기
BIBLE_ID = os.getenv("BIBLE_ID")
BIBLE_API_KEY = os.getenv("BIBLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "bible-rag")



# def get_bible_books():
#   url = f"https://rest.api.bible/v1/bibles/{BIBLE_ID}/books?include-chapters=true" # 요청할 API URL

#   try:
#     # 기본 GET 요청
#     response = requests.get(url, headers={"accept": "application/json", "api-key": f"{BIBLE_API_KEY}"})
    
#         # 상태 코드 확인
#     if response.status_code == 200:
#     # JSON 형식으로 응답 받기
#       data = response.json()['data']
#       print("응답 데이터:", data)
#     else:
#       print(f"오류: 상태 코드 {response.status_code}")
        
#   except requests.exceptions.RequestException as e:
#     print(f"요청 실패: {e}")

def get_bible_contents(chapterId: str):
  url = f"https://rest.api.bible/v1/bibles/{BIBLE_ID}/chapters/{chapterId}?content-type=text"

  try:
    response = requests.get(url, headers={"accept": "application/json", "api-key": f"{BIBLE_API_KEY}"})
    if response.status_code == 200:
      data = response.json()['data']
      print("응답 데이터:", data)
      return data
    else:
      print(f"오류: 상태 코드 {response.status_code}")
  except requests.exceptions.RequestException as e:
    print(f"요청 실패: {e}")
  return None


def convert_to_documents(bible_data: dict, chapter_id: str) -> Document:
    print("concerting to document...")
    content = bible_data.get("content", "")
    reference = bible_data.get("reference", "")
    
    # 메타데이터 설정
    metadata = {
        "chapter_id": chapter_id,
        "reference": reference,
        "source": "Bible API"
    }
    
    return Document(page_content=content, metadata=metadata)


# Pinecone Vector Store에 저장
def save_to_pinecone(documents: list[Document], namespace: str = "bible"):
  try:
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = PineconeVectorStore(
      index_name=PINECONE_INDEX,
      embedding=embeddings,
      namespace=namespace,
    )
    ids = [doc.metadata.get("chapter_id", f"doc-{i}") for i, doc in enumerate(documents)]
    vector_store.add_documents(documents=documents, ids=ids)
    print(f"✅ {len(documents)}개의 문서를 Pinecone에 저장했습니다.")
    return vector_store
  except Exception as e:
    print(f"❌ Pinecone 저장 중 오류: {e}")


# 배치 저장 (여러 장을 한 번에 저장)
def save_bible_batch_to_pinecone(start_chapter: str, num_chapters: int = 5):
    """
    여러 장의 성경 내용을 한 번에 Pinecone에 저장
    
    Args:
        start_chapter: 시작 장 (예: "GEN.1")
        num_chapters: 저장할 장의 수
    """
    documents = []
    current_chapter = start_chapter
    
    print(f"성경 {num_chapters}개 장을 임베딩하여 저장 중...")
    
    for i in range(num_chapters):
        if current_chapter is None:
            break
        
        # 성경 내용 받아오기
        bible_data = get_bible_contents(current_chapter)
        
        if bible_data:
            # Document로 변환
            doc = convert_to_documents(bible_data, current_chapter)
            documents.append(doc)
            
            # 다음 장으로 이동
            if "next" in bible_data:
                current_chapter = bible_data["next"]["id"]
            else:
                current_chapter = None
        
        # 배치 크기가 5개마다 저장
        if len(documents) >= 5 or i == num_chapters - 1:
            save_to_pinecone(documents, namespace="bible")
            documents = []  # 초기화
    
    print("✅ 모든 성경 내용이 Pinecone에 저장되었습니다.")




if __name__ == "__main__":
  # 방법 1: 처음 10개 장을 Pinecone에 저장
  save_bible_batch_to_pinecone(start_chapter="GEN.1", num_chapters=10000)
  
  # 방법 2: 단일 문서 저장
  # bible_data = get_bible_contents("GEN.1")
  # if bible_data:
  #     doc = convert_to_documents(bible_data, "GEN.1")
  #     init_pinecone()
  #     save_to_pinecone([doc], namespace="bible")
