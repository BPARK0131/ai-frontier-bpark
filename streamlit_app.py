import os
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, String, Integer, DateTime
from transformers import AutoTokenizer
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Streamlit 애플리케이션 제목 설정
st.title("RAG 기반 질의응답 시스템")

# API_KEY.txt 파일을 환경 변수로 불러오는 함수 정의
def load_config(file_path):
    with open(file_path) as f:
        for line in f:
            key, value = line.strip().split('=')
            os.environ[key] = value

# 환경 변수 설정
load_config('/workspaces/ai-frontier-bpark/API_KEY.txt')
os.environ["TOKENIZERS_PARALLELISM"] = "true"

gip_base_url = "https://api.platform.a15t.com/v1"

# CSV 데이터 전처리 및 로드 (캐시 적용)
@st.cache_data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, delimiter=",", quotechar='"')
    df.fillna("", inplace=True)
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df.drop_duplicates(inplace=True)
    df['event_time'] = pd.to_datetime(df['event_time'])
    df['Event Year'] = df['event_time'].dt.year
    df['Event Quarter'] = df['event_time'].dt.to_period('Q').astype(str)
    df['Event Month'] = df['event_time'].dt.strftime('%Y-%m')
    df['gen'] = df['gen'].astype('category')
    df['role'] = df['role'].astype('category')
    df['syslog'] = df['syslog'].str.lower()
    df['keyword'] = df['keyword'].str.lower().replace({
        "link down": "link issue",
        "temperature alarm": "temp alarm"
    })
    df['Event Date'] = df['event_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df['after_action'] = df['after_action'].apply(lambda x: x[:500] if len(x) > 500 else x)
    return df

file_path = "/workspaces/ai-frontier-bpark/dummy_data_241018.csv"
df = load_and_preprocess_data(file_path)

# SQLite 데이터베이스 생성 및 설정 (캐시 적용)
@st.cache_resource
def create_sql_database(df):
    engine = create_engine("sqlite:///fault_data.db")
    df.to_sql("fault_events", con=engine, if_exists="replace", index=False, dtype={
        'event_time': DateTime,
        'gen': String,
        'vendor': String,
        'role': String,
        'device_name': String,
        'syslog': String,
        'card': String,
        'unit_name': String,
        'alarm_group': String,
        'keyword': String,
        'service_effect': String,
        'after_action': String,
        'action_result': String,
        'Event Year': Integer,
        'Event Quarter': String,
        'Event Month': String
    })
    return SQLDatabase(engine)

db = create_sql_database(df)

# 임베딩 및 벡터 스토어 생성 (캐시 적용)
@st.cache_resource
def create_vector_store(docs):
    model_name = "intfloat/multilingual-e5-large-instruct"
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        show_progress=True
    )
    vectorstore = FAISS.from_documents(documents=docs, embedding=hf_embeddings)
    return vectorstore, hf_embeddings

# 텍스트 분할기 생성 및 문서 분할
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large-instruct"),
    chunk_size=400,
    chunk_overlap=30
)
docs = [Document(page_content=row['syslog'], metadata={"event_time": row['event_time']}) for _, row in df.iterrows()]
splitted_docs = text_splitter.split_documents(docs)
vectorstore, hf_embeddings = create_vector_store(splitted_docs)

# 검색기 설정 (BM25 및 FAISS)
bm25_retriever = BM25Retriever.from_documents(splitted_docs)
bm25_retriever.k = 5
faiss_retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 5})

# 앙상블 검색기 생성
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.5, 0.5],
    search_type="similarity"
)

# LLM 설정 및 SQL 쿼리 관련 체인 정의
sql_llm = ChatOpenAI(model_name="azure/openai/gpt-4o-mini-2024-07-18",
                    streaming=True, callbacks=[StreamingStdOutCallbackHandler()],
                    temperature=0.2, base_url=gip_base_url)

# SQL 쿼리 생성 및 실행 툴 설정
execute_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain(sql_llm, db)

# 질문에 따른 쿼리 생성 템플릿 설정
answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.
If you don't know the answer, just say that you don't know.
Answer in Korean.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)
answer = answer_prompt | sql_llm | StrOutputParser()

# SQL 쿼리 정제 함수 정의
def clean_and_validate_query(query):
    clean_query = re.sub(r'```sql|```|SQLQuery:', '', query).strip()
    return clean_query.replace("\n", " ").strip()

# 라우팅을 위한 Runnable 설정
router_prompt = PromptTemplate(
    input_variables=["input"],
    template="다음 질문이 SQL 데이터베이스에서 조회할 수 있는지 여부를 판단하세요.\n질문: {input}\nSQL 조회가 가능하다면 'SQL'이라고 답하고, 그렇지 않다면 'RAG'라고 답하세요."
)
router_llm = ChatOpenAI(model_name="azure/openai/gpt-4o-mini-2024-07-18",
                    streaming=True, callbacks=[StreamingStdOutCallbackHandler()],
                    temperature=0.5, base_url=gip_base_url)
router_runnable = router_prompt | router_llm | StrOutputParser()

# RAG 프롬프트 생성 및 체인 정의
rag_prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Answer in Korean.

#Question:
{question}
#Context:
{context}

#Answer:"""
)
rag_llm = ChatOpenAI(model_name="azure/openai/gpt-4o-mini-2024-07-18",
                    streaming=True, callbacks=[StreamingStdOutCallbackHandler()],
                    temperature=0, base_url=gip_base_url)
rag_chain = (
    {"context": ensemble_retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | rag_llm
    | StrOutputParser()
)

# 최종 체인 라우팅
multi_prompt_chain = RunnableLambda(
    lambda inputs: query_runnable.invoke({"question": inputs["input"]})
    if router_runnable.invoke(inputs) == "SQL"
    else rag_chain.invoke(inputs["input"])
)

# 최종 답변을 위한 함수 정의
def get_answer(user_input):
    result = multi_prompt_chain.invoke({"input": user_input})
    if isinstance(result, str) and result == "지원되지 않는 요청입니다. SQL 쿼리만 지원됩니다.":
        return result
    elif "SQL" in router_runnable.invoke({"input": user_input}):
        write_query_result = result
        write_query_result["query"] = clean_and_validate_query(write_query_result["query"])
        sql_result = execute_query.invoke({"query": write_query_result["query"]})
        final_answer_input = {
            "question": user_input,
            "query": write_query_result["query"],
            "result": sql_result
        }
        final_answer = answer.invoke(final_answer_input)
        return f"질문: {user_input}\nSQL 조회를 통한 답변입니다: {final_answer.strip()}"
    else:
        return f"질문: {user_input}\nRAG 조회를 통한 답변입니다: {result.strip()}"

# Streamlit 사용자 인터페이스
user_question = st.text_input("질문을 입력하세요:")
if st.button("질문하기"):
    if user_question:
        return_answer = get_answer(user_question)
        st.write("\n===========\n최종답변\n===========")
        st.write(return_answer)
