from typing import List, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

from app.config import LLM_PROVIDER, GROQ_API_KEY, GEMINI_API_KEY

REFUSAL_TEXT = "Je ne dispose pas de cette information dans le corpus."


class GenerationService:
    """
    Service responsible for generating an answer based on a query and a retrieved context.
    """
    def __init__(self, llm=None):
        # Dependency Injection for testability
        self.llm = llm or self._initialize_llm()
        self.chain = self._build_chain()

    def _initialize_llm(self):
        if LLM_PROVIDER == "groq":
            return ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant", temperature=0.1)
        elif LLM_PROVIDER == "gemini":
            return ChatGoogleGenerativeAI(google_api_key=GEMINI_API_KEY, model="gemini-2.0-flash")
        else:
            raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")

    def _build_chain(self):
        # Prompt strict pour un assistant RAG anti-hallucination
        system_prompt = (
            "Tu es AssistKB, l'assistant interne de l'entreprise. "
            "Tu reponds en francais, uniquement a partir du contexte fourni.\n"
            "Regles strictes :\n"
            '1. Si le contexte ne contient pas la reponse, dis exactement : "' + REFUSAL_TEXT + '"\n'
            "2. Cite tes sources entre crochets, ex. [source 1], selon la numerotation du contexte.\n"
            "3. N'utilise aucune connaissance externe. Reste concis et factuel.\n\n"
            "Contexte :\n{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
        ])

        # Prompt -> LLM (pas de StrOutputParser : on garde l'AIMessage
        # pour lire l'usage tokens, requis par les metriques d'exploitation)
        return prompt | self.llm

    def generate(self, query: str, context_docs: List[Document]) -> Tuple[str, dict]:
        """
        Generates a cited response based on the retrieved documents.
        Returns (answer, token usage).
        """
        if not context_docs:
            return REFUSAL_TEXT, {"input_tokens": 0, "output_tokens": 0}

        # Format documents into a single context string with numbering for citations
        context_text = "\n\n".join(
            f"[source {i + 1}] {doc.metadata.get('source', '?')}\n{doc.page_content}"
            for i, doc in enumerate(context_docs)
        )

        response = self.chain.invoke({"context": context_text, "question": query})
        usage = response.usage_metadata or {}
        return response.content, {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        }
