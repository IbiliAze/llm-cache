from uuid import uuid4

from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

# from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from app.agents.researcher.agent import ResearcherAgent
from app.agents.researcher.nodes import ResearcherNodes
from app.agents.researcher.routes import ResearcherRoutes
from app.agents.researcher.state import ResearcherState
from app.agents.researcher.tools import ResearcherTools
from app.common.mcp.mcp_client import McpClient
from app.common.rag.rag import Rag
from app.security.language_detector import LanguageDetector
from core.config.settings import get_settings
from core.models.models import Models

DIM_GREEN = '\033[2;32m'
BRIGHT_GREEN = '\033[92m'
GRAY = '\033[90m'
RESET = '\033[0m'
DIM_RED = '\033[2;31m'
BRIGHT_RED = '\033[91m'

settings = get_settings()
models_to_use = Models()
embedding_llm = models_to_use.embedding_llm
vectorstore = OpenSearchVectorSearch(
  opensearch_url=settings.opensearch_url,
  index_name=settings.opensearch_documents_index,
  embedding_function=embedding_llm,
  http_auth=(
    (settings.opensearch_user, settings.opensearch_password)
    if settings.opensearch_user
    else None
  ),
)
rag = Rag(vectorstore)

if rag.get_document_count() == 0:
  rag.add_texts(
    [
      'TradeOps is our internal platform for managing trade lifecycle '
      'operations, including trade capture, settlement, and reconciliation.',
      'Eight Mile Services is a vendor providing outsourced back-office '
      'support for trade settlement and client onboarding.',
    ],
    source='seed',
  )


db_path = 'checkpoints.db'

with SqliteSaver.from_conn_string(db_path) as saver:
  print(f'DB was created at {db_path}')
  models = Models()
  mcp_client = McpClient(name='eightmile')
  tools = ResearcherTools(rag=rag, mcp_client=mcp_client)
  tool_list = tools.load_tools()
  nodes = ResearcherNodes(models=models, tools=tool_list)
  routes = ResearcherRoutes()
  agent = ResearcherAgent(nodes=nodes, routes=routes, saver=saver)
  agent.get_graph_png()
  config: RunnableConfig = {'configurable': {'thread_id': str(uuid4())}}

  agent.get_graph_png()

  language_detector = LanguageDetector()

  def ask(question: str) -> None:
    """Send one turn and print each message as the graph produces it."""

    language_detector_result = language_detector.check(question)
    if not language_detector_result.allowed:
      print(f'{BRIGHT_RED}>>> {language_detector_result.reason}')
      return

    print(f'\n{DIM_GREEN}>>> {question}{RESET}')
    state: ResearcherState = {
      'messages': [HumanMessage(question)],
      'error': '',
      'retry_count': 0,
      'context_summary': '',
      'current_agent': None,
      'handoff_reason': '',
    }
    for response in agent.stream_messages(state, config=config):
      print(
        f'{BRIGHT_GREEN}>>> [ {response.current_agent.upper()} ] {response.message.content}{RESET}\n'
        f'  {GRAY}• model: {response.model_used}{RESET}\n',
        f'  {GRAY}• hand-off reason: {response.handoff_reason}{RESET}\n',
        flush=True,
      )

  ask('Hi, my name is ibi')
  ask('I need help with billing')
  ask('What is the capital of Azerbaijan?')
  ask('What do you know about tradeops or blue svs LTD?')
  ask('What is my name?')
  ask('what services do you provide?')
  ask('salam, necesen?')

  # print(agent.get_current_state(config))
  # print()

  # for i, snapshot in enumerate(agent.get_state_history(config)):
  #   print(f' {i}: {snapshot.values["messages"]}')
