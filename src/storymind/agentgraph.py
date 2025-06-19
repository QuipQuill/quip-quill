# Importing chat models
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

# from langgraph.prebuilt import create_react_agent
# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Notice we're using an in-memory checkpointer. This is convenient for our tutorial (it saves it all in-memory). 
# In a production application, you would likely change this to use SqliteSaver or PostgresSaver and connect to your own DB.
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda
from langchain.output_parsers.json import SimpleJsonOutputParser
from typing_extensions import TypedDict
from typing import Annotated
import logging
from dotenv import load_dotenv
import os
import random

from langchain_core.messages.utils import (
    trim_messages,
    count_tokens_approximately
)

# -----------------------------------------------------------------------------------------------------------

# Tools
from src.storymind.gm_tools import get_toolpack

# Импортируем менеджер игры
from src.storymind.manager import Manager as gm
from src.storymind.prompts import warning_templates, validator_system_prompt, updator_system_prompt

# INIT CONFIG .yaml
from scripts.utils import load_config, _strip_json_string


# -----------------------------------------------------------------------------------------------------------
# Создаем папку для логов, если она не существует
if not os.path.exists('logs'):
    os.makedirs('logs')

# -----------------------------------------------------------------------------------------------------------
def get_master_agent(system_prompt: str ,tools, provider: str = "google_genai") -> PromptTemplate:
    """Функция для получения главного агента с инструментами."""
    match provider:
        case "google_genai":
            # Инициализируйте модель Gemini 2.0 Flash через Langchain
            llm = init_chat_model("gemini-2.5-flash-preview-05-20", model_provider="google_genai", configurable_fields={
                "temperature": 0.8,  # Настройка температуры для генерации ответов
                "top_p": 0.9,  # Настройка top_p для контроля разнообразия ответов
                "max_tokens": 16384,  # Увеличиваем лимит токенов для больших ответов
                "timeout": 600,  # Увеличиваем таймаут для долгих запросов
            })
            # llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
            # llm = init_chat_model("gemini-2.0-flash-lite", model_provider="google_genai")

        case "mistral":
            # Инициализируйте модель Mistral через Langchain
            llm = init_chat_model("mistral-large-latest", model_provider="mistralai")
        case _:
            # Инициализируйте модель OpenAI через Langchain, но с другими параметрами baseurl
            llm = ChatOpenAI(
                model=os.getenv("RAG_LLM_MODEL"),
                base_url=os.getenv('RAG_BASE_URL'),
                temperature=0.8, 
                top_p=0.9,
                max_tokens=16384,  # Увеличиваем лимит токенов для больших ответов
                timeout=60,
                openai_api_key=os.getenv('OPENAI_API_KEY')
            )
    llm_with_tools = llm.bind_tools(tools)
    master_agent = PromptTemplate.from_template(system_prompt) | llm_with_tools 
    return master_agent, llm

def create_graph(app_config, llm_provider = 'google_genai', max_tokens_trim = 2048, lite_mode = True) -> StateGraph:
    """
    Function to create a state graph for the text-based RPG game using LangGraph.
    Args: 
        app_config (dict): Configuration dictionary loaded from config.yaml.
        llm_provider (str): Provider for the LLM, default is 'google_genai'.
        max_tokens_trim (int): Maximum number of tokens to keep in the messages, default is 2048.
        lite_mode (bool): If True, uses a lighter system prompt for the game master agent. Default is True.
    """

    # Каждый node может получать текущее State в качестве входных данных и выводить обновлённое состояние.
    # Обновления для messages будут добавляться к существующему списку, а не перезаписывать его, благодаря 
    # встроенной функции add_messages с синтаксисом Annotated

    system_prompt = app_config.get("prompts", {}).get("system_lite" if lite_mode else "system", None)
    if not system_prompt:
        logging.critical("System prompt not found in config.yaml. Please check your configuration file. Will use default system prompt instead.")
        system_prompt = "You are a Game Master (GM) for a text-based role-playing game. You are forbidden to deviate from the state of the game and you are obliged to navigate only on the information available to you."

    graph_config = app_config.get('agentgraph',{})

    # Инициализация LLM с инструментами
    tools = get_toolpack()  
    llm_agent, model = get_master_agent(system_prompt ,tools=tools, provider= graph_config['provider'] if 'provider' in graph_config else llm_provider) 

    validator_agent = PromptTemplate.from_template(validator_system_prompt) | model
    updator_agent = PromptTemplate.from_template(updator_system_prompt) | model | RunnableLambda(_strip_json_string) | SimpleJsonOutputParser()


    # Общее сосотояние для графа
    class State(TypedDict):
        messages: Annotated[list, add_messages]
        # current_player_state: BaseMessage
    
    def _trim_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
        """Функция для обрезки сообщений в состоянии."""
        return trim_messages(
            messages,
            strategy="last",
            token_counter=count_tokens_approximately,
            max_tokens=max_tokens_trim,  # Устанавливаем лимит на количество токенов
            start_on=("human", "system"),  # Начинаем с human и system сообщений
            end_on=("human","system"),
            include_system = False
        )
    
    def _get_agent_state_message(config: RunnableConfig) -> str:
        """Функция для получения состояния агента в виде сообщения."""
        return {"role": "system","content": config['configurable']['gm'].get_agent_state()}

    def trimmer(state: State, config: RunnableConfig) -> Command[State]:
        """Функция для обрезки сообщений в состоянии."""
        # Оптимизируем, чтобы не захламлять историю чата лишними сообщениями
        # Добавляем сообщение с состоянием агента (БУДЕТ ПЕРЕНЕСЕН В ОТДЕЛЬНЫ УЗЕЛ В ДАЛЬНЕЙШЕМ)
        return {
            'messages': _trim_messages(state["messages"]),
            # "current_player_state": _get_agent_state_message(config)
            }
        

    def validator(state: State, config: RunnableConfig) -> Command[State]:
        """Функция для валидации сообщений от игрока."""
        config['configurable']['gm'].state["current_player_state"] = _get_agent_state_message(config)

        if not config['configurable'].get('generated_mode', False):
            # Если режим генерации не включен, добавляем сообщение с текущим состоянием агента
            return 'passed'
    
        if type(state["messages"][-1]).__name__ == "SystemMessage":
            logging.debug("Всегда пропускаем запрос от системы! " + state["messages"][-1].content)
            return 'passed'

        # Invoke + rag current state
        result = validator_agent.invoke(state["messages"] + [config['configurable']['gm'].state["current_player_state"]]).content.strip().lower()
        if 'passed' in result:
            return 'passed'
        if 'failed' in result:
            return 'failed'
        if 'edit' in result:
            return 'edit'
    
    def updator(state: State, config: RunnableConfig) -> Command[State]:
        """Функция для обновления состояния игры."""
        Gm: gm = config['configurable']['gm']
        # messages = state["messages"][:-5] if len(state['messages']) >= 5 else state["messages"]
        entities = updator_agent.invoke(state['messages'] + [config['configurable']['gm'].state["current_player_state"]])
        logging.debug(f"[Agent:Updator]: Extracted JSON: \n {entities}")
        return {'messages':[ 
                {"role": "system", "content": "[Agent:Updator]:\n" + \
                    "\n".join([
                        Gm.add_entity(e['name'].strip().lower(), e['description'].strip(), e.get('type', "ADDED_ITEM"), rel_type=e.get("relationship_type"))
                        for e in entities if e and e.get("name") and e.get("description")
                    ])
                }
            ]}  # Фильтруем пустые строки

    def warning_bot(state: State, config: RunnableConfig) -> Command[State]:
        """Функция для предупреждения игрока о нарушении правил игры."""
        # Здесь можно добавить логику предупреждения игрока
        # Например, если игрок пытается использовать запрещённые действия
        return {'messages': [{'role': 'assistant', 'content': random.choice(warning_templates)}]}

    # Создаем главного агента
    def chatbot(state: State, config: RunnableConfig) -> Command[State]:
        """Функция для обработки сообщений от игрока."""
        return {"messages": [llm_agent.invoke(state["messages"] + [config['configurable']['gm'].state["current_player_state"]])]}
    


    # Создание графа состояний
    graph_builder = StateGraph(State)
        
    # Память 
    memory = InMemorySaver()

    # Add tools
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)
    # Make graph
    graph_builder.add_node("trimmer", trimmer)
    graph_builder.add_node("warning_bot", warning_bot)


    graph_builder.add_node("updator", updator)
    graph_builder.add_node("chatbot", chatbot)

    # Any time a tool is called, we return to the chatbot to decide the next step

    # Other edges
    graph_builder.add_conditional_edges("trimmer", validator , {
        'passed': "chatbot",  # Если валидация прошла, переходим к чат-боту
        'edit': "updator",
        'failed': "warning_bot"  # Если валидация не прошла, возвращаемся к чат-боту
    })

    graph_builder.add_edge("updator", "chatbot")
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    graph_builder.add_edge("tools", "chatbot")

    graph_builder.set_entry_point("trimmer")

    # Хранимим граф в памяти в оперативной памяти
    return graph_builder.compile(checkpointer=memory)


def stream_graph_updates(graph, config, user_input: str, role: str = "user"):
    for event in graph.stream(
        {"messages": [{"role": role, "content": user_input}]},
        config=config,
        stream_mode='values'):
        
        event["messages"][-1].pretty_print()
        
        # for value in event.values():
        #     print("Assistant:", value["messages"][-1].content)

# -----------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Инициализация конфигурации
    app_config = load_config("./config.yaml")
    # -----------------------------------------------------------------------------------------------------------
    # Загрузка переменных окружения из .env файла
    load_dotenv()


    # -----------------------------------------------------------------------------------------------------------
    # Импортируем менеджер игры
    game_manager = gm(
        load=False
    )

    game_manager.restart()  # Перезапускаем менеджер игры

    agent_name = "player" 
    start_location = "entrance rune hall"

    # Инициализация агента в динамическом графе
    game_manager.initalize_agent(agent_name, start_location=start_location) 

    # Настройка конфигурации для запуска
    run_config = {"configurable": {"thread_id": "1"}, 'gm': game_manager, "agent_name": agent_name, "generated_mode": True}

    # Создание графа взаимодействия с игроком
    graph = create_graph(
        app_config=app_config, 
    )

    # Start chat
    print("Welcome to the text-based RPG! Type 'quit', 'q' or 'exit' to end the game.")

    # Start message

    init_message = {'role': 'system', 'content': 'You need to welcome the player!'}

    stream_graph_updates(graph, run_config, init_message['content'], role=init_message['role'])

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(graph, run_config, user_input)
        game_manager.update()



    snapshot = graph.get_state(run_config)
    print("Snapshot:", snapshot)

    # СОХРАНЯЕМ СООБЩЕНИЯ
    from datetime import date

    current_date = date.today()
    # Save messages to log
    with open('./logs/MESSAGES.log', 'w', encoding="utf-8") as f:
        for m in snapshot.values["messages"]:
            f.write(f"{current_date} - {type(m).__name__} - {m.content}\n")