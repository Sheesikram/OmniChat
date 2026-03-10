from __future__ import annotations

import json
import logging
import os
import random
import re
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional, Union

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Third-party imports with graceful degradation
try:
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("langchain-openai not installed. LLM features disabled.")

try:
    from langgraph.graph import END, StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("langgraph not installed. Using fallback implementation.")

# ============================================================================
# ENUMS
# ============================================================================

class Intent(Enum):
    GREETING = auto()
    GOODBYE = auto()
    QUESTION = auto()
    COMMAND = auto()
    EMOTION = auto()
    MEMORY = auto()
    GENERAL = auto()

class Command(Enum):
    HELP = "help"
    HISTORY = "history"
    CLEAR = "clear"
    STATS = "stats"
    EXPORT = "export"
    PERSONA = "persona"
    THEME = "theme"
    QUIT = "quit"

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Message:
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    intent: Optional[Intent] = None

@dataclass
class ConversationStats:
    total_messages: int = 0
    intent_distribution: dict = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)

    def update(self, intent: Intent):
        self.total_messages += 1
        self.intent_distribution[intent.name] = self.intent_distribution.get(intent.name, 0) + 1

# ============================================================================
# STATE CLASS - Robust implementation
# ============================================================================

class State(dict):
    """State dictionary that works with both dot notation and dict access"""
    
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            return None
    
    def __setattr__(self, key, value):
        self[key] = value
    
    def copy(self):
        """Create a copy of the state"""
        return State(self.items())

# ============================================================================
# MEMORY
# ============================================================================

class ConversationMemory:
    def __init__(self, max_history: int = 50, persist_path: Optional[str] = None):
        self.history: deque[Message] = deque(maxlen=max_history)
        self.persist_path = persist_path or "conversation_history.json"
        
    def add(self, message: Message):
        self.history.append(message)
        
    def get_context(self, n: int = 5) -> list[Message]:
        return list(self.history)[-n:]
    
    def save(self):
        if not self.persist_path:
            return
        try:
            data = {
                "history": [
                    {
                        "role": m.role,
                        "content": m.content,
                        "timestamp": m.timestamp.isoformat(),
                        "intent": m.intent.name if m.intent else None
                    }
                    for m in self.history
                ]
            }
            with open(self.persist_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved conversation to {self.persist_path}")
        except Exception as e:
            logger.error(f"Failed to save: {e}")
    
    def load(self):
        if not self.persist_path or not Path(self.persist_path).exists():
            return
        try:
            with open(self.persist_path, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded conversation from {self.persist_path}")
        except Exception as e:
            logger.error(f"Failed to load: {e}")

# ============================================================================
# THEME
# ============================================================================

class Theme:
    THEMES = {
        "default": {
            "primary": "\033[94m", "secondary": "\033[92m", "accent": "\033[93m",
            "error": "\033[91m", "reset": "\033[0m", "bold": "\033[1m",
            "bot_prefix": "🤖", "user_prefix": "👤"
        },
        "minimal": {
            "primary": "", "secondary": "", "accent": "", "error": "",
            "reset": "", "bold": "", "bot_prefix": ">", "user_prefix": "<"
        }
    }
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.colors = self.THEMES.get(name, self.THEMES["default"])
    
    def format(self, text: str, color_key: str = "primary") -> str:
        return f"{self.colors.get(color_key, '')}{text}{self.colors.get('reset', '')}"

# ============================================================================
# PERSONA
# ============================================================================

class Persona:
    PERSONAS = {
        "default": {
            "name": "Assistant",
            "greeting": "Hello! I'm your AI assistant. How can I help you today?",
            "goodbye": "Goodbye! Have a great day!",
            "style": "helpful"
        },
        "pirate": {
            "name": "Captain Codebeard",
            "greeting": "Ahoy matey! Welcome aboard the good ship AI!",
            "goodbye": "Fair winds and following seas!",
            "style": "pirate"
        },
        "professor": {
            "name": "Dr. Wisdom",
            "greeting": "Greetings. I am prepared to elucidate any topic.",
            "goodbye": "Until our next scholarly discourse.",
            "style": "academic"
        }
    }
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.data = self.PERSONAS.get(name, self.PERSONAS["default"])
    
    def transform(self, text: str) -> str:
        if self.name == "pirate":
            return text.replace("hello", "ahoy").replace("yes", "aye") + " Arr!"
        return text

# ============================================================================
# INTENT CLASSIFIER
# ============================================================================

class IntentClassifier:
    def __init__(self):
        self.patterns = {
            Intent.GREETING: [r'\b(hi|hello|hey|howdy|hola)\b', r'good (morning|afternoon|evening)'],
            Intent.GOODBYE: [r'\b(bye|goodbye|see you|farewell)\b'],
            Intent.QUESTION: [r'\b(what|who|where|when|why|how)\b.*\?', r'^(what|who|where|when|why|how)\b'],
            Intent.COMMAND: [r'^/(help|history|clear|stats|export|persona|theme|quit)'],
            Intent.EMOTION: [r'\b(happy|sad|angry|excited|love|hate)\b'],
        }
        self.command_map = {cmd.value: cmd for cmd in Command}
    
    def classify(self, text: str) -> tuple[Intent, Optional[Command], dict]:
        text_lower = text.lower().strip()
        
        # Check for commands first
        if text_lower.startswith('/'):
            cmd_part = text_lower[1:].split()[0]
            if cmd_part in self.command_map:
                args = text_lower[1:].split()[1:]
                return Intent.COMMAND, self.command_map[cmd_part], {"args": args}
        
        # Check patterns
        for intent, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return intent, None, {}
        
        return Intent.GENERAL, None, {}

# ============================================================================
# RESPONSE GENERATOR
# ============================================================================

class ResponseGenerator:
    FALLBACKS = {
        Intent.GREETING: ["Hello there!", "Hi! How can I help?"],
        Intent.GOODBYE: ["Goodbye!", "See you later!"],
        Intent.GENERAL: ["That's interesting. Tell me more.", "I see. What else?"],
        Intent.QUESTION: ["That's a great question!", "I'd need to think about that."],
        Intent.EMOTION: ["I understand. Tell me more about how you feel."],
    }
    
    def generate(self, user_input: str, intent: Intent, persona: Persona, memory: ConversationMemory) -> str:
        # Handle specific intents with persona responses
        if intent == Intent.GREETING:
            return persona.data["greeting"]
        elif intent == Intent.GOODBYE:
            return persona.data["goodbye"]
        
        # Try LLM if available
        if LANGCHAIN_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            try:
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
                context = "\n".join([f"{m.role}: {m.content}" for m in memory.get_context(3)])
                prompt = f"You are {persona.data['name']}. {persona.data['style']}.\n\nContext:\n{context}\n\nUser: {user_input}\n\nAssistant:"
                response = llm.invoke(prompt)
                content = getattr(response, "content", None)
                if content:
                    return content.strip()
            except Exception as e:
                logger.warning(f"LLM failed: {e}")
        
        # Fallback
        responses = self.FALLBACKS.get(intent, self.FALLBACKS[Intent.GENERAL])
        return persona.transform(random.choice(responses))

# ============================================================================
# NODE FUNCTIONS
# ============================================================================

def create_nodes(memory, stats, persona, theme, classifier, generator):
    
    def preprocess(state: State) -> State:
        text = (state.get("user_input") or "").strip()
        text = " ".join(text.split())
        new_state = state.copy()
        new_state["clean_input"] = text
        new_state["user_input"] = text
        return new_state
    
    def classify_intent(state: State) -> State:
        text = state.get("clean_input") or state.get("user_input", "")
        intent, command, metadata = classifier.classify(text)
        new_state = state.copy()
        new_state["intent"] = intent
        new_state["command"] = command
        new_state["metadata"] = metadata
        logger.info(f"Classified as {intent.name}")
        return new_state
    
    def handle_command(state: State) -> State:
        command = state.get("command")
        new_state = state.copy()
        
        if command == Command.HELP:
            response = """Commands:
/help, /history, /clear, /stats, /export, /persona <name>, /theme <name>, /quit"""
        elif command == Command.HISTORY:
            recent = memory.get_context(5)
            response = "\n".join([f"{m.timestamp.strftime('%H:%M')} {m.role}: {m.content[:50]}" for m in recent]) or "No history."
        elif command == Command.CLEAR:
            os.system('cls' if os.name == 'nt' else 'clear')
            response = "Screen cleared."
        elif command == Command.STATS:
            duration = datetime.now() - stats.start_time
            response = f"Messages: {stats.total_messages}, Duration: {duration.seconds//60}m"
        elif command == Command.EXPORT:
            memory.save()
            response = "Conversation saved."
        elif command == Command.PERSONA:
            args = state.get("metadata", {}).get("args", [])
            if args and args[0] in Persona.PERSONAS:
                persona.name = args[0]
                persona.data = Persona.PERSONAS[args[0]]
                response = f"Switched to {persona.data['name']}"
            else:
                response = f"Current: {persona.data['name']}. Options: {', '.join(Persona.PERSONAS.keys())}"
        elif command == Command.THEME:
            args = state.get("metadata", {}).get("args", [])
            if args and args[0] in Theme.THEMES:
                theme.name = args[0]
                theme.colors = Theme.THEMES[args[0]]
                response = f"Switched to {args[0]} theme"
            else:
                response = f"Current: {theme.name}. Options: {', '.join(Theme.THEMES.keys())}"
        elif command == Command.QUIT:
            response = "/quit"
        else:
            response = "Unknown command"
        
        new_state["response"] = response
        new_state["is_command"] = True
        return new_state
    
    def generate_response(state: State) -> State:
        if state.get("is_command"):
            return state
        
        user_text = state.get("clean_input") or state.get("user_input", "")
        intent = state.get("intent", Intent.GENERAL)
        
        # Ensure intent is Intent enum
        if isinstance(intent, str):
            try:
                intent = Intent[intent]
            except KeyError:
                intent = Intent.GENERAL
        
        start_time = time.time()
        response_text = generator.generate(user_text, intent, persona, memory)
        elapsed = time.time() - start_time
        
        stats.update(intent)
        
        memory.add(Message(role="user", content=user_text, intent=intent))
        memory.add(Message(role="assistant", content=response_text, intent=intent))
        
        new_state = state.copy()
        new_state["response"] = response_text
        new_state["response_time"] = elapsed
        return new_state
    
    def postprocess(state: State) -> State:
        response = state.get("response", "No response")
        is_command = state.get("is_command", False)
        
        new_state = state.copy()
        if is_command:
            new_state["formatted_response"] = theme.format(response, "secondary")
        else:
            new_state["formatted_response"] = theme.format(response, "primary")
        return new_state
    
    return {
        "preprocess": preprocess,
        "classify_intent": classify_intent,
        "handle_command": handle_command,
        "generate_response": generate_response,
        "postprocess": postprocess
    }

# ============================================================================
# GRAPH BUILDER
# ============================================================================

def build_graph(memory, stats, persona, theme):
    classifier = IntentClassifier()
    generator = ResponseGenerator()
    nodes = create_nodes(memory, stats, persona, theme, classifier, generator)
    
    if LANGGRAPH_AVAILABLE:
        graph = StateGraph(State)
        
        for name, func in nodes.items():
            graph.add_node(name, func)
        
        graph.set_entry_point("preprocess")
        graph.add_edge("preprocess", "classify_intent")
        
        def route(state):
            cmd = state.get("command") if isinstance(state, dict) else getattr(state, "command", None)
            return "handle_command" if cmd else "generate_response"
        
        graph.add_conditional_edges("classify_intent", route, {
            "handle_command": "handle_command",
            "generate_response": "generate_response"
        })
        
        graph.add_edge("handle_command", "postprocess")
        graph.add_edge("generate_response", "postprocess")
        graph.add_edge("postprocess", END)
        
        compiled = graph.compile()
        
        class Wrapper:
            def __init__(self, compiled_graph, nodes):
                self.graph = compiled_graph
                self.nodes = nodes
            
            def invoke(self, initial_state: dict, config=None):
                # Convert to State if needed
                if not isinstance(initial_state, State):
                    initial_state = State(initial_state)
                
                try:
                    # Try LangGraph first
                    result = self.graph.invoke(initial_state, config)
                    if result is not None:
                        return result if isinstance(result, State) else State(result)
                    
                    # Fallback to manual execution if LangGraph returns None
                    logger.warning("LangGraph returned None, using manual execution")
                    return self._manual_execute(initial_state)
                    
                except Exception as e:
                    logger.error(f"LangGraph error: {e}, using manual execution")
                    return self._manual_execute(initial_state)
            
            def _manual_execute(self, state):
                """Execute nodes manually as fallback"""
                try:
                    s = self.nodes["preprocess"](state)
                    s = self.nodes["classify_intent"](s)
                    if s.get("command"):
                        s = self.nodes["handle_command"](s)
                    else:
                        s = self.nodes["generate_response"](s)
                    s = self.nodes["postprocess"](s)
                    return s
                except Exception as e:
                    logger.error(f"Manual execution failed: {e}")
                    error_state = State(state)
                    error_state["response"] = "Sorry, I encountered an error."
                    error_state["formatted_response"] = "Sorry, I encountered an error."
                    return error_state
        
        return Wrapper(compiled, nodes)
    
    else:
        # Simple fallback without LangGraph
        class SimpleGraph:
            def __init__(self, nodes):
                self.nodes = nodes
            
            def invoke(self, state, config=None):
                if not isinstance(state, State):
                    state = State(state)
                s = self.nodes["preprocess"](state)
                s = self.nodes["classify_intent"](s)
                if s.get("command"):
                    s = self.nodes["handle_command"](s)
                else:
                    s = self.nodes["generate_response"](s)
                s = self.nodes["postprocess"](s)
                return s
        
        return SimpleGraph(nodes)

# ============================================================================
# CHATBOT
# ============================================================================

class ChatBot:
    def __init__(self):
        self.memory = ConversationMemory(max_history=100)
        self.stats = ConversationStats()
        self.persona = Persona("default")
        self.theme = Theme("default")
        self.app = build_graph(self.memory, self.stats, self.persona, self.theme)
        self.memory.load()
        self._print_welcome()
    
    def _print_welcome(self):
        print(f"\n{self.theme.colors['bold']}{self.theme.colors['accent']}")
        print("=" * 60)
        print("       ADVANCED LANGGRAPH CHATBOT")
        print("=" * 60)
        print(f"{self.theme.colors['reset']}")
        print(f"Type /help for commands, /quit to exit")
        print(f"{self.theme.format(self.persona.data['greeting'], 'secondary')}\n")
    
    def run(self):
        try:
            while True:
                try:
                    user_input = input(f"{self.theme.colors['user_prefix']} > ").strip()
                    
                    if not user_input:
                        continue
                    
                    if user_input.lower() in ['/quit', 'exit', 'quit']:
                        print(f"\n{self.theme.format(self.persona.data['goodbye'], 'secondary')}")
                        break
                    
                    # Echo user input
                    print(f"{self.theme.colors['user_prefix']} {user_input}")
                    
                    # Process
                    start = time.time()
                    result = self.app.invoke({"user_input": user_input})
                    elapsed = time.time() - start
                    
                    if result is None:
                        print(f"{self.theme.format('Error: No response', 'error')}")
                        continue
                    
                    response = result.get("formatted_response") or result.get("response", "No response")
                    print(f"{self.theme.colors['bot_prefix']} {response}\n")
                    
                    if elapsed > 1:
                        print(f"{self.theme.format(f'[{elapsed:.1f}s]', 'accent')}")
                    
                    # Check for quit command
                    if result.get("response") == "/quit":
                        break
                        
                except KeyboardInterrupt:
                    print(f"\n{self.theme.format('Goodbye!', 'secondary')}")
                    break
                except Exception as e:
                    logger.error(f"Error: {e}", exc_info=True)
                    print(f"{self.theme.format(f'Error: {e}', 'error')}")
        
        finally:
            self.memory.save()

# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", choices=list(Persona.PERSONAS.keys()), default="default")
    parser.add_argument("--theme", choices=list(Theme.THEMES.keys()), default="default")
    parser.add_argument("--no-persist", action="store_true")
    parser.add_argument("--api-key", help="OpenAI API key")
    args = parser.parse_args()
    
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    
    bot = ChatBot()
    
    if args.persona != "default":
        bot.persona = Persona(args.persona)
        bot.persona.data = Persona.PERSONAS[args.persona]
    
    if args.theme != "default":
        bot.theme = Theme(args.theme)
        bot.theme.colors = Theme.THEMES[args.theme]
    
    if args.no_persist:
        bot.memory.persist_path = None
    
    bot.run()

if __name__ == "__main__":
    main()