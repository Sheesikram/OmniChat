# 🤖 Advanced LangGraph Chatbot

A production-ready, feature-rich conversational AI built with Python, LangGraph, and OpenAI. Features multi-intent classification, persistent memory, customizable personas, themes, and a robust command system.

## ✨ Features

- **Multi-Intent Classification**: Automatically detects greetings, questions, commands, emotions, and more
- **Persistent Memory**: Saves conversation history to JSON with context awareness
- **Multiple Personas**: Switch between Default, Pirate, Professor, and more personalities
- **Customizable Themes**: Default, Dark, and Minimal color schemes
- **Command System**: Built-in commands for history, stats, themes, personas
- **LLM Integration**: OpenAI GPT-4o-mini support with graceful fallback
- **Robust Error Handling**: Works with or without dependencies, never crashes
- **Async Support**: Ready for API integration

## 🚀 Quick Start

### Installation

```bash
# Clone or download the script
git clone &lt;your-repo-url&gt;
cd chatbot

# Install dependencies (optional but recommended)
pip install langgraph langchain-openai

# Set OpenAI API key (optional - bot works without it)
export OPENAI_API_KEY="your-api-key-here"
# On Windows: set OPENAI_API_KEY=your-api-key-here


# Basic usage
python chatbot.py

# With specific persona and theme
python chatbot.py --persona pirate --theme dark

# Without persistence
python chatbot.py --no-persist

# With API key via command line
python chatbot.py --api-key your-key-here

🤖 > hi
Hello! I'm your AI assistant. How can I help you today?

👤 > what is the weather like?
🤖 That's a great question! Let me think...

👤 > I feel happy today
🤖 Wonderful to hear!

| Command           | Description                      | Example           |
| ----------------- | -------------------------------- | ----------------- |
| `/help`           | Show all available commands      | `/help`           |
| `/history`        | Show recent conversation history | `/history`        |
| `/clear`          | Clear the terminal screen        | `/clear`          |
| `/stats`          | Show conversation statistics     | `/stats`          |
| `/export`         | Save conversation to JSON file   | `/export`         |
| `/persona <name>` | Switch persona                   | `/persona pirate` |
| `/theme <name>`   | Switch color theme               | `/theme dark`     |
| `/quit`           | Exit the application             | `/quit`           |


User Input → Preprocess → Classify Intent → [Command|Generate Response] → Postprocess → Output
                ↓              ↓                    ↓
           Clean text    Detect intent      Update memory

           # Test without OpenAI (fallback mode)
unset OPENAI_API_KEY
python chatbot.py

# Test with specific scenarios
python chatbot.py --persona professor
> /help
> what is quantum physics?
> /persona pirate
> tell me a joke
> /stats
> /export
> /quit


📦 Dependencies
Required
Python 3.8+
Optional (for full features)
plain
Copy
langgraph>=0.0.40
langchain-openai>=0.0.5
Graceful Degradation
The bot works without any dependencies using built-in fallback implementations.
