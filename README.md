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
