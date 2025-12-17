# 🧩 Ideal AI - Universal LLM Connector

> **One Connector to Rule Them All**

A production-ready, Open Source **Python LLM Connector** providing a unified interface for **Text, Vision, Audio, Image & Video** across 15+ providers (DeepSeek, Ollama, OpenAI, Alibaba, etc.).

Features **dynamic model injection** (add new providers at runtime without code changes) and native support for **Smolagents** & **LangChain** workflows.

[![PyPI version](https://img.shields.io/pypi/v/ideal-ai.svg)](https://pypi.org/project/ideal-ai/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Devgoodcode/ideal-ai/blob/main/examples/demo_ideal_universal_connector.ipynb)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Idealcom/ideal-ai-llm-connector-demo)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## ✨ Features

- **🔗 Universal LLM Connector** - One unified interface for 15+ providers (OpenAI, Ollama, DeepSeek, Google, etc.).
- **🎯 Multi-Modal Powerhouse** - Text, Vision, Audio (STT), Image Gen, Video Gen (Wan 2.1), Speech (TTS).
- **💉 Dynamic Model Injection** - Register new models or providers at runtime without changing source code.
- **🤖 Agent & Workflow Ready** - Native wrapper for **Smolagents** and fully compatible with **LangChain** / **LangGraph**.
- **🎙️ Native Voice Chat** - Ready-to-use pipeline for full audio-to-audio interaction.
- **🛡️ Production-Grade** - Robust error handling, async polling (for Video/Audio), and binary management.
- **💼 100% Open Source** - Apache 2.0 License, free for commercial use.
- **📦 PIP-Installable** - `pip install ideal-ai`

## 📺 See it in action

[![Watch the Demo](https://img.youtube.com/vi/f1DwFRpo2HA/0.jpg)](https://www.youtube.com/watch?v=f1DwFRpo2HA)

> *One Connector to Rule Them All. Watch the full demo (2.50 min).*

## 🚀 Quick Start

### Installation

```bash
pip install ideal-ai
```

### Basic Usage

```python
from ideal_ai import IdealUniversalLLMConnector
import os

# Initialize with your API keys
connector = IdealUniversalLLMConnector(
    api_keys={
        "openai": os.getenv("OPENAI_API_KEY"),
        "google": os.getenv("GOOGLE_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    }
)

# Text generation
response = connector.invoke(
    provider="openai",
    model_id="gpt-4o",
    messages=[{"role": "user", "content": "Explain quantum computing simply."}]
)
print(response["text"])

# Vision (multimodal)
with open("image.jpg", "rb") as f:
    image_bytes = f.read()

analysis = connector.invoke_image(
    provider="google",
    model_id="gemini-2.5-flash",
    image_input=image_bytes,
    prompt="What's in this image?"
)
print(analysis["text"])

# Image generation
result = connector.invoke_image_generation(
    provider="openai",
    model_id="dall-e-3",
    prompt="A futuristic robot in a cyberpunk city"
)
# result["images"] contains base64 or URLs
```

## 🎯 Pre-Configured Providers (Out-of-the-Box)

The following providers are pre-registered in `config.json` for immediate use.
**Note:** You can easily **inject any other model** or provider (OpenAI-compatible, Ollama, etc.) at runtime without changing the package code.

| Provider | Text | Vision | Audio | Speech | Image Gen | Video Gen |
|----------|:----:|:------:|:-----:|:------:|:---------:|:---------:|
| **OpenAI** | ✅ | ✅ | - | ✅ | ✅ | - |
| **Google (Gemini)** | ✅ | ✅ | - | - | - | - |
| **Anthropic (Claude)** | ✅ | ✅ | - | - | - | - |
| **Ollama (Local)** | ✅ | ✅ | - | - | - | - |
| **Alibaba (Qwen)** | ✅ | - | - | - | - | ✅ |
| **Infomaniak** | ✅ | - | ✅ | - | ✅ | - |
| **DeepSeek** | ✅ | - | - | - | - | - |
| **Moonshot AI** | ✅ | ✅ | - | - | - | - |
| **Perplexity** | ✅ | - | - | - | - | - |
| **Hugging Face** | ✅ | - | - | - | - | - |
| **MiniMax** | ✅ | - | - | - | - | - |

## 📚 Advanced Usage

### Adding Custom Models at Runtime

The power of Ideal AI is its extensibility. Add any model without modifying source code:

```python
# Define your custom model configuration
custom_model = {
    "myprovider:custom-model": {
        "api_key_name": "myprovider",
        "families": {
            "text": "openai_compatible"  # Reuse existing recipe
        },
        "url_template": "https://api.myprovider.com/v1/chat/completions"
    }
}

# Initialize connector with custom model
connector = IdealUniversalLLMConnector(
    api_keys={"myprovider": "your-api-key"},
    custom_models=custom_model
)

# Use it immediately
response = connector.invoke("myprovider", "custom-model", messages)
```

### Dynamic Model Injection

```python
# Add model after initialization
connector.register_model(
    "provider:new-model",
    {
        "families": {"text": "openai_compatible"},
        "url_template": "https://api.example.com/chat"
    }
)
```

### Audio Transcription

```python
# Transcribe audio with Infomaniak Whisper
transcription = connector.invoke_audio(
    provider="infomaniak",
    model_id="whisper",
    audio_file_path="recording.m4a",
    language="en"
)
print(transcription["text"])
```

### Speech Synthesis (TTS)

```python
# Generate speech from text
audio_result = connector.invoke_speech_generation(
    provider="openai",
    model_id="tts-1",
    text="Hello, this is a test.",
    voice="nova"
)

# Save audio file
with open("output.mp3", "wb") as f:
    f.write(audio_result["audio_bytes"])
```

### Video Generation

```python
# Generate video with Alibaba Wan (async polling handled automatically)
video_result = connector.invoke_video_generation(
    provider="alibaba",
    model_id="wan2.1-t2v-turbo",
    prompt="A robot walking in a futuristic city",
    size="1280*720"
)
print(f"Video URL: {video_result['videos'][0]}")
```

## 🤖 Smolagents Integration

Perfect for building AI agents:

```python
from ideal_ai import IdealUniversalLLMConnector, IdealSmolagentsWrapper
from smolagents import CodeAgent

connector = IdealUniversalLLMConnector(api_keys={...})

# Wrap for smolagents
model = IdealSmolagentsWrapper(
    connector=connector,
    provider="openai",
    model_id="gpt-4o"
)

# Use with any smolagents agent
agent = CodeAgent(tools=[...], model=model)
agent.run("Build a web scraper for news articles")
```

## 🦜🔗 LangChain & LangGraph Ready

Ideal AI fits perfectly into **LangGraph** nodes or **LangChain** workflows. No complex wrappers needed—just call it directly inside your nodes.

```python
from ideal_ai import IdealUniversalLLMConnector
from langgraph.graph import StateGraph

connector = IdealUniversalLLMConnector(api_keys={...})

# Use directly in a LangGraph node
def chatbot_node(state):
    response = connector.invoke(
        provider="deepseek",       # Switch provider instantly!
        model_id="deepseek-chat",
        messages=state["messages"]
    )
    return {"messages": [response["text"]]}

# Build your graph...
workflow = StateGraph(dict)
workflow.add_node("chatbot", chatbot_node)
```

---

## 🏗️ Clean Architecture & Enterprise Patterns

`ideal-ai` is built to be modular. For production applications, you can easily wrap it in a **Service Layer** to centralize your AI logic.

This approach gives you absolute control to **inject custom parsers**, **switch providers dynamically** (e.g., using Ollama for local development and OpenAI for production), and keep your business logic clean.

### The Pattern: `AIService` Wrapper

```python
# src/services/ai_service.py
from ideal_ai import IdealUniversalLLMConnector
import os


class AIService:
    """
    Centralized Service for AI interactions.
    Use this layer to manage environment-specific logic (Dev vs Prod).
    """
    def __init__(self):
        # Initialize the engine once
        self._engine = IdealUniversalLLMConnector(
            api_keys={
                "openai": os.getenv("OPENAI_API_KEY")
            }
        )

    def chat_with_user(self, user_message: str) -> str:
        """
        Your app's simplified contract.
        Centralizes the decision of which model/provider to use.
        """
        # Logic: Use free local model for Dev, powerful model for Prod
        is_prod = os.getenv("ENV") == "production"
        provider = "openai" if is_prod else "ollama"
        model = "gpt-4o" if is_prod else "llama3.2"

        response = self._engine.invoke(
            provider=provider,
            model_id=model,
            messages=[{"role": "user", "content": user_message}]
        )
        return response["text"]
```

### Benefits of This Pattern

- ✅ **Separation of Concerns** - Business logic stays clean
- ✅ **Environment-Aware** - Dev uses local models, Prod uses powerful APIs
- ✅ **Provider Abstraction** - Swap providers without touching your code
- ✅ **Testable** - Mock the service layer easily
- ✅ **Maintainable** - All AI logic in one place

---

## 🔧 Configuration System

Ideal AI uses a two-level configuration system:

1. **Families (Recipes)** - Define how to interact with API types
2. **Models (Cards)** - Define which family each model uses for each modality

All default configurations are stored in `config.json` and can be extended without touching Python code.

### Custom Parser Example

If a provider's response format is non-standard:

```python
# Define custom parser
def my_parser(raw_response):
    return raw_response["data"]["content"]["text"]

# Inject it
connector = IdealUniversalLLMConnector(
    parsers={"provider:model": my_parser}
)
```

## 🐛 Debugging

Enable debug mode to inspect payloads and responses:

```python
response = connector.invoke(
    provider="openai",
    model_id="gpt-4o",
    messages=[...],
    debug=True  # Shows raw API calls and responses
)
```

## 📦 Installation from Source

```bash
# Clone repository
git clone https://github.com/Devgoodcode/ideal-ai.git
cd ideal-ai

# Install in development mode
pip install -e .

# Or build and install
pip install build
python -m build
pip install dist/ideal_ai-0.1.0-py3-none-any.whl
```

## 🧪 Running Examples

Check the `examples/` folder for comprehensive demos:

```bash
# Open demo notebook
jupyter notebook examples/demo_ideal_universal_connector.ipynb
```

The demo notebook covers the following capabilities in order:
- 1️⃣ **Text Generation Loop**: Unified iteration over OpenAI, Google, DeepSeek, Alibaba, etc.
- 2️⃣ **Vision/Multimodal**: Image analysis with Gemini, GPT-4o or Claude.
- 3️⃣ **Image Generation**: Creation with DALL-E 3 or Flux.
- 4️⃣ **Audio Transcription**: Speech-to-Text (STT) with Whisper.
- 5️⃣ **Speech Synthesis**: Text-to-Speech (TTS) with OpenAI.
- 6️⃣ **Video Generation**: Async video creation with Alibaba Wan.
- 7️⃣ **Runtime Injection**: How to add custom models/providers on the fly.
- 8️⃣ **Conversational Memory**: Handling multi-turn chat history.
- 9️⃣ **AI Agents**: Integration with Hugging Face's smolagents.
- 🧪 **Interactive UI (Bonus)**: A full widget-based dashboard to test all modalities (including Voice Chat) without code.

## 🔑 Environment Variables

Create a `.env` file or set environment variables:

```bash
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AI...
ANTHROPIC_API_KEY=sk-ant-...
ALIBABA_API_KEY=sk-...
INFOMANIAK_AI_TOKEN=...
INFOMANIAK_PRODUCT_ID=...
OLLAMA_URL=http://localhost:11434
```

## 🚀 Built-in Models (Extensible to Any Provider)

These models are pre-registered in `config.json` for immediate use.
**Remember:** You are not limited to this list! You can inject **any** new model or provider at runtime.

### Text Generation
- OpenAI: `gpt-4o`, `gpt-3.5-turbo`, `gpt-5`
- Google: `gemini-2.5-flash`
- DeepSeek: `deepseek-chat` (V3), `deepseek-reasoner` (R1)
- Infomaniak: `apertus-70b` (Souverain), `mixtral`
- Anthropic: `claude-haiku-4-5`
- Alibaba: `qwen-turbo`, `qwen-plus`, `qwen3-max`
- Ollama: `llama3.2`, `qwen2:7b`, `deepseek-r1:8b`

### Vision/Multimodal
- OpenAI: `gpt-4o`
- Google: `gemini-2.5-flash`
- Anthropic: `claude-haiku-4-5`
- Ollama: `llava`, `qwen3-vl:30b`

### Audio Transcription
- Infomaniak: `whisper`

### Speech Synthesis
- OpenAI: `tts-1`, `tts-1-hd`

### Image Generation
- OpenAI: `dall-e-3`
- Infomaniak: `flux-schnell`, `sdxl-lightning`

### Video Generation
- Alibaba: `wan2.1-t2v-turbo`, `wan2.2-t2v-plus`, `wan2.5-t2v-preview`

## 📖 Documentation

For detailed API documentation, see:

- [GitHub Repository](https://github.com/Devgoodcode/ideal-ai)
- [Connector API](https://github.com/Devgoodcode/ideal-ai/blob/main/src/ideal_ai/connector.py) - Full method signatures with docstrings
- [Configuration Schema](https://github.com/Devgoodcode/ideal-ai/blob/main/src/ideal_ai/config.json) - Available families and models
- [Examples](https://github.com/Devgoodcode/ideal-ai/tree/main/examples) - Working code samples

## 🤝 Contributing

Contributions welcome! 

To add a new provider:

1. Add family configuration to `config.json` (or pass as `custom_families`)
2. Add model configurations using that family
3. Test with the demo notebook

No Python code changes needed for most additions!

## 📝 License

Apache License 2.0 - See [LICENSE](LICENSE) file for details.

## 👤 Author & Support

**Gilles Blanchet**
- 🛠️ Created by: [IA-Agence.ai](https://ia-agence.ai/ideal-ai-universal-llm-connector/) - *Enterprise AI Architecture & Custom Integration.*
- 🌐 Agency: [Idealcom.ch](https://idealcom.ch)
- 🐙 GitHub: [@Devgoodcode](https://github.com/Devgoodcode)
- 💼 LinkedIn: [Gilles Blanchet](https://www.linkedin.com/in/gilles-blanchet-566ab759/)

## 🙏 Acknowledgments

This project is a labor of love, built on the shoulders of giants. Special thanks to:

* **🤗 Hugging Face**: For the fantastic *Agents Course*. It inspired me to create this connector to easily apply their concepts using my own existing tools (like Ollama & Infomaniak) without the hassle of writing wrappers.
* **My AI Co-pilots & Mentors**:
    * **Microsoft Copilot**: For the architectural breakthroughs (Families & Invoke concepts) and our late-night debates.
    * **Perplexity**: For laying down the initial code foundation.
    * **Google Gemini**: For the massive refactoring, patience, and pedagogical support in improving the core logic.
    * **Kilo Code (Kimi & Claude)**: For the security testing, English translation, and PyPI publishing preparation.
* **The Model Providers**: Ollama, Alibaba, Moonshot, MiniMax, OpenAI, Perplexity, Hugging Face, DeepSeek, Apertus, Anthropic, LangChain and Infomaniak for their incredible technologies and platforms.
* **The Open Source Community**: For the endless passion and knowledge sharing.

Built with ❤️ and passion, inspired by the open source AI community's need for a truly universal, maintainable LLM interface.

*The adventure is just beginning...*

---

**One Connector to Rule Them All** 🧙‍♂️