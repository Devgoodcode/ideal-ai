# 🧩 Ideal AI - Universal LLM Connector

> **One Connector to Rule Them All**

A production-ready Python package providing a unified interface for multiple AI providers: Ollama, OpenAI, Google Gemini, Anthropic Claude, Alibaba Qwen, and more.

[![PyPI version](https://img.shields.io/pypi/v/ideal-ai.svg)](https://pypi.org/project/ideal-ai/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Devgoodcode/ideal-ai/blob/main/examples/demo_ideal_universal_connector.ipynb)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Idealcom/ideal-ai-llm-connector-demo)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## ✨ Features

- **🔗 Unified API** - One interface for 15+ LLM providers
- **🎯 Multi-Modal** - Text, Vision, Audio, Image Gen, Video Gen, Speech Synthesis
- **🔌 Plug & Play** - Add new models at runtime without code changes
- **🤖 Agent-Ready** - Built-in Smolagents wrapper for AI agents
- **🛡️ Production-Grade** - Async polling, binary handling, error recovery
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

## 🎯 Supported Providers & Modalities

| Provider | Text | Vision | Audio | Speech | Image Gen | Video Gen |
|----------|------|--------|-------|--------|-----------|-----------|
| **OpenAI** | ✅ | ✅ | - | ✅ | ✅ | - |
| **Google (Gemini)** | ✅ | ✅ | - | - | - | - |
| **Anthropic (Claude)** | ✅ | ✅ | - | - | - | - |
| **Ollama (Local)** | ✅ | ✅ | - | - | - | - |
| **Alibaba (Qwen)** | ✅ | - | - | - | - | ✅ |
| **Infomaniak** | ✅ | - | ✅ | - | ✅ | - |
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
jupyter notebook examples/demo.ipynb
```

The demo includes:
- Text generation with multiple providers
- Vision/multimodal analysis
- Image generation comparison
- Video generation (async)
- Audio transcription
- Speech synthesis
- Voice chat pipeline

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

## 🛠️ Supported Models (Default)

### Text Generation
- OpenAI: `gpt-4o`, `gpt-3.5-turbo`, `gpt-5`
- Google: `gemini-2.5-flash`
- Anthropic: `claude-haiku-4-5`
- Ollama: `llama3.2`, `qwen2:7b`, `deepseek-r1:8b`, `gemma3:1b`
- Alibaba: `qwen-turbo`, `qwen-plus`, `qwen3-max`
- Others: Moonshot, Perplexity, Hugging Face, MiniMax

### Vision/Multimodal
- OpenAI: `gpt-4o`
- Google: `gemini-2.5-flash`
- Anthropic: `claude-haiku-4-5`
- Ollama: `llava`, `qwen3-vl:30b`
- Moonshot: `moonshot-v1-8k-vision-preview`

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

Contributions welcome! To add a new provider:

1. Add family configuration to `config.json` (or pass as `custom_families`)
2. Add model configurations using that family
3. Test with the demo notebook

No Python code changes needed for most additions!

## 📝 License

Apache License 2.0 - See [LICENSE](LICENSE) file for details.

## 👤 Author & Support

**Gilles Blanchet**
- 🛠️ Created by: [IA-Agence.ai](https://ia-agence.ai/ideal-ai-universal-llm-connector/) - *Need help integrating Generative AI? Let's talk.*
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
* **The Model Providers**: Ollama, Alibaba, Moonshot, MiniMax, OpenAI, Perplexity, Hugging Face, Anthropic, LangChain and Infomaniak for their incredible technologies and platforms.
* **The Open Source Community**: For the endless passion and knowledge sharing.

Built with ❤️ and passion, inspired by the open source AI community's need for a truly universal, maintainable LLM interface.

*The adventure is just beginning...*

---

**One Connector to Rule Them All** 🧙‍♂️