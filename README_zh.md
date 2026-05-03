# 🔌 Ideal AI - 通用 LLM 连接器

> **一个连接器，适配所有模型**

Ideal AI 是一个生产级的开源 **Python LLM 连接器**，为**文本、多模态、音频、图片生成、视频生成**提供统一接口，支持 15+ 家主流厂商（OpenAI、DeepSeek、阿里通义千问 Qwen、小米、百度、智谱 GLM、Moonshot、MiniMax、Infomaniak、Ollama 等）。

通过一个统一接口，轻松实现**本地模型（Ollama）与云端模型（OpenAI / DeepSeek / Qwen 等）之间的无缝切换**，特别适合需要多模型路由、混合部署的团队。

[![PyPI version](https://img.shields.io/pypi/v/ideal-ai.svg)](https://pypi.org/project/ideal-ai/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Devgoodcode/ideal-ai/blob/main/examples/demo_ideal_universal_connector.ipynb)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Idealcom/ideal-ai-llm-connector-demo)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

---

## ✨ 核心特性

- **🔗 统一接口**：一个接口适配 15+ 家厂商，包括中国主流 LLM 提供商（DeepSeek、Qwen、小米、百度、智谱 GLM 等）
- **🎯 多模态支持**：文本生成、视觉理解、音频转写（STT）、语音合成（TTS）、图片生成、视频生成（Wan 2.x）
- **💉 动态模型注入**：无需修改源代码，运行时即可注册任何新的模型或 Provider（支持 OpenAI-compatible 和 Ollama 模型）
- **🤖 原生适配代理框架**：为 **Smolagents** 提供原生封装，完全兼容 **LangChain** / **LangGraph** 工作流
- **🎙️ 原生语音对话**：开箱即用的端到端音频交互流程
- **🛡️ 生产级能力**：完善的错误处理、异步轮询（视频/音频）、重试机制、二进制资源管理
- **💼 100% 开源**：Apache 2.0 许可，可免费商用
- **📦 PIP 安装**：`pip install ideal-ai`

---

## 🚀 快速开始

### 安装

```bash
pip install ideal-ai
```

### 基本用法

```python
from ideal_ai import IdealUniversalLLMConnector
import os

# 初始化连接器，配置 API Keys
connector = IdealUniversalLLMConnector(
    api_keys={
        "deepseek": os.getenv("DEEPSEEK_API_KEY"),
        "alibaba": os.getenv("ALIBABA_API_KEY"),
    }
)

# 文本生成示例
response = connector.invoke(
    provider="deepseek",
    model_id="v4-pro",
    messages=[{"role": "user", "content": "用中文简单解释一下什么是多模态大模型？"}]
)
print(response["text"])

# 图像理解（多模态）
with open("image.jpg", "rb") as f:
    image_bytes = f.read()

analysis = connector.invoke_image(
    provider="alibaba",
    model_id="qwen3.6-35b",
    image_input=image_bytes,
    prompt="这张图片里有什么？"
)
print(analysis["text"])

# 图片生成
result = connector.invoke_image_generation(
    provider="alibaba",
    model_id="qwen-image-max",
    prompt="一个未来感十足的机器人行走在赛博朋克城市中"
)
# result["images"] 包含 base64 或 URL
```

---

## 🎯 已预配置的 Provider 与模型 – 完全可扩展（可在运行时注入任意 Provider / 模型）

以下 Provider 已在 `config.json` 中预配置，可以直接使用。  
**提示：** 你也可以在不修改源码的前提下，在运行时动态注入任意其他模型或 Provider（OpenAI-compatible、Ollama 等）。

| 提供商 (Provider) | 文本 (Text) | 视觉 / 多模态 (Vision) | 音频 STT (Audio) | 语音合成 TTS (Speech) | 图片生成 (Image Gen) | 视频生成 (Video Gen) |
|-------------------|:----------:|:----------------------:|:----------------:|:---------------------:|:--------------------:|:--------------------:|
| **OpenAI**        | ✅<br><small>gpt-4o, 3.5, 5</small> | ✅<br><small>gpt-4o, 5</small> | - | ✅<br><small>tts-1</small> | ✅<br><small>dall-e-3</small> | - |
| **Google**        | ✅<br><small>gemini-2.5</small> | ✅<br><small>gemini-2.5</small> | - | - | - | - |
| **Anthropic**     | ✅<br><small>claude-haiku-4.5</small> | ✅<br><small>claude-haiku-4-5</small> | - | - | - | - |
| **Ollama**        | ✅<br><small>llama3.2, r1, qwen3</small> | ✅<br><small>gemma3, llava, qwen3-vl</small> | - | - | ✅<br><small>flux2 (4b/9b), z-image</small> | - |
| **Alibaba**       | ✅<br><small>qwen3-max, plus, turbo, qwen3.6-35b</small> | ✅<br><small>qwen3.6-35b</small> | - | - | ✅<br><small>qwen-image-max</small> | ✅<br><small>wan2.1-2.5</small> |
| **Infomaniak**    | ✅<br><small>apertus-70b, mixtral</small> | - | ✅<br><small>whisper</small> | - | ✅<br><small>flux-schnell</small> | - |
| **DeepSeek**      | ✅<br><small>V3, R1, V4-Pro, V4-Flash</small> | - | - | - | - | - |
| **Moonshot**      | ✅<br><small>kimi-k2.5</small> | ✅<br><small>kimi-vision</small> | - | - | - | - |
| **Zhipu AI**      | ✅<br><small>glm-4.7</small> | ✅<br><small>glm-4.7</small> | - | - | - | - |
| **Baidu**         | ✅<br><small>ernie-3.5, 4.0</small> | - | - | - | - | - |
| **Perplexity**    | ✅<br><small>sonar</small> | - | - | - | - | - |
| **Hugging Face**  | ✅<br><small>gpt-oss-120b</small> | - | - | - | - | - |
| **MiniMax**       | ✅<br><small>M2</small> | - | - | - | - | - |
| **Mistral**       | ✅<br><small>Small 4</small> | - | - | - | - | - |
| **xAI**           | ✅<br><small>Grok 4</small> | - | - | - | - | - |
| **Xiaomi**        | ✅<br><small>MiMo-V2.5, V2.5-Pro</small> | ✅<br><small>MiMo-V2.5</small> | - | - | - | - |

## 📚 高级用法 

### 运行时动态添加自定义模型

```python
# 定义自定义模型配置
custom_model = {
    "myprovider:custom-model": {
        "api_key_name": "myprovider",
        "families": {
            "text": "openai_compatible"  # 复用已有的 recipe
        },
        "url_template": "https://api.myprovider.com/v1/chat/completions"
    }
}

# 初始化连接器并注入自定义模型
connector = IdealUniversalLLMConnector(
    api_keys={"myprovider": "your-api-key"},
    custom_models=custom_model
)

# 立即使用
response = connector.invoke("myprovider", "custom-model", messages)
```

### 语音转写（STT）

```python
# 使用 Infomaniak Whisper 进行音频转写
transcription = connector.invoke_audio(
    provider="infomaniak",
    model_id="whisper",
    audio_file_path="recording.m4a",
    language="zh"
)
print(transcription["text"])
```

### 语音合成（TTS）

```python
# 从文本生成语音
audio_result = connector.invoke_speech_generation(
    provider="openai",
    model_id="tts-1",
    text="你好，这是一个语音测试。",
    voice="nova"
)

# 保存音频文件
with open("output.mp3", "wb") as f:
    f.write(audio_result["audio_bytes"])
```

### 视频生成

```python
# 使用阿里巴巴 Wan 生成视频（自动处理异步轮询）
video_result = connector.invoke_video_generation(
    provider="alibaba",
    model_id="wan2.1-t2v-turbo",
    prompt="一个机器人行走在未来城市中",
    size="1280*720"
)
print(f"视频 URL: {video_result['videos'][0]}")
```

---

## 🔑 环境变量配置

创建 `.env` 文件或设置以下环境变量：

```bash
# 国际厂商
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AI...
ANTHROPIC_API_KEY=sk-ant-...

# 中国厂商
DEEPSEEK_API_KEY=sk-...
ALIBABA_API_KEY=sk-...
XIAOMI_API_KEY=...
BAIDU_API_KEY=...
ZHIPU_API_KEY=...

# 其他厂商
MISTRAL_API_KEY=...
XAI_API_KEY=...
INFOMANIAK_AI_TOKEN=...
INFOMANIAK_PRODUCT_ID=...

# 本地模型
OLLAMA_URL=http://localhost:11434
```

---

## 🤖 代理与工作流集成

Ideal AI 可作为底层 LLM 引擎，无缝嵌入到 **Smolagents**、**LangChain**、**LangGraph** 等代理框架和工作流系统中。

### Smolagents 集成

```python
from ideal_ai import IdealUniversalLLMConnector, IdealSmolagentsWrapper
from smolagents import CodeAgent

# 初始化连接器
connector = IdealUniversalLLMConnector(
    api_keys={"deepseek": os.getenv("DEEPSEEK_API_KEY")}
)

# 封装为 Smolagents 模型
model = IdealSmolagentsWrapper(
    connector=connector,
    provider="deepseek",
    model_id="v4-pro"
)

# 与任意 smolagents 代理协同工作
agent = CodeAgent(tools=[...], model=model)
agent.run("请帮我构建一个新闻文章爬虫")
```

### LangChain / LangGraph

Ideal AI 可直接在 **LangGraph** 节点或 **LangChain** 工作流中调用，无需复杂的封装器。

```python
from ideal_ai import IdealUniversalLLMConnector
from langgraph.graph import StateGraph

connector = IdealUniversalLLMConnector(
    api_keys={"deepseek": os.getenv("DEEPSEEK_API_KEY")}
)

# 直接在 LangGraph 节点中使用
def chatbot_node(state):
    response = connector.invoke(
        provider="deepseek",       # 随时切换 Provider！
        model_id="v4-pro",
        messages=state["messages"]
    )
    return {"messages": [response["text"]]}

# 构建工作流图...
workflow = StateGraph(dict)
workflow.add_node("chatbot", chatbot_node)
```

---

## ✅ 适用场景

- **多供应商管理**：适合需要统一接口管理多家 LLM 提供商的团队，降低对接复杂度和维护成本
- **混合部署架构**：需要在本地模型（Ollama）与云端模型（OpenAI / DeepSeek / Qwen）之间灵活切换，兼顾成本与性能
- **多模态 AI 应用**：希望快速搭建文本、图像、语音、视频等多模态 Agent、知识助手或原型应用的开发者
- **企业级生产环境**：需要稳定、可扩展、易维护的 LLM 接入方案，支持动态模型切换和故障转移

---

## 🏗️ 企业级架构模式

`ideal-ai` 采用模块化设计，适合在生产环境中通过 **服务层（Service Layer）** 封装，实现 AI 逻辑的集中管理。

这种方式让您可以**注入自定义解析器**、**动态切换 Provider**（例如开发环境使用 Ollama，生产环境使用 OpenAI），同时保持业务逻辑的清晰。

### 模式：`AIService` 封装

```python
# src/services/ai_service.py
from ideal_ai import IdealUniversalLLMConnector
import os

class AIService:
    """
    集中式 AI 交互服务层
    用于管理环境相关的逻辑（开发环境 vs 生产环境）
    """
    def __init__(self):
        # 初始化引擎一次
        self._engine = IdealUniversalLLMConnector(
            api_keys={
                "openai": os.getenv("OPENAI_API_KEY"),
                "ollama": os.getenv("OLLAMA_URL")
            }
        )

    def chat_with_user(self, user_message: str) -> str:
        """
        应用的简化接口
        集中管理使用哪个模型/Provider 的决策
        """
        # 逻辑：开发环境使用免费本地模型，生产环境使用强大的云端模型
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

### 该模式的优势

- ✅ **关注点分离**：业务逻辑保持清晰
- ✅ **环境感知**：开发环境使用本地模型，生产环境使用云端 API
- ✅ **Provider 抽象**：无需修改代码即可切换 Provider
- ✅ **易于测试**：可轻松 Mock 服务层
- ✅ **易于维护**：所有 AI 逻辑集中在一处

---

## 🧪 运行示例

查看 `examples/` 文件夹中的完整示例：

```bash
# 打开演示 Notebook
jupyter notebook examples/demo_ideal_universal_connector.ipynb
```

演示 Notebook 涵盖 **13 项综合功能**，按结构化、循序渐进的顺序展示：

| 步骤 | 功能 | 说明 |
|------|------|------|
| 0️⃣ | **安装** | 通过 `pip install -U ideal-ai` 快速设置 |
| 1️⃣ | **文本生成循环** | 统一遍历 15+ 家 Provider |
| 2️⃣ | **视觉/多模态** | 图像分析（Gemini、GPT-4o、Claude、Kimi、GLM、Qwen-VL） |
| 3️⃣ | **图片生成** | 创作艺术（DALL-E 3、Flux、Z-image、Qwen-Image） |
| 4️⃣ | **音频转写** | 使用 Infomaniak Whisper 自动轮询的 STT |
| 5️⃣ | **语音合成（TTS）** | 使用 OpenAI 的自然文本转语音 |
| 6️⃣ | **视频生成** | 使用阿里巴巴 Wan 的异步视频创建（自动轮询） |
| 7️⃣ | **运行时注入** | 即时注册自定义模型/Provider |
| 8️⃣ | **对话记忆** | 跨 Provider 的多轮聊天历史 |
| 9️⃣ | **AI 代理** | 使用 **smolagents** 的自主代理 |
| 🔟 | **自定义解析器** | 处理专有 API 响应格式 |
| 1️⃣1️⃣ | **调试模式** | 检查原始 API 负载和响应 |
| 1️⃣2️⃣ | **交互式测试界面** | **额外功能**：适用于所有模态的一体化图形化仪表板 |
| 1️⃣3️⃣ | **总结** | 下一步和致谢 |

---

## 🐛 调试模式

启用调试模式以检查负载和响应：

```python
response = connector.invoke(
    provider="deepseek",
    model_id="v4-pro",
    messages=[...],
    debug=True  # 显示原始 API 调用和响应
)
```

---

## 📦 从源代码安装

```bash
# 克隆仓库
git clone https://github.com/Devgoodcode/ideal-ai.git
cd ideal-ai

# 以开发模式安装
pip install -e .

# 或构建并安装
pip install build
python -m build
pip install dist/ideal_ai-*.whl
```

---

## 📖 文档与资源

- [GitHub 仓库](https://github.com/Devgoodcode/ideal-ai)
- [连接器 API 文档](https://github.com/Devgoodcode/ideal-ai/blob/main/src/ideal_ai/connector.py) - 完整的方法签名和文档字符串
- [配置 Schema](https://github.com/Devgoodcode/ideal-ai/blob/main/src/ideal_ai/config.json) - 可用的 families 和 models
- [示例代码](https://github.com/Devgoodcode/ideal-ai/tree/main/examples) - 可运行的代码示例

---

## 🤝 贡献

欢迎贡献！

添加新的 Provider：

1. 在 `config.json` 中添加 family 配置（或作为 `custom_families` 传递）
2. 使用该 family 添加模型配置
3. 使用演示 Notebook 进行测试

大多数情况下无需更改 Python 代码！

---

## 📝 开源协议

Apache License 2.0 - 详见 [LICENSE](LICENSE) 文件。
本项目可免费用于商业用途。

---

## 👤 作者与支持

**Gilles Blanchet**

- 🛠️ 创建者：[IA-Agence.ai](https://ia-agence.ai/ideal-ai-universal-llm-connector/) - *企业级 AI 架构与定制集成*
- 🌐 机构：[Idealcom.ch](https://idealcom.ch)
- 🐙 GitHub：[@Devgoodcode](https://github.com/Devgoodcode)
- 💼 LinkedIn：[Gilles Blanchet](https://www.linkedin.com/in/gilles-blanchet-566ab759/)

---

## 🙏 致谢

这个项目是用心打造的作品，建立在众多巨人的肩膀上。特别感谢：

* **🤗 Hugging Face**：感谢精彩的 *Agents 课程*。它激励我创建这个连接器，以便轻松应用他们的概念，使用我自己现有的工具（如 Ollama 和 Infomaniak），而无需编写封装器的麻烦。
* **我的 AI 协作伙伴与导师**：
    * **Microsoft Copilot**：提供架构突破（Families 和 Invoke 概念）以及深夜的辩论。
    * **Perplexity**：奠定最初的代码基础。
    * **Google Gemini**：进行大规模重构，耐心支持，并在改进核心逻辑方面提供教学支持。
    * **Kilo Code（Kimi 和 Claude）**：进行安全测试、英文翻译和 PyPI 发布准备。
* **模型提供商**：Ollama、Alibaba、Moonshot、MiniMax、OpenAI、Perplexity、Hugging Face、DeepSeek、Google、Zhipu AI、Baidu、Apertus、Anthropic、LangChain 和 Infomaniak，感谢他们令人惊叹的技术和平台。
* **开源社区**：感谢无尽的热情和知识分享。

用 ❤️ 和热情构建，灵感来自开源 AI 社区对真正通用、可维护的 LLM 接口的需求。

*冒险才刚刚开始...*

---

**一个连接器，适配所有模型** 🧙‍♂️
