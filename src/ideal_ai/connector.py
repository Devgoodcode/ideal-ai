"""
IdealUniversalLLMConnector - Universal connector for various LLM providers

A production-grade, extensible connector that provides a unified interface for:
- Text generation (OpenAI, Google, Anthropic, Alibaba, Ollama, etc.)
- Vision/Multimodal (Image analysis with GPT-4V, Gemini, Claude, etc.)
- Audio transcription (Whisper via Infomaniak)
- Speech synthesis (OpenAI TTS)
- Image generation (DALL-E, Flux, SDXL)
- Video generation (Alibaba Wan)

Features intelligent model resolution based on modality, supports custom models,
and includes a robust Smolagents wrapper for agent-based workflows.
"""

import os
import requests
import logging
import json
import string
import time
import base64
from io import BytesIO
from pathlib import Path
from typing import List, Union, Any, Callable, Dict, Optional

try:
    from PIL import Image as PILImage
except ImportError:
    print("Warning: PIL (Pillow) is not installed. Image utilities will not work.")
    PILImage = None

try:
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
except Exception:
    HumanMessage = SystemMessage = AIMessage = BaseMessage = None

try:
    import google.generativeai as genai
except Exception:
    genai = None


class IdealUniversalLLMConnector:
    """
    Universal LLM Connector providing unified access to multiple AI providers.
    
    This connector uses a declarative configuration system based on:
    - Families (templates/recipes): Define how to format, call, and parse API responses
    - Models (cards): Define which family to use for each modality
    
    The configuration can be extended at runtime with custom models and families.
    """
    
    def __init__(
        self,
        api_keys: Optional[Dict[str, str]] = None,
        ollama_url: str = "http://localhost:11434",
        custom_models: Optional[Dict[str, dict]] = None,
        custom_families: Optional[Dict[str, dict]] = None,
        custom_aliases: Optional[Dict[str, str]] = None,
        parsers: Optional[Dict[str, Callable[[Dict], str]]] = None
    ):
        """
        Initialize the universal connector.
        
        Args:
            api_keys: Dictionary of API keys (e.g., {"openai": "sk-...", "google": "..."})
            ollama_url: Base URL for local Ollama instance
            custom_models: Additional model configurations to register
            custom_families: Additional family configurations to register
            custom_aliases: Model aliases (e.g., {"gpt4": "openai:gpt-4o"})
            parsers: Custom response parsers for specific models
        """
        self.logger = logging.getLogger(__name__)
        self.api_keys = api_keys or {}
        self.ollama_url = ollama_url.rstrip("/")
        self.parsers = parsers or {}
        
        # Cache for recursive resolution (includes modality in key)
        self._resolve_cache: Dict[str, dict] = {}

        # Load configuration from JSON file
        config_path = Path(__file__).parent / "config.json"
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            self.default_families = config_data.get("families", {})
            self.default_model_configs = config_data.get("models", {})
            
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Loaded {len(self.default_families)} families and "
                                f"{len(self.default_model_configs)} models from config.json")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Configuration file not found at {config_path}. "
                "This file should be included in the package installation."
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file {config_path}: {e}")
        
        # Convert formatter/caller/parser strings to actual method references
        self._resolve_method_references()
        
        # Merge with user configurations
        self.families = {**self.default_families, **(custom_families or {})}
        self.model_configs = {**self.default_model_configs, **(custom_models or {})}
        self.aliases = custom_aliases or {}

    def _resolve_method_references(self) -> None:
        """
        Convert string method names in config.json to actual method references.
        This allows the JSON config to reference class methods by name.
        """
        for family_name, family_config in self.default_families.items():
            for key in ["formatter", "caller", "parser"]:
                method_name = family_config.get(key)
                if isinstance(method_name, str) and method_name.startswith("_"):
                    method = getattr(self, method_name, None)
                    if method and callable(method):
                        family_config[key] = method
                    else:
                        self.logger.warning(
                            f"Method '{method_name}' not found for family '{family_name}'"
                        )
        
        for model_key, model_config in self.default_model_configs.items():
            parser_name = model_config.get("parser")
            if isinstance(parser_name, str) and parser_name.startswith("_"):
                method = getattr(self, parser_name, None)
                if method and callable(method):
                    model_config["parser"] = method

    def register_model(self, key: str, config: dict) -> None:
        """
        Register a new model configuration at runtime.
        
        Args:
            key: Model identifier (format: "provider:model_id")
            config: Model configuration dictionary
        """
        self.model_configs[key] = config
        self._resolve_cache = {}

    def register_alias(self, alias: str, target_key: str) -> None:
        """
        Register a model alias.
        
        Args:
            alias: Alias name
            target_key: Target model key
        """
        self.aliases[alias] = target_key
        self._resolve_cache = {}

    def list_available_families(
        self, 
        include_models: bool = False
    ) -> Union[Dict[str, List[str]], Dict[str, Dict[str, List[str]]]]:
        """
        List available families (recipes) grouped by modality.

        Args:
            include_models: If True, return dict grouped by modality with associated models.
                          If False, return dict grouped by modality with just family names.

        Returns:
            Dictionary of available families, optionally with models
        """
        grouped_families_names: Dict[str, List[str]] = {
            "text": [],
            "vision": [],
            "audio": [],
            "image_gen": [],
            "speech_gen": [],
            "video_gen": [],
            "other": []
        }
        
        current_families = self.families

        for name in current_families.keys():
            # Categorize families by modality based on naming conventions
            if "vision" in name:
                grouped_families_names["vision"].append(name)
            elif "whisper" in name or "audio" in name:
                grouped_families_names["audio"].append(name)
            elif "image_gen" in name:
                grouped_families_names["image_gen"].append(name)
            elif "tts" in name or "speech" in name:
                grouped_families_names["speech_gen"].append(name)
            elif "video_gen" in name:
                grouped_families_names["video_gen"].append(name)
            elif ("text" in name or "openai" in name or "ollama" == name or 
                  "qwen" in name or "huggingface" in name or "google_sdk" in name or
                  "anthropic" in name or "alibaba_api" in name):
                grouped_families_names["text"].append(name)
            else:
                grouped_families_names["other"].append(name)

        # Sort each list
        for key in grouped_families_names:
            grouped_families_names[key].sort()
            
        # Remove empty groups
        grouped_families_names = {k: v for k, v in grouped_families_names.items() if v}

        if not include_models:
            return grouped_families_names

        # Build models-by-family structure
        models_by_family: Dict[str, Dict[str, List[str]]] = {}

        for model_key, model_config in self.model_configs.items():
            families_map = model_config.get("families")
            if not isinstance(families_map, dict):
                continue

            for modality, family_name in families_map.items():
                if modality not in models_by_family:
                    models_by_family[modality] = {}
                if family_name not in models_by_family[modality]:
                    models_by_family[modality][family_name] = []
                
                models_by_family[modality][family_name].append(model_key)

        # Sort model lists
        for modality in models_by_family:
            for family_name in models_by_family[modality]:
                models_by_family[modality][family_name].sort()

        final_result = grouped_families_names.copy()
        
        for modality in final_result:
            if modality in models_by_family:
                updated_families = {}
                for family_name in grouped_families_names[modality]:
                    family_config = self.families.get(family_name, {})
                    url_template_str = family_config.get("url_template")
                    
                    # Special case: Ollama
                    if not url_template_str and "ollama" in family_name:
                        url_template_str = f"[Uses ollama_url: {self.ollama_url}]"

                    # Special case: SDK
                    if not url_template_str and family_config.get("api_type") == "sdk":
                        url_template_str = "[Uses Python SDK (Google) - No direct HTTP URL]"
                    
                    # If URL still undefined, fetch from model level
                    if not url_template_str:
                        models_list = models_by_family.get(modality, {}).get(family_name, [])
                        detailed_models = []
                        
                        for model_id in models_list:
                            model_config = self.model_configs.get(model_id, {})
                            model_url = model_config.get("url_template", "[Not defined for this model]")
                            detailed_models.append({
                                "model": model_id,
                                "url_template": model_url
                            })
                        
                        family_data = {
                            "models": detailed_models,
                            "url_template": "See individual model URLs"
                        }
                    else:
                        family_data = {
                            "models": models_by_family.get(modality, {}).get(family_name, []),
                            "url_template": url_template_str
                        }
                    
                    updated_families[family_name] = family_data
                    
                final_result[modality] = updated_families
            else:
                final_result[modality] = {
                    fname: {
                        "models": [],
                        "url_template": self.families.get(fname, {}).get("url_template", "[Not defined]")
                    }
                    for fname in grouped_families_names[modality]
                }

        return final_result

    def resolve_model_config(self, provider: str, model_id: str, modality: str) -> dict:
        """
        Resolve complete configuration for (provider, model_id, modality) by applying:
        1. Alias resolution
        2. Inheritance (recursive)
        3. Family selection (based on modality)
        4. URL and header interpolation
        
        Returns a flat dictionary ready for use.
        
        Args:
            provider: Provider name (e.g., "openai", "google")
            model_id: Model identifier (e.g., "gpt-4o", "gemini-2.5-flash")
            modality: Modality type (e.g., "text", "vision", "audio")
            
        Returns:
            Resolved configuration dictionary
            
        Raises:
            ValueError: If model configuration is invalid or not found
        """
        base_key = f"{provider}:{model_id}"
        cache_key = f"{base_key}::{modality}"
        
        # Check cache (key includes modality)
        if cache_key in self._resolve_cache:
            return self._resolve_cache[cache_key]

        # Handle alias
        key = self.aliases.get(base_key, base_key)
        
        if key != base_key:
            try:
                provider, model_id = key.split(":", 1)
            except ValueError:
                raise ValueError(f"Alias '{base_key}' resolved to malformed key: '{key}'")

        # Load base model card
        model_cfg = self.model_configs.get(key)
        if not model_cfg:
            try:
                available_structure = self.list_available_families(include_models=True)
                available_structure_str = json.dumps(available_structure, indent=2)
            except Exception:
                available_structure_str = "Unable to list families/models (internal error)."

            error_message = (
                f"❌ No configuration found for '{key}' (resolved from '{base_key}').\n\n"
                f"💡 To add this model:\n\n"
                f"   1. **Choose the right 'recipe' (family)** for your task (modality).\n"
                f"      See available families below (`📋 Available families and models...`).\n"
                f"      *For Ollama text => `\"ollama_text\"`.*\n"
                f"      *For OpenAI-compatible API => `\"openai_compatible\"`.*\n\n"
                f"   2. **Create the model 'card'** in a `custom_models` dictionary.\n"
                f"      - The **key**: unique identifier (`\"provider:model_id\"`).\n"
                f"      - The **value**: dict containing `\"families\"` and, if needed, `\"url_template\"` or `\"api_key_name\"`.\n\n"
                f"      *Example 1: Adding '{key}' (if it's a standard Ollama text model)*\n"
                f"      ```python\n"
                f"      my_ollama_model = {{\n"
                f"          \"{key}\": {{\n"
                f"              \"families\": {{ \"text\": \"ollama_text\" }}\n"
                f"          }}\n"
                f"      }}\n"
                f"      ```\n\n"
                f"      *Example 2: Adding an Infomaniak model (using dynamic URL)*\n"
                f"      ```python\n"
                f"      my_infomaniak_model = {{\n"
                f"          \"infomaniak:mixtral-8x7b\": {{\n"
                f"              \"families\": {{ \"text\": \"openai_compatible\" }},\n"
                f"              \"api_key_name\": \"infomaniak\",\n"
                f"              \"url_template\": \"https://api.infomaniak.com/1/ai/$infomaniak_product/openai/chat/completions\"\n"
                f"          }}\n"
                f"      }}\n"
                f"      ```\n\n"
                f"   3. **Pass this dictionary** when initializing:\n"
                f"      ```python\n"
                f"      connector = IdealUniversalLLMConnector(\n"
                f"          custom_models=my_ollama_model,\n"
                f"          api_keys={{\"infomaniak\": \"YOUR_KEY\", \"infomaniak_product\": \"YOUR_PRODUCT_ID\"}}\n"
                f"      )\n"
                f"      ```\n\n"
                f"📋 Available families and models:\n"
                f"{available_structure_str}\n"
            )
            raise ValueError(error_message)
            
        model_cfg = model_cfg.copy()

        # Handle inheritance
        inherit = model_cfg.get("inherit_from")
        if inherit:
            try:
                base_cfg = self.resolve_model_config(provider, inherit, modality)
                merged = {**base_cfg, **model_cfg}
                merged.pop("inherit_from", None)
                model_cfg = merged
            except RecursionError:
                raise RecursionError(f"Circular dependency detected for '{key}'")
            except ValueError as e:
                raise ValueError(f"Error inheriting from '{provider}:{inherit}' for '{key}': {e}")

        # Apply family (based on modality)
        families_map = model_cfg.get("families", {})
        family_name = families_map.get(modality)

        if not family_name:
            raise ValueError(
                f"Model '{key}' does not support modality '{modality}'. "
                f"Supported modalities: {list(families_map.keys())}"
            )
        
        if family_name not in self.families:
            raise ValueError(
                f"Unknown family '{family_name}' (for modality '{modality}') for model '{key}'"
            )
        
        fam_cfg = self.families[family_name].copy()
        merged = {**fam_cfg, **model_cfg}
        merged.pop("families", None)
        model_cfg = merged

        # Interpolate templates
        infomaniak_product_id = self.api_keys.get("infomaniak_product")
        
        vars_dict = {
            "provider": provider,
            "model_id": model_id,
            "ollama_url": self.ollama_url,
            "infomaniak_product": infomaniak_product_id
        }
        
        if infomaniak_product_id:
            model_cfg["infomaniak_product"] = infomaniak_product_id
        
        api_key_name = model_cfg.get("api_key_name")
        if api_key_name:
            api_key = self.api_keys.get(api_key_name)
            if not api_key:
                self.logger.warning(f"API key '{api_key_name}' not provided for '{key}'.")
            vars_dict["api_key"] = api_key
            model_cfg["api_key"] = api_key
        
        vars_dict.update(model_cfg.get("template_vars", {}))

        try:
            url_tpl = model_cfg.get("url_template")
            if url_tpl:
                model_cfg["url"] = string.Template(url_tpl).substitute(vars_dict)

            headers_tpl = model_cfg.get("headers_template", {})
            if headers_tpl:
                model_cfg["headers"] = {
                    k: string.Template(v).substitute(vars_dict)
                    for k, v in headers_tpl.items()
                }
        except KeyError as e:
            raise ValueError(f"Missing template variable for '{key}': ${e.args[0]}.")

        # Final validation
        for req_key in ["formatter", "caller", "parser"]:
            if not model_cfg.get(req_key) or not callable(model_cfg[req_key]):
                raise ValueError(
                    f"Final configuration for '{key}' is invalid: "
                    f"'{req_key}' is missing or not callable."
                )

        # Cache and return
        self._resolve_cache[cache_key] = model_cfg
        return model_cfg

    # ====================================================================
    # INVOKE METHODS (Public API)
    # ====================================================================

    def invoke(
        self, 
        provider: str, 
        model_id: str, 
        messages: List[Union[dict, Any]], 
        debug: bool = False, 
        **kwargs
    ) -> dict:
        """
        Call a text LLM using dynamic configuration.
        
        Args:
            provider: Provider name (e.g., "openai", "ollama")
            model_id: Model identifier (e.g., "gpt-4o", "llama3.2")
            messages: List of message dicts or compatible objects
            debug: Enable debug output
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Dictionary with keys: "text" (parsed response), "raw" (original response)
        """
        if debug:
            print(f"\n[DEBUG INVOKE] Calling {provider}:{model_id} (modality: text)")
            
        try:
            config = self.resolve_model_config(provider, model_id, modality="text")
        except (ValueError, KeyError) as e:
            if debug:
                print(f"❌ [DEBUG INVOKE] Configuration error: {type(e).__name__}")
            else:
                self.logger.error(f"Configuration error for {provider}:{model_id}: {e}")
            raise e

        formatter = config["formatter"]
        caller = config["caller"]
        default_parser = config["parser"]
        key = f"{provider}:{model_id}"

        converted_messages = self._ensure_converted(messages)
        converted_messages = self._sanitize_messages_for_strict_apis(converted_messages)
        
        call_context = {
            **config,
            **kwargs,
            "model_id": model_id,
            "provider": provider,
            "debug": debug,
            "messages": converted_messages
        }

        # Call formatter
        payload = formatter(**call_context)

        if debug:
            try:
                # Mask sensitive data in debug output
                safe_payload = self._mask_sensitive_data(payload)
                payload_str = json.dumps(safe_payload, indent=2)
            except TypeError:
                payload_str = str(payload)
            print(f"[DEBUG INVOKE] Payload: {payload_str}")

        # Call network
        raw_response = caller(payload, **call_context)

        if debug:
            print(f"[DEBUG INVOKE] Raw response (type {type(raw_response)})")

        # Custom parser logic
        active_parser = self.parsers.get(key) or self.parsers.get(provider)

        if active_parser:
            if debug:
                print(f"🔧 [DEBUG INVOKE] Using custom parser for {key}...")
            try:
                parsed_text = active_parser(raw_response)
                return {"text": parsed_text, "raw": raw_response}
            except Exception as e:
                print(f"⚠️ [DEBUG INVOKE] Custom parser failed: {e}. Trying default parser.")

        # Use default parser
        try:
            parsed_text = default_parser(raw_response)
            
            if parsed_text is not None:
                return {"text": parsed_text, "raw": raw_response}
            
            if debug:
                print(f"⚠️ [DEBUG INVOKE] Default parser returned None.")

        except Exception as e:
            print(f"❌ [DEBUG INVOKE] Default parser failed: {e}")

        # Guided error message
        error_msg = (
            f"❌ Parsing failed for {provider}:{model_id}.\n"
            f"The API response format may have changed or the model is unsupported.\n\n"
            f"👉 QUICK FIX:\n"
            f"1. Enable debug: invoke(..., debug=True) to see raw JSON above.\n"
            f"2. Write a function: def my_parser(raw): return raw['your_path_to_text']\n"
            f"3. Inject it: connector = IdealUniversalLLMConnector(..., parsers={{'{key}': my_parser}})"
        )
        
        return {
            "text": None,
            "error": error_msg,
            "raw": raw_response
        }

    def invoke_image(
        self, 
        provider: str, 
        model_id: str, 
        image_input: Any, 
        prompt: str, 
        debug: bool = False, 
        **kwargs
    ) -> dict:
        """
        Call a vision/multimodal model for image analysis.
        
        Args:
            provider: Provider name
            model_id: Model identifier
            image_input: Image as bytes or PIL.Image object
            prompt: Text prompt describing the task
            debug: Enable debug output
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with keys: "text" (analysis result), "raw" (original response)
        """
        if debug:
            print(f"\n[DEBUG INVOKE_IMAGE] Calling {provider}:{model_id} (modality: vision)")

        # Handle image input
        image_pil = None
        image_bytes_for_b64 = None

        if PILImage and isinstance(image_input, PILImage.Image):
            image_pil = image_input
            if provider != "google":
                if debug:
                    print("[DEBUG INVOKE_IMAGE] Converting PIL -> bytes for Base64")
                img_io = BytesIO()
                image_pil.save(img_io, format='PNG')
                image_bytes_for_b64 = img_io.getvalue()
            else:
                if debug:
                    print("[DEBUG INVOKE_IMAGE] PIL.Image detected, passing directly for Gemini.")
        elif isinstance(image_input, bytes):
            image_bytes_for_b64 = image_input
            if provider == "google":
                if not PILImage:
                    raise ImportError("PIL (Pillow) required to convert bytes -> PIL for Gemini.")
                try:
                    if debug:
                        print("[DEBUG INVOKE_IMAGE] Converting bytes -> PIL for Gemini SDK")
                    image_pil = PILImage.open(BytesIO(image_bytes_for_b64))
                except Exception as e:
                    raise ValueError(f"Unable to open image (bytes) as PIL for Gemini: {e}")
            else:
                if debug:
                    print("[DEBUG INVOKE_IMAGE] Bytes detected, passing directly for Base64.")
        else:
            raise TypeError(f"image_input must be 'bytes' or 'PIL.Image', received {type(image_input)}")

        # Main logic
        try:
            config = self.resolve_model_config(provider, model_id, modality="vision")
        except (ValueError, KeyError) as e:
            print(f"❌ [DEBUG INVOKE_IMAGE] Configuration error: {e}")
            raise e

        formatter = config["formatter"]
        caller = config["caller"]
        parser = config["parser"]

        call_context = {
            **config,
            **kwargs,
            "model_id": model_id,
            "provider": provider,
            "debug": debug,
            "prompt": prompt
        }
        
        # Add image to context in correct format
        if provider == "google":
            call_context["image_object_pil"] = image_pil
        else:
            call_context["image_bytes"] = image_bytes_for_b64

        payload = formatter(**call_context)
        raw_response = caller(payload, **call_context)
        parsed_text = parser(raw_response)

        return {"text": parsed_text, "raw": raw_response}

    def invoke_audio(
        self,
        provider: str,
        model_id: str,
        audio_file_path: str,
        language: Optional[str] = None,
        debug: bool = False,
        **kwargs
    ) -> dict:
        """
        Transcribe audio file to text.
        
        Args:
            provider: Provider name (e.g., "infomaniak")
            model_id: Model identifier (e.g., "whisper")
            audio_file_path: Path to audio file
            language: Language code (e.g., "en", "fr")
            debug: Enable debug output
            **kwargs: Additional parameters (max_wait, poll_interval, etc.)
            
        Returns:
            Dictionary with keys: "text" (transcription), "raw" (original response)
        """
        if debug:
            print(f"\n[DEBUG INVOKE_AUDIO] Calling {provider}:{model_id} (modality: audio)")
            
        try:
            config = self.resolve_model_config(provider, model_id, modality="audio")
        except (ValueError, KeyError) as e:
            print(f"❌ [DEBUG INVOKE_AUDIO] Configuration error: {e}")
            raise e
        
        formatter = config["formatter"]
        caller = config["caller"]
        parser = config["parser"]

        call_context = {
            **config,
            **kwargs,
            "model_id": model_id,
            "provider": provider,
            "debug": debug,
            "audio_file_path": audio_file_path,
            "language": language
        }
        
        payload = formatter(**call_context)
        raw_final_response = caller(payload, **call_context)
        parsed_text = parser(raw_final_response)

        if parsed_text:
            parsed_text = parsed_text.strip()
            
        return {"text": parsed_text, "raw": raw_final_response}

    def invoke_speech_generation(
        self,
        provider: str,
        model_id: str,
        text: str,
        voice: str = "alloy",
        response_format: str = "mp3",
        debug: bool = False,
        **kwargs
    ) -> dict:
        """
        Generate audio from text (Text-to-Speech).
        
        Args:
            provider: Provider name (e.g., "openai")
            model_id: Model identifier (e.g., "tts-1")
            text: Text to convert to speech
            voice: Voice name (e.g., "alloy", "nova", "shimmer")
            response_format: Audio format (e.g., "mp3", "opus")
            debug: Enable debug output
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with keys: "audio_bytes" (binary audio data), "raw" (original response)
        """
        if debug:
            print(f"\n[DEBUG SPEECH_GEN] Calling {provider}:{model_id} (modality: speech_gen)")

        try:
            config = self.resolve_model_config(provider, model_id, modality="speech_gen")
        except (ValueError, KeyError) as e:
            print(f"❌ [DEBUG SPEECH_GEN] Configuration error: {e}")
            raise e

        formatter = config["formatter"]
        caller = config["caller"]
        parser = config["parser"]

        call_context = {
            **config,
            **kwargs,
            "model_id": model_id,
            "provider": provider,
            "debug": debug,
            "text": text,
            "voice": voice,
            "response_format": response_format,
        }
        
        payload = formatter(**call_context)
        
        if debug:
            safe_payload = self._mask_sensitive_data(payload)
            print(f"[DEBUG SPEECH_GEN] Payload: {json.dumps(safe_payload, indent=2)}")

        raw_bytes = caller(payload, **call_context)
        parsed_bytes = parser(raw_bytes)

        return {"audio_bytes": parsed_bytes, "raw": raw_bytes}

    def invoke_image_generation(
        self,
        provider: str,
        model_id: str,
        prompt: str,
        num_images: int = 1,
        width: int = 1024,
        height: int = 1024,
        debug: bool = False,
        **kwargs
    ) -> dict:
        """
        Generate one or more images from a text prompt.
        
        Args:
            provider: Provider name (e.g., "openai", "infomaniak")
            model_id: Model identifier (e.g., "dall-e-3", "flux-schnell")
            prompt: Image description prompt
            num_images: Number of images to generate
            width: Image width in pixels
            height: Image height in pixels
            debug: Enable debug output
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with keys: "images" (list of base64 strings or URLs), "raw"
        """
        if debug:
            print(f"\n[DEBUG IMAGE_GEN] Calling {provider}:{model_id} (modality: image_gen)")
            
        try:
            config = self.resolve_model_config(provider, model_id, modality="image_gen")
        except (ValueError, KeyError) as e:
            print(f"❌ [DEBUG IMAGE_GEN] Configuration error: {e}")
            raise e

        formatter = config["formatter"]
        caller = config["caller"]
        parser = config["parser"]

        call_context = {
            **config,
            **kwargs,
            "model_id": model_id,
            "provider": provider,
            "debug": debug,
            "prompt": prompt,
            "num_images": num_images,
            "width": width,
            "height": height,
        }
        
        payload = formatter(**call_context)
        
        if debug:
            safe_payload = self._mask_sensitive_data(payload)
            print(f"[DEBUG IMAGE_GEN] Payload: {json.dumps(safe_payload, indent=2)}")

        raw_response = caller(payload, **call_context)
        parsed_images = parser(raw_response)

        return {"images": parsed_images, "raw": raw_response}

    def invoke_video_generation(
        self,
        provider: str,
        model_id: str,
        prompt: str,
        size: str = "832*480",
        debug: bool = False,
        **kwargs
    ) -> dict:
        """
        Generate video from a text prompt.
        
        Args:
            provider: Provider name (e.g., "alibaba")
            model_id: Model identifier (e.g., "wan2.1-t2v-turbo")
            prompt: Video description prompt
            size: Video resolution (e.g., "832*480", "1280*720")
            debug: Enable debug output
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with keys: "videos" (list of URLs), "raw" (original response)
        """
        if debug:
            print(f"\n[DEBUG VIDEO_GEN] Calling {provider}:{model_id} (modality: video_gen)")

        try:
            config = self.resolve_model_config(provider, model_id, modality="video_gen")
        except (ValueError, KeyError) as e:
            print(f"❌ [DEBUG VIDEO_GEN] Configuration error: {e}")
            raise e

        formatter = config["formatter"]
        caller = config["caller"]
        parser = config["parser"]

        call_context = {
            **config,
            **kwargs,
            "model_id": model_id,
            "provider": provider,
            "debug": debug,
            "prompt": prompt,
            "size": size,
        }
        
        payload = formatter(**call_context)
        
        if debug:
            safe_payload = self._mask_sensitive_data(payload)
            print(f"[DEBUG VIDEO_GEN] Payload: {json.dumps(safe_payload, indent=2)}")

        # Caller handles submission AND polling
        raw_final_response = caller(payload, **call_context)
        parsed_videos = parser(raw_final_response)

        return {"videos": parsed_videos, "raw": raw_final_response}

    # ====================================================================
    # FORMATTERS (Used by families)
    # ====================================================================

    def _format_for_google_text(self, **context) -> dict:
        """Format messages for Google Gemini SDK (text-only)."""
        messages = context.get("messages", [])
        return {
            "messages": messages,
            "temperature": context.get("temperature", 0.7)
        }

    def _format_for_openai_style(self, **context) -> dict:
        """Format messages for OpenAI-compatible APIs."""
        messages = context.get("messages", [])
        model_name = context.get("api_model_name", context.get("model_id"))
        
        return {
            "model": model_name,
            "messages": messages,
            "temperature": context.get("temperature", 0.1)
        }

    def _format_for_huggingface(self, **context) -> dict:
        """Format messages for Hugging Face Inference API."""
        messages = context.get("messages", [])
        return {
            "inputs": messages[-1]["content"],
            "parameters": {"temperature": context.get("temperature", 0.7)}
        }

    def _format_for_qwen(self, **context) -> dict:
        """Format messages for Alibaba Qwen API."""
        messages = context.get("messages", [])
        return {
            "model": context.get("model_id"),
            "input": {"messages": messages},
            "parameters": {"temperature": context.get("temperature", 0.7)}
        }

    def _format_for_ollama(self, **context) -> dict:
        """Format messages for Ollama API (supports multimodal text-invoke)."""
        messages = context.get("messages", [])
        
        final_messages = []
        for msg in messages:
            if "images" in msg and msg.get("role") == "user":
                final_messages.append({
                    "role": "user",
                    "content": msg["content"],
                    "images": msg["images"]
                })
            else:
                final_messages.append(msg)

        return {
            "model": context.get("model_id"),
            "messages": final_messages,
            "stream": False,
            "options": {"temperature": context.get("temperature", 0.7)}
        }

    def _format_for_anthropic(self, **context) -> dict:
        """Format messages for Anthropic Claude API."""
        messages = context.get("messages", [])
        system_prompt = None
        anthropic_messages = []
        
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content")
            elif msg.get("role") in ["user", "assistant"]:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        payload = {
            "model": context.get("model_id"),
            "messages": anthropic_messages,
            "max_tokens": context.get("max_tokens", 1024),
            "temperature": context.get("temperature", 0.7)
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        return payload

    def _format_for_google_vision(self, **context) -> dict:
        """Format payload for Gemini Vision SDK."""
        image_obj = context.get("image_object_pil")
        if not image_obj:
            raise ValueError("PIL image object missing in context for Gemini Vision.")
        
        return {
            "content": [
                context.get("prompt"),
                image_obj
            ]
        }

    def _format_for_ollama_vision(self, **context) -> dict:
        """Format payload for Ollama Vision (from invoke_image)."""
        prompt = context.get("prompt")
        image_bytes = context.get("image_bytes")
        
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        messages = [{
            "role": "user",
            "content": prompt,
            "images": [b64]
        }]
        
        return {
            "model": context.get("model_id"),
            "messages": messages,
            "stream": False,
            "options": {"temperature": context.get("temperature", 0.7)}
        }

    def _format_for_ollama_vision_generate(self, **context) -> dict:
        """Format simple payload for Ollama /api/generate endpoint (Gemma3)."""
        prompt = context.get("prompt")
        image_bytes = context.get("image_bytes")
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        
        return {
            "model": context.get("model_id"),
            "prompt": prompt,
            "images": [b64],
            "stream": False,
            "options": {"temperature": context.get("temperature", 0.7)}
        }

    def _format_for_openai_vision_style(self, **context) -> dict:
        """
        Format payload for OpenAI-compatible vision APIs (GPT-4V, Moonshot Vision).
        Includes Base64-encoded image in user content.
        """
        prompt = context.get("prompt")
        image_bytes = context.get("image_bytes")
        
        if not image_bytes:
            raise ValueError("Image bytes missing for OpenAI-style vision formatting.")
            
        b64_string = base64.b64encode(image_bytes).decode("utf-8")
        
        user_content = [
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_string}"
                }
            }
        ]
        
        messages = [{
            "role": "user",
            "content": user_content
        }]
        
        return {
            "model": context.get("model_id"),
            "messages": messages,
            "temperature": context.get("temperature", 0.1)
        }

    def _format_for_anthropic_vision(self, **context) -> dict:
        """
        Format payload for Anthropic Claude Vision API.
        Accepts image as bytes and text prompt.
        """
        prompt = context.get("prompt")
        image_bytes = context.get("image_bytes")
        
        if not image_bytes:
            raise ValueError("Image bytes missing for Anthropic vision formatting.")
        
        b64_string = base64.b64encode(image_bytes).decode("utf-8")
        
        # Detect media type
        media_type = "image/jpeg"
        if image_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
            media_type = "image/png"
        elif image_bytes.startswith(b'\xFF\xD8\xFF'):
            media_type = "image/jpeg"
        
        user_content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64_string
                }
            },
            {
                "type": "text",
                "text": prompt
            }
        ]

        payload = {
            "model": context.get("model_id"),
            "messages": [
                {"role": "user", "content": user_content}
            ],
            "max_tokens": context.get("max_tokens", 1024),
            "temperature": context.get("temperature", 0.7)
        }
        
        return payload

    def _format_for_whisper(self, **context) -> dict:
        """Format data dict for Whisper multipart upload."""
        data = {"model": "whisper"}
        if context.get("language"):
            data["language"] = context.get("language")
        return data

    def _format_for_openai_tts(self, **context) -> dict:
        """Format payload for OpenAI TTS API."""
        model_name = context.get("model_id")
        
        payload = {
            "model": model_name,
            "input": context.get("text"),
            "voice": context.get("voice", "alloy"),
            "response_format": context.get("response_format", "mp3")
        }
        
        if context.get("speed"):
            payload["speed"] = context.get("speed")
            
        return payload

    def _format_for_image_gen(self, **context) -> dict:
        """Format JSON payload for image generation APIs (Infomaniak, OpenAI)."""
        model_name = context.get("api_model_name", context.get("model_id"))
        
        payload = {
            "model": model_name,
            "prompt": context.get("prompt"),
            "n": min(max(1, context.get("num_images", 1)), 5),
            "size": f"{context.get('width', 1024)}x{context.get('height', 1024)}",
            "quality": "standard",
            "response_format": context.get("response_format", "b64_json")
        }
        
        if context.get("negative_prompt"):
            payload["negative_prompt"] = context.get("negative_prompt")
        if context.get("style"):
            payload["style"] = context.get("style")
        if context.get("guidance_scale"):
            payload["guidance_scale"] = float(context.get("guidance_scale"))
            
        return payload

    def _format_for_alibaba_video_gen(self, **context) -> dict:
        """Format JSON payload for Alibaba (Qwen) Video Generation."""
        model_name = context.get("api_model_name", context.get("model_id"))
        size = context.get("size", "832*480")
        
        parameters = {
            "size": size,
            "prompt_extend": True
        }
        
        if context.get("style"):
            parameters["style"] = context.get("style")
        if context.get("negative_prompt"):
            parameters["negative_prompt"] = context.get("negative_prompt")
        
        payload = {
            "model": model_name,
            "input": {
                "prompt": context.get("prompt")
            }
        }
        
        if parameters:
            payload["parameters"] = parameters
            
        return payload

    # ====================================================================
    # CALLERS (Used by families)
    # ====================================================================

    def _call_google_sdk_text(self, payload: dict, **context) -> Any:
        """Call Google Gemini SDK (text-only)."""
        if not genai:
            raise ImportError("Please install google-generativeai: pip install google-generativeai")
        
        api_key = context.get("api_key")
        if not api_key:
            raise ValueError("Gemini API key missing (expected in resolved config)")
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(context.get("model_id"))
        
        messages_list = payload.get("messages", [])
        last = ""
        if isinstance(messages_list, list):
            for msg in reversed(messages_list):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    last = msg.get("content")
                    break
                if isinstance(msg, str):
                    last = msg
                    break
        
        if not last:
            raise ValueError("No user message found for Gemini.")
            
        response = model.generate_content(last)
        return response

    def _call_ollama(self, payload: dict, **context) -> dict:
        """Call Ollama API (JSON)."""
        url = context.get("url")
        if not url:
            raise ValueError("Ollama URL missing in resolved config")

        timeout_seconds = context.get("request_timeout", 180)
        if context.get("debug", False):
            print(f"[DEBUG OLLAMA CALLER] Using timeout: {timeout_seconds}s")

        try:
            response = requests.post(url, json=payload, timeout=timeout_seconds)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"Ollama did not respond within {timeout_seconds} seconds for URL {url}."
            )
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Error connecting to Ollama ({url}): {e}")

    def _call_http_post(self, payload: dict, **context) -> dict:
        """Generic caller for all HTTP POST-based APIs (JSON)."""
        url = context.get("url")
        headers = context.get("headers")
        debug = context.get("debug", False)
        
        if not url:
            raise ValueError("[CALLER] URL missing in context")
        if not headers:
            raise ValueError("[CALLER] Headers missing in context")

        timeout_seconds = context.get("request_timeout", 90)

        if debug:
            # Mask API key in debug output
            safe_headers = self._mask_api_key_in_headers(headers)
            print(f"✅ [DEBUG CALLER] Sending POST to {url}")
            print(f"✅ [DEBUG CALLER] Headers: {safe_headers}")
            print(f"✅ [DEBUG CALLER] Using timeout: {timeout_seconds}s")

        response = requests.post(url, json=payload, headers=headers, timeout=timeout_seconds)
        
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"❌ [DEBUG CALLER] HTTP error {e.response.status_code} for {url}: {e.response.text}")
            raise e
            
        return response.json()

    def _call_google_sdk_vision(self, payload: dict, **context) -> Any:
        """Call Google Gemini SDK with multimodal content."""
        if not genai:
            raise ImportError("Please install google-generativeai: pip install google-generativeai")
        
        api_key = context.get("api_key")
        if not api_key:
            raise ValueError("Gemini API key missing (expected in resolved config)")
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(context.get("model_id"))
        
        content_list = payload.get("content")
        if not content_list or len(content_list) < 2:
            raise ValueError(f"Invalid vision payload for Gemini. Received: {payload}")
            
        response = model.generate_content(content_list)
        return response

    def _call_http_post_binary(self, payload: dict, **context) -> bytes:
        """Generic caller for HTTP POST APIs returning binary data (audio/image)."""
        url = context.get("url")
        headers = context.get("headers")
        debug = context.get("debug", False)
        
        if not url:
            raise ValueError("[CALLER BINARY] URL missing in context")

        if debug:
            safe_headers = self._mask_api_key_in_headers(headers)
            print(f"✅ [DEBUG CALLER BINARY] Sending POST to {url}. Expecting binary content.")
            print(f"✅ [DEBUG CALLER BINARY] Headers: {safe_headers}")

        response = requests.post(url, json=payload, headers=headers, timeout=90)
        
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            try:
                error_details = response.json()
                print(f"❌ [DEBUG CALLER BINARY] HTTP error {e.response.status_code}: {error_details}")
            except json.JSONDecodeError:
                print(f"❌ [DEBUG CALLER BINARY] HTTP error {e.response.status_code}: {e.response.text[:100]}...")
            raise e
            
        return response.content

    def _call_and_poll_infomaniak_whisper(self, payload: dict, **context) -> dict:
        """
        Handle multipart upload AND polling for Infomaniak Whisper.
        CRITICAL: Preserves async polling logic for production.
        """
        debug = context.get("debug", False)
        
        api_key = context.get("api_key")
        product_id = context.get("infomaniak_product")
        url_post = context.get("url")
        headers = context.get("headers", {})
        audio_file_path = context.get("audio_file_path")

        if not api_key or not product_id:
            raise ValueError("Infomaniak key or Product ID missing")
        if not url_post:
            raise ValueError("Whisper upload URL missing")
        if not audio_file_path:
            raise ValueError("audio_file_path missing")
            
        data = payload

        try:
            with open(audio_file_path, "rb") as f:
                files = {"file": f}
                if debug:
                    print(f"[DEBUG WHISPER] POST to {url_post} with data={data}")
                resp = requests.post(url_post, headers=headers, files=files, data=data, timeout=90)
                resp.raise_for_status()
                result = resp.json()
                if debug:
                    print(f"[DEBUG WHISPER] Raw POST response: {result}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Error during POST to Infomaniak: {exc}")

        # If immediate result available
        if isinstance(result, dict) and result.get("text"):
            return result

        # Otherwise, poll for async result
        batch_id = result.get("batch_id")
        if not batch_id:
            raise RuntimeError(f"No batch_id received. Raw response: {result}")

        url_result = f"https://api.infomaniak.com/1/ai/{product_id}/results/{batch_id}"
        max_wait = context.get("max_wait", 300)
        poll_interval = context.get("poll_interval", 15)
        
        start = time.time()
        while time.time() - start < max_wait:
            if debug:
                print(f"[DEBUG WHISPER POLL] GET {url_result}")
            try:
                polling = requests.get(url_result, headers=headers, timeout=30)
            except requests.exceptions.RequestException as exc:
                if debug:
                    print(f"[DEBUG WHISPER POLL] GET request error: {exc}")
                time.sleep(poll_interval)
                continue

            if polling.status_code == 200:
                try:
                    contenu = polling.json()
                    if debug:
                        print("[DEBUG WHISPER POLL] JSON response received:", contenu)
                    
                    texte = contenu.get("data") or contenu.get("text") or contenu.get("result")
                    if texte and str(texte).lower() != "processing":
                        return contenu
                        
                    if contenu.get("url"):
                        try:
                            dl_resp = requests.get(contenu["url"], timeout=30)
                            dl_resp.raise_for_status()
                            return dl_resp.text
                        except requests.exceptions.RequestException as exc:
                            if debug:
                                print(f"[DEBUG WHISPER POLL] Error downloading URL: {exc}")
                
                except json.JSONDecodeError:
                    if isinstance(polling.text, str) and polling.text.strip():
                        if debug:
                            print("[DEBUG WHISPER POLL] Raw text response received.")
                        return {"text": polling.text}
                    if debug:
                        print("[DEBUG WHISPER POLL] Empty or non-JSON response.")
            else:
                if debug:
                    print(f"[DEBUG WHISPER POLL] Unexpected HTTP code: {polling.status_code} - body: {polling.text}")
            
            time.sleep(poll_interval)

        raise TimeoutError(f"Transcription not available after {max_wait} seconds.")

    def _call_and_poll_alibaba_video(self, payload: dict, **context) -> dict:
        """
        Handle submission (POST) AND polling (GET) for Alibaba Video Generation.
        CRITICAL: Preserves async polling logic for production.
        """
        debug = context.get("debug", False)
        
        api_key = context.get("api_key")
        url_post = context.get("url")
        poll_url_template_str = context.get("poll_url_template")
        headers = context.get("headers", {})
        
        if not api_key:
            raise ValueError("[ALIBABA VIDEO] Alibaba API key missing")
        if not url_post or not poll_url_template_str:
            raise ValueError("[ALIBABA VIDEO] Submission or polling URL missing")

        # 1. Submit task (POST)
        try:
            if debug:
                print(f"[DEBUG ALIBABA VIDEO] POST to {url_post} with payload...")
            resp_post = requests.post(url_post, headers=headers, json=payload, timeout=30)
            resp_post.raise_for_status()
            result_post = resp_post.json()
            if debug:
                print(f"[DEBUG ALIBABA VIDEO] POST response: {result_post}")
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Error during POST to Alibaba Video: {exc}")

        task_id = result_post.get("output", {}).get("task_id")
        if not task_id:
            raise RuntimeError(f"No task_id received from Alibaba. Response: {result_post}")

        # 2. Poll task (GET)
        url_poll = string.Template(poll_url_template_str).substitute({"task_id": task_id})
        
        max_wait = context.get("max_wait", 300)
        poll_interval = context.get("poll_interval", 10)
        
        start = time.time()
        while time.time() - start < max_wait:
            if debug:
                print(f"[DEBUG ALIBABA VIDEO POLL] GET {url_poll}")
            try:
                polling = requests.get(url_poll, headers=headers, timeout=30)
                polling.raise_for_status()
                contenu = polling.json()
                
                if debug:
                    print(f"[DEBUG ALIBABA VIDEO POLL] GET response: {contenu}")

                task_status = contenu.get("output", {}).get("task_status")

                if task_status == "SUCCEEDED":
                    if debug:
                        print("[DEBUG ALIBABA VIDEO POLL] Task succeeded!")
                    return contenu
                
                if task_status == "FAILED":
                    raise RuntimeError(
                        f"Alibaba Video task failed: {contenu.get('output', {}).get('message')}"
                    )

                # Status is "PENDING" or "RUNNING", wait
                time.sleep(poll_interval)
                
            except requests.exceptions.RequestException as exc:
                if debug:
                    print(f"[DEBUG ALIBABA VIDEO POLL] GET request error: {exc}")
                time.sleep(poll_interval)
            
        raise TimeoutError(
            f"Alibaba video generation not available after {max_wait} seconds for task {task_id}."
        )

    # ====================================================================
    # PARSERS (Used by families)
    # ====================================================================

    def _parse_google(self, raw: Any) -> Optional[str]:
        """Parse Google Gemini SDK response."""
        return getattr(raw, "text", None) or (raw.get("text") if isinstance(raw, dict) else None)

    def _parse_openai_style(self, raw: Any) -> Optional[str]:
        """Parse OpenAI-compatible API response."""
        try:
            return raw.get("choices", [])[0].get("message", {}).get("content")
        except (AttributeError, IndexError, TypeError):
            return None

    def _parse_qwen(self, raw: Any) -> Optional[str]:
        """Parse Alibaba Qwen API response."""
        try:
            return raw.get("output", {}).get("text")
        except (AttributeError, TypeError):
            return None

    def _parse_ollama(self, raw: Any) -> Optional[str]:
        """Parse Ollama /api/chat response."""
        try:
            return raw.get("message", {}).get("content")
        except (AttributeError, TypeError):
            return None

    def _parse_ollama_generate(self, raw: Any) -> Optional[str]:
        """Parse Ollama /api/generate response."""
        try:
            return raw.get("response")
        except (AttributeError, TypeError):
            return None

    def _parse_huggingface(self, raw: Any) -> Optional[str]:
        """Parse Hugging Face Inference API response."""
        try:
            return raw[0].get("generated_text")
        except (AttributeError, IndexError, TypeError):
            return None

    def _parse_whisper_final_response(self, raw: Any) -> Optional[str]:
        """Parse Infomaniak Whisper final response (handles various formats)."""
        if isinstance(raw, str):
            try:
                data = json.loads(raw)
                return data.get("text")
            except json.JSONDecodeError:
                return raw.strip()
        
        if not isinstance(raw, dict):
            return None
            
        if "text" in raw:
            return raw.get("text")
            
        texte = raw.get("data") or raw.get("result")
        
        if not texte:
            if "result" in raw and not raw.get("data"):
                return raw.get("result")
            return None
            
        try:
            if isinstance(texte, str) and texte.strip().startswith("{") and "text" in texte:
                texte_dict = json.loads(texte)
                return texte_dict.get("text", "").strip()
            else:
                return str(texte).strip()
        except Exception as e:
            self.logger.warning(f"Error parsing Whisper response: {e}. Raw: {texte}")
            return str(texte)

    def _parse_audio_output(self, raw: bytes) -> bytes:
        """Parser for binary outputs (audio). Returns raw bytes."""
        if isinstance(raw, bytes):
            return raw
        raise TypeError("Audio parser expects binary data (bytes).")

    def _parse_minimax_clean(self, raw: Any) -> Optional[str]:
        """
        Parse Minimax response (OpenAI-compatible) and remove <think> tags.
        """
        import re
        
        try:
            raw_text = raw.get("choices", [])[0].get("message", {}).get("content")
        except (AttributeError, IndexError, TypeError):
            return None

        if not raw_text:
            return None

        # Remove <think> tags and their content
        cleaned_text = re.sub(r'<think>.*?<\/think>', '', raw_text, flags=re.IGNORECASE | re.DOTALL)
        
        return cleaned_text.strip()

    def _parse_image_gen_response(self, raw: Any) -> Optional[List[str]]:
        """Parse image generation API response (URLs or base64)."""
        images = []
        try:
            if isinstance(raw, dict) and raw.get("data"):
                for item in raw["data"]:
                    if isinstance(item, dict):
                        if "url" in item:
                            images.append(item["url"])
                        elif "b64_json" in item:
                            images.append(item["b64_json"])
            return images
        except Exception as e:
            self.logger.error(f"Error parsing Image Gen response: {e}")
            return None

    def _parse_alibaba_video_gen(self, raw: Any) -> Optional[List[str]]:
        """
        Parse final JSON response from Alibaba Video Generation.
        Looks for URL in output['video_url'] OR output['results'][0]['url'].
        """
        videos = []
        try:
            if isinstance(raw, dict) and raw.get("output"):
                output = raw["output"]
                
                # Primary path: Direct video_url
                if output.get("task_status") == "SUCCEEDED" and output.get("video_url"):
                    videos.append(output["video_url"])
                
                # Alternative path: Through 'results'
                if output.get("task_status") == "SUCCEEDED" and output.get("results"):
                    for item in output["results"]:
                        if isinstance(item, dict) and "url" in item:
                            videos.append(item["url"])
            
            return list(set(videos))  # Return unique URLs only
        
        except Exception as e:
            self.logger.error(f"Error parsing Alibaba Video response: {e}")
            return None

    def _parse_anthropic(self, raw: Any) -> Optional[str]:
        """Parse Anthropic Claude API response."""
        try:
            if isinstance(raw, dict) and raw.get("content"):
                if isinstance(raw["content"], list) and len(raw["content"]) > 0:
                    first_block = raw["content"][0]
                    if isinstance(first_block, dict) and first_block.get("type") == "text":
                        return first_block.get("text")
            return None
        except (AttributeError, IndexError, TypeError, KeyError):
            self.logger.warning(f"Unable to parse Anthropic response: {raw}")
            return None

    # ====================================================================
    # UTILITY METHODS
    # ====================================================================

    def _ensure_converted(self, messages: List[Union[dict, Any]]) -> List[dict]:
        """
        Robustly convert any input type to standard list of message dicts.
        Handles: str, dict, LangChain BaseMessage, Smolagents ChatMessage, generic objects.
        Also translates exotic roles (tool-call) for strict APIs.
        
        Args:
            messages: List of messages in various formats
            
        Returns:
            Normalized list of message dictionaries
        """
        if not messages:
            return []
            
        converted_messages = []
        
        for msg in messages:
            try:
                final_role = "user"
                final_content = ""

                # Case 1: Already a complete dictionary
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    final_role = msg["role"]
                    final_content = msg["content"]
                
                # Case 2: Simple string
                elif isinstance(msg, str):
                    final_role = "user"
                    final_content = msg

                # Case 3: LangChain BaseMessage
                elif type(msg).__name__ in ["HumanMessage", "AIMessage", "SystemMessage", "BaseMessage"]:
                    role_map = {"human": "user", "ai": "assistant", "system": "system"}
                    msg_type = getattr(msg, "type", "user")
                    final_role = role_map.get(msg_type, msg_type)
                    final_content = getattr(msg, "content", "")

                # Case 4: Duck typing (Smolagents / ChatMessage)
                elif hasattr(msg, "role") and hasattr(msg, "content"):
                    final_role = msg.role
                    final_content = str(msg.content)

                # Case 5: Fallback
                elif hasattr(msg, "content"):
                    final_content = str(msg.content)
                else:
                    final_content = str(msg)

                # Translate exotic roles for strict APIs
                if final_role in ["tool-call", "tool-use"]:
                    final_role = "assistant"
                elif final_role in ["tool-response", "tool-result", "tool"]:
                    final_role = "user"
                    if not str(final_content).startswith("Tool output"):
                        final_content = f"Tool output: {final_content}"

                converted_messages.append({"role": final_role, "content": final_content})
            
            except Exception as e:
                print(f"⚠️ Warning: Message conversion failed: {e}")
                converted_messages.append({"role": "user", "content": str(msg)})

        return converted_messages

    @staticmethod
    def _sanitize_messages_for_strict_apis(messages: List[dict]) -> List[dict]:
        """
        Clean history for strict APIs (Moonshot, Alibaba, etc.) that reject
        two consecutive messages with the same role. Merges consecutive same-role messages.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Sanitized list with no consecutive same-role messages
        """
        if not messages:
            return []
            
        sanitized = []
        for msg in messages:
            if not sanitized:
                sanitized.append(msg)
                continue
            
            last_msg = sanitized[-1]
            
            # If same role as previous -> MERGE
            if msg['role'] == last_msg['role']:
                new_content = str(last_msg['content']) + "\n\n" + str(msg['content'])
                last_msg['content'] = new_content
            else:
                sanitized.append(msg)
                
        return sanitized

    def _mask_sensitive_data(self, data: Any) -> Any:
        """
        Recursively mask API keys and sensitive data in payloads for secure logging.
        
        Args:
            data: Data structure to mask (dict, list, or primitive)
            
        Returns:
            Masked copy of the data
        """
        if isinstance(data, dict):
            masked = {}
            for k, v in data.items():
                if any(sensitive in k.lower() for sensitive in ["key", "token", "secret", "password"]):
                    masked[k] = "***MASKED***"
                else:
                    masked[k] = self._mask_sensitive_data(v)
            return masked
        elif isinstance(data, list):
            return [self._mask_sensitive_data(item) for item in data]
        else:
            return data

    def _mask_api_key_in_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """
        Mask API keys in headers for safe debug output.
        
        Args:
            headers: HTTP headers dictionary
            
        Returns:
            Headers with masked sensitive values
        """
        safe_headers = {}
        for k, v in headers.items():
            if any(sensitive in k.lower() for sensitive in ["authorization", "api-key", "x-api-key"]):
                # Show last 6 characters only
                safe_headers[k] = f"...{v[-6:]}" if len(v) > 6 else "***"
            else:
                safe_headers[k] = v
        return safe_headers

    # ====================================================================
    # STATIC UTILITY METHODS
    # ====================================================================

    @staticmethod
    def save_base64_image(b64_string: str, output_path: str) -> bool:
        """
        Save a base64-encoded image to file.
        
        Args:
            b64_string: Base64-encoded image string
            output_path: Path where to save the image
            
        Returns:
            True if successful, False otherwise
        """
        if not PILImage:
            print("❌ Error: PIL (Pillow) library required to save images.")
            return False
        try:
            if ',' in b64_string:
                b64_string = b64_string.split(',', 1)[-1]
                
            img_data = base64.b64decode(b64_string)
            img = PILImage.open(BytesIO(img_data))
            img.save(output_path)
            print(f"✅ Image saved to: {output_path}")
            return True
        except Exception as e:
            print(f"❌ Error saving base64 image: {e}")
            return False

    @staticmethod
    def download_image_from_url(url: str, output_path: str) -> bool:
        """
        Download an image from URL and save to file.
        
        Args:
            url: Image URL
            output_path: Path where to save the image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"✅ Image downloaded to: {output_path}")
            return True
        except Exception as e:
            print(f"❌ Error downloading image: {e}")
            return False


# ====================================================================
# SMOLAGENTS WRAPPER
# ====================================================================
# This code allows using this connector directly as a "Model"
# in the smolagents library (CodeAgent, ToolCallingAgent, etc.)

try:
    from smolagents.models import Model, ChatMessage
    
    class IdealSmolagentsWrapper(Model):
        """
        Integrated and robust wrapper to connect IdealUniversalLLMConnector to smolagents.
        Includes the generate() method and required token counters.
        """
        def __init__(self, connector: 'IdealUniversalLLMConnector', provider: str, model_id: str):
            """
            Initialize the wrapper.
            
            Args:
                connector: IdealUniversalLLMConnector instance
                provider: Provider name
                model_id: Model identifier
            """
            super().__init__()
            self.connector = connector
            self.provider = provider
            self.model_id = model_id

        def __call__(
            self,
            messages: List[Dict[str, str]],
            stop_sequences: Optional[List[str]] = None,
            grammar: Optional[str] = None,
            **kwargs
        ) -> ChatMessage:
            """Standard call method required by smolagents."""
            return self.chat(messages, stop_sequences=stop_sequences, grammar=grammar, **kwargs)

        def chat(
            self, 
            messages: List[Dict[str, str]], 
            stop_sequences: Optional[List[str]] = None, 
            grammar: Optional[str] = None, 
            **kwargs
        ) -> ChatMessage:
            """
            Chat logic (corrected and complete version).
            
            Args:
                messages: List of message dictionaries
                stop_sequences: Optional stop sequences (not currently used)
                grammar: Optional grammar specification (not currently used)
                **kwargs: Additional parameters
                
            Returns:
                ChatMessage object with the response
            """
            try:
                response = self.connector.invoke(
                    provider=self.provider,
                    model_id=self.model_id,
                    messages=messages,
                    **kwargs
                )
                
                text = response.get("text")
                if not text:
                    text = str(response.get("raw", ""))

                # Enhanced cleaning with ast.literal_eval parsing
                import ast
                
                text = text.strip()
                
                # Detect complex list format: "[{'type': 'text', ...}]"
                if text.startswith("[") and ("type" in text or "content" in text or "text" in text):
                    try:
                        parsed = ast.literal_eval(text)
                        
                        if isinstance(parsed, list) and len(parsed) > 0:
                            extracted_text = ""
                            for item in parsed:
                                if isinstance(item, dict):
                                    if item.get("type") == "text":
                                        extracted_text += item.get("text", "")
                                    elif "content" in item:
                                        extracted_text += item.get("content", "")
                            
                            if extracted_text:
                                text = extracted_text
                                
                    except Exception:
                        # Safety net: If parsing fails, keep text as-is
                        pass

                # Clean thinking tags (R1/DeepSeek)
                if "<think>" in text and "</think>" in text:
                    text = text.split("</think>")[-1].strip()

                return ChatMessage(
                    role="assistant",
                    content=str(text),
                    tool_calls=None,
                    raw=response
                )
            except Exception as e:
                return ChatMessage(role="assistant", content=f"❌ Connector Error: {e}")

        def generate(
            self, 
            input_messages: List[Dict[str, str]], 
            stop_sequences: Optional[List[str]] = None, 
            **kwargs
        ) -> ChatMessage:
            """
            Method required by recent smolagents versions.
            
            Args:
                input_messages: List of message dictionaries
                stop_sequences: Optional stop sequences
                **kwargs: Additional parameters
                
            Returns:
                ChatMessage object with the response
            """
            return self.chat(input_messages, stop_sequences=stop_sequences, **kwargs)

        @property
        def last_input_token_count(self) -> int:
            """Return input token count (placeholder for stats)."""
            return 0
            
        @property
        def last_output_token_count(self) -> int:
            """Return output token count (placeholder for stats)."""
            return 0

except ImportError:
    # Fallback to prevent crashes if smolagents is not installed
    class IdealSmolagentsWrapper:
        """Placeholder wrapper when smolagents is not installed."""
        def __init__(self, *args, **kwargs):
            raise ImportError("Please install 'smolagents' to use this wrapper: pip install smolagents")