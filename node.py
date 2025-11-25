from openai import OpenAI
import time
import json
import os


class RN_Translator_Node():
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_text": ("STRING", {"multiline": True, "default": "请输入要翻译的文本"}),
                "source_language": ("STRING", {"default": "自动检测"}),
                "target_language": ("STRING", {"default": "中文"}),
                "translation_style": (["标准", "正式", "口语化", "学术", "商务"], {"default": "标准"}),
                "temperature": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.1}),
            },
            "optional": {
                "input_text": ("STRING", {"forceInput": True}),
                "apiBaseUrl": ("STRING", {"default": "default"}),
                "apiKey": ("STRING", {"default": "default"}),
                "model": ("STRING", {"default": "default"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("翻译结果",)
    FUNCTION = "translate_text"
    CATEGORY = "RN翻译"

    def _translate_impl(self, source_text, source_language, target_language, translation_style, temperature,
                        input_text=None, apiBaseUrl=None, apiKey=None, model=None, system_prompt=None):
        env_api_baseurl = (
                os.environ.get("LLM_API_BASEURL")
                or os.environ.get("OPENAI_BASE_URL")
                or os.environ.get("OPENAI_API_BASE_URL")
                or os.environ.get("DEEPSEEK_API_BASE_URL")
        )
        env_api_key = (
                os.environ.get("LLM_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or os.environ.get("DEEPSEEK_API_KEY")
        )
        env_model = (
                os.environ.get("LLM_MODEL")
                or os.environ.get("OPENAI_MODEL")
                or os.environ.get("DEEPSEEK_MODEL")
        )

        used_api_baseurl = None
        used_api_key = None
        used_model = None

        used_api_baseurl = (apiBaseUrl or env_api_baseurl or "https://api.openai.com/v1")
        used_model = (model or env_model or "gpt-4o-mini")
        used_api_key = (apiKey or env_api_key or "")
        if not used_api_key:
            return ("错误：请提供API密钥",)

        final_text = (source_text or "")
        if input_text is not None:
            final_text = (str(input_text) + "\n" + final_text).strip()
        if not final_text.strip():
            return ("错误：请输入要翻译的文本",)

        try:
            client = OpenAI(api_key=used_api_key, base_url=used_api_baseurl)

            # 构建翻译提示词
            if system_prompt is None:
                system_prompt = "你是一个专业的翻译助手，能够准确地将文本从一种语言翻译成另一种语言。"

            # 根据翻译风格调整提示词
            style_prompts = {
                "标准": "请提供标准的翻译",
                "正式": "请使用正式、专业的语言进行翻译",
                "口语化": "请使用口语化、通俗易懂的语言进行翻译",
                "学术": "请使用学术性的语言进行翻译，保持专业性",
                "商务": "请使用商务场合适用的语言进行翻译"
            }

            translation_instruction = style_prompts.get(translation_style, "请提供标准的翻译")

            # 构建语言说明
            lang_instruction = f""
            if source_language != "自动检测":
                lang_instruction += f"从{source_language}"
            else:
                lang_instruction += "从源语言"

            lang_instruction += f"翻译成{target_language}"

            user_prompt = f"""
{translation_instruction}。
{lang_instruction}。

要翻译的文本：
{final_text}

请只返回翻译结果，不要添加任何解释或注释。"""

            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt.strip()}
            ]

            completion = client.chat.completions.create(
                model=used_model,
                messages=messages,
                temperature=temperature,
                max_tokens=4096
            )

            if completion is not None and hasattr(completion, 'choices') and len(completion.choices) > 0:
                translated_text = completion.choices[0].message.content.strip()
                return (translated_text,)
            else:
                return ("错误：API返回空结果",)

        except Exception as e:
            error_msg = f"翻译错误：{str(e)}"
            return (error_msg,)

    def translate_text(self, source_text, source_language, target_language, translation_style, temperature,
                       input_text=None, apiBaseUrl="", apiKey="", model=""):
        return self._translate_impl(source_text, source_language, target_language, translation_style,
                                    temperature, input_text=input_text,
                                    apiBaseUrl=apiBaseUrl, apiKey=apiKey, model=model)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float(time.time())

