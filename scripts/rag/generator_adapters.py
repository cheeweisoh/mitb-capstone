import asyncio
import os
from typing import Any, Protocol, Sequence

import torch
from llama_index.core.base.llms.types import (ChatMessage, ChatResponse,
                                               CompletionResponse,
                                               CompletionResponseGen,
                                               LLMMetadata, MessageRole)
from llama_index.core.llms.callbacks import (llm_chat_callback,
                                             llm_completion_callback)
from llama_index.core.llms.custom import CustomLLM
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openai import OpenAI
from transformers import (AutoModelForCausalLM, AutoModelForImageTextToText,
                          AutoProcessor, AutoTokenizer, BitsAndBytesConfig)

from rag_prompts import SYSTEM_PROMPT, VERIFY_SYSTEM_PROMPT


class GeneratorAdapter(Protocol):
    async def agenerate(self, system_prompt: str, user_prompt: str) -> str | None: ...


def _strip_after_markers(text: str, markers: list[str]) -> str:
    cleaned = text.strip()
    for marker in markers:
        if marker in cleaned:
            cleaned = cleaned.split(marker)[0].strip()
    cleaned = cleaned.split("\n\n")[0].strip()
    return cleaned


# ---------------------------------------------------------------------------
# gpt-4o-mini
# ---------------------------------------------------------------------------


class ChatGPTAdapter:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0, max_tokens: int = 384) -> None:
        self.llm = OpenAI(model=model, temperature=temperature, max_tokens=max_tokens)

    async def agenerate(self, system_prompt: str, user_prompt: str) -> str | None:
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=MessageRole.USER, content=user_prompt),
        ]
        response = await self.llm.achat(messages)
        content = response.message.content
        return content.strip() if content and content.strip() else None


def build_chatgpt_adapter() -> ChatGPTAdapter:
    return ChatGPTAdapter()


# ---------------------------------------------------------------------------
# Meditron / MedAlpaca (base/completion models, HuggingFaceLLM.acomplete())
# ---------------------------------------------------------------------------


class HFCompletionAdapter:
    """Wraps a HuggingFaceLLM configured for raw-string completion (no chat
    template). For the QA-generation call, system_prompt is intentionally
    ignored since each model's exact prompt format (including its system
    instructions) is baked into `completion_to_prompt` at construction time,
    matching the legacy per-model prompt strings exactly. For the verifier
    call (identified by VERIFY_SYSTEM_PROMPT), completion_to_prompt is bypassed
    and the prompt is primed with a literal '{' so the base model continues
    directly into a JSON object instead of prose -- these models were never
    trained to emit JSON on request, so priming the first token is far more
    reliable than asking nicely."""

    def __init__(self, llm: HuggingFaceLLM, postprocess) -> None:
        self.llm = llm
        self.postprocess = postprocess

    async def agenerate(self, system_prompt: str, user_prompt: str) -> str | None:
        if system_prompt == VERIFY_SYSTEM_PROMPT:
            prompt = f"System: {system_prompt}\n\n{user_prompt}\n\nJSON verdict:\n{{"
            response = await self.llm.acomplete(prompt, formatted=True)
            return "{" + response.text
        response = await self.llm.acomplete(user_prompt)
        return self.postprocess(response.text)


def build_meditron_adapter() -> HFCompletionAdapter:
    model_id = "epfl-llm/meditron-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = HuggingFaceLLM(
        model_name=model_id,
        tokenizer=tokenizer,
        model_kwargs={"dtype": torch.bfloat16},
        device_map="auto",
        max_new_tokens=256,
        generate_kwargs={
            "do_sample": False,
            "use_cache": True,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
        is_chat_model=False,
        completion_to_prompt=lambda prompt: f"System: {SYSTEM_PROMPT}\n\n{prompt}\n\nDoctor answer:",
    )

    def postprocess(text: str | None) -> str | None:
        if not text:
            return None
        cleaned = _strip_after_markers(text, ["### Answer", "Assistant:", "Question:", "Guideline excerpts:"])
        return cleaned or None

    return HFCompletionAdapter(llm, postprocess)


def build_medalpaca_adapter() -> HFCompletionAdapter:
    model_id = "medalpaca/medalpaca-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = HuggingFaceLLM(
        model_name=model_id,
        tokenizer=tokenizer,
        model_kwargs={"dtype": torch.bfloat16},
        device_map="auto",
        max_new_tokens=256,
        generate_kwargs={
            "do_sample": False,
            "use_cache": True,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
        is_chat_model=False,
        completion_to_prompt=lambda prompt: (
            "Below is a patient question and clinical guideline context. "
            "Provide a concise and specific answer in two to three sentences, with citations like [C1].\n\n"
            f"### Instruction:\n{prompt}\n\n### Response:\n"
        ),
    )

    def postprocess(text: str | None) -> str | None:
        if not text:
            return None
        cleaned = _strip_after_markers(text, ["### Instruction", "Question:", "Guideline excerpts:", "Below is"])
        return cleaned or None

    return HFCompletionAdapter(llm, postprocess)


# ---------------------------------------------------------------------------
# OpenBioLLM (chat-capable, 4-bit quantized, custom Llama3 chat template)
# ---------------------------------------------------------------------------

LLAMA3_CHAT_TEMPLATE = """{{ bos_token }}{% for message in messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>

{{ message['content'] | trim }}<|eot_id|>{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>

{% endif %}"""


class HFChatAdapter:
    """Wraps a HuggingFaceLLM configured with is_chat_model=True and an
    explicit messages_to_prompt hook, called via .achat()."""

    def __init__(self, llm: HuggingFaceLLM, postprocess) -> None:
        self.llm = llm
        self.postprocess = postprocess

    async def agenerate(self, system_prompt: str, user_prompt: str) -> str | None:
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=MessageRole.USER, content=user_prompt),
        ]
        response = await self.llm.achat(messages)
        return self.postprocess(response.message.content)


def _llama3_messages_to_prompt(tokenizer) -> Any:
    def messages_to_prompt(messages: Sequence[ChatMessage]) -> str:
        messages_dict = [{"role": message.role.value, "content": message.content} for message in messages]
        return tokenizer.apply_chat_template(messages_dict, add_generation_prompt=True, tokenize=False)

    return messages_to_prompt


def build_openbio_adapter() -> HFChatAdapter:
    model_id = "aaditya/Llama3-OpenBioLLM-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"

    llm = HuggingFaceLLM(
        model_name=model_id,
        tokenizer=tokenizer,
        model_kwargs={"quantization_config": bnb_config},
        device_map="auto",
        max_new_tokens=512,
        stopping_ids=terminators,
        generate_kwargs={
            "do_sample": False,
            "eos_token_id": terminators,
            "pad_token_id": tokenizer.eos_token_id,
        },
        is_chat_model=True,
        messages_to_prompt=_llama3_messages_to_prompt(tokenizer),
    )

    def postprocess(text: str | None) -> str | None:
        return text.strip() if text and text.strip() else None

    return HFChatAdapter(llm, postprocess)


# ---------------------------------------------------------------------------
# MedGemma (AutoModelForImageTextToText + AutoProcessor, not a standard
# CausalLM, so HuggingFaceLLM does not fit -- custom CustomLLM subclass).
# ---------------------------------------------------------------------------


class MedGemmaLLM(CustomLLM):
    context_window: int = 8192
    num_output: int = 384
    model_name: str = "google/medgemma-4b-it"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        # Left padding so every sequence in a batch ends at the same offset,
        # letting generated tokens be sliced out with a single input_len.
        self._processor.tokenizer.padding_side = "left"

    @classmethod
    def class_name(cls) -> str:
        return "MedGemmaLLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
            is_chat_model=True,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        raise NotImplementedError("MedGemmaLLM is chat-only; use chat()/achat().")

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError("MedGemmaLLM is chat-only; use chat()/achat().")

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        hf_messages = [
            {"role": message.role.value, "content": [{"type": "text", "text": message.content}]}
            for message in messages
        ]
        inputs = self._processor.apply_chat_template(
            hf_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device, dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            raw = self._model.generate(**inputs, max_new_tokens=self.num_output, do_sample=False)
            answer = self._processor.decode(raw[0][input_len:], skip_special_tokens=True).replace("<end_of_turn>", "").strip()

        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=answer))

    def chat_batch(self, batched_messages: Sequence[Sequence[ChatMessage]]) -> list[str]:
        hf_conversations = [
            [{"role": message.role.value, "content": [{"type": "text", "text": message.content}]} for message in messages]
            for messages in batched_messages
        ]
        inputs = self._processor.apply_chat_template(
            hf_conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(self._model.device, dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            raw = self._model.generate(**inputs, max_new_tokens=self.num_output, do_sample=False)
            return [
                self._processor.decode(raw[i][input_len:], skip_special_tokens=True).replace("<end_of_turn>", "").strip()
                for i in range(raw.shape[0])
            ]


class MedGemmaAdapter:
    def __init__(self) -> None:
        self.llm = MedGemmaLLM()

    async def agenerate(self, system_prompt: str, user_prompt: str) -> str | None:
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=MessageRole.USER, content=user_prompt),
        ]
        response = await self.llm.achat(messages)
        content = response.message.content
        return content.strip() if content and content.strip() else None

    async def agenerate_batch(self, system_prompt: str, user_prompts: list[str]) -> list[str | None]:
        batched_messages = [
            [ChatMessage(role=MessageRole.SYSTEM, content=system_prompt), ChatMessage(role=MessageRole.USER, content=up)]
            for up in user_prompts
        ]
        answers = await asyncio.to_thread(self.llm.chat_batch, batched_messages)
        return [answer.strip() if answer and answer.strip() else None for answer in answers]


def build_medgemma_adapter() -> MedGemmaAdapter:
    return MedGemmaAdapter()


# ---------------------------------------------------------------------------
# II-Medical (standard CausalLM, "thinking" block + thinking=False template kwarg)
# ---------------------------------------------------------------------------


def _iimedical_messages_to_prompt(tokenizer) -> Any:
    def messages_to_prompt(messages: Sequence[ChatMessage]) -> str:
        messages_dict = [
            {"role": message.role.value, "content": [{"type": "text", "text": message.content}]}
            for message in messages
        ]
        prompt = tokenizer.apply_chat_template(
            messages_dict,
            thinking=False,
            add_generation_prompt=True,
            tokenize=False,
            return_tensors=None,
        )
        return prompt + "<think>\n</think>\n"

    return messages_to_prompt


def build_iimedical_adapter() -> HFChatAdapter:
    model_id = "Intelligent-Internet/II-Medical-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    llm = HuggingFaceLLM(
        model_name=model_id,
        tokenizer=tokenizer,
        model_kwargs={"dtype": torch.bfloat16},
        device_map="auto",
        max_new_tokens=512,
        generate_kwargs={
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id,
        },
        is_chat_model=True,
        messages_to_prompt=_iimedical_messages_to_prompt(tokenizer),
    )

    def postprocess(text: str | None) -> str | None:
        if not text:
            return None
        cleaned = text.replace("<Answer>", "").replace("</Answer>", "").strip()
        return cleaned or None

    return HFChatAdapter(llm, postprocess)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ADAPTER_BUILDERS = {
    "chatgpt": build_chatgpt_adapter,
    "meditron": build_meditron_adapter,
    "medalpaca": build_medalpaca_adapter,
    "openbio": build_openbio_adapter,
    "medgemma": build_medgemma_adapter,
    "iimedical": build_iimedical_adapter,
}
