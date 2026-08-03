# app/model_server/backend.py
#
# 로컬 HuggingFace 모델(transformers) 로딩/생성 로직.
# 프로세스 시작 시 두 모델(7B VL + 1.5B 텍스트)을 한 번만 로드해 전역 싱글턴으로
# 유지한다 — app.model_server.server 가 이 모듈을 통해서만 추론을 호출하므로,
# Supervisor/Knowledge/Execution/Perception 4개 A2A 프로세스가 이 서버 하나에
# HTTP로 붙어도 GPU에는 모델이 각 1벌만 올라간다(Ollama가 하던 역할과 동일 토폴로지).

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

from app.core.config import QWEN_TEXT_MODEL_NAME, QWEN_VL_MODEL_NAME

logger = logging.getLogger(__name__)


def _resolve_dtype():
    import torch

    return torch.float16 if torch.cuda.is_available() else torch.float32


@dataclass
class LoadedTextModel:
    """Qwen2.5-1.5B-Instruct — 텍스트 전용, tool-calling 지원."""

    tokenizer: Any
    model: Any

    def _build_inputs(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict]]):
        text = self.tokenizer.apply_chat_template(
            messages, tools=tools or None, tokenize=False, add_generation_prompt=True,
        )
        return self.tokenizer([text], return_tensors="pt").to(self.model.device)

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        import torch

        inputs = self._build_inputs(messages, tools)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else None,
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def stream_generate(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Iterator[str]:
        from transformers import TextIteratorStreamer

        inputs = self._build_inputs(messages, tools=None)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else None,
            streamer=streamer,
        )
        thread = threading.Thread(target=self.model.generate, kwargs=kwargs)
        thread.start()
        for token_text in streamer:
            if token_text:
                yield token_text
        thread.join()


@dataclass
class LoadedVLModel:
    """Qwen2-VL-7B-Instruct(bitsandbytes 4bit) — 텍스트/멀티모달 겸용."""

    processor: Any
    model: Any

    def _build_inputs(self, messages: List[Dict[str, Any]]):
        from qwen_vl_utils import process_vision_info

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        return self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

    def generate(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        import torch

        inputs = self._build_inputs(messages)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else None,
            )
        generated = output_ids[:, inputs["input_ids"].shape[-1]:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    def stream_generate(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Iterator[str]:
        from transformers import TextIteratorStreamer

        inputs = self._build_inputs(messages)
        streamer = TextIteratorStreamer(
            self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else None,
            streamer=streamer,
        )
        thread = threading.Thread(target=self.model.generate, kwargs=kwargs)
        thread.start()
        for token_text in streamer:
            if token_text:
                yield token_text
        thread.join()


_text_model: Optional[LoadedTextModel] = None
_vl_model: Optional[LoadedVLModel] = None


def load_models() -> None:
    """서버 기동 시 한 번 호출된다. 이미 로드돼 있으면 아무것도 하지 않는다(재시작 안전)."""
    global _text_model, _vl_model

    if _text_model is None:
        logger.info("텍스트 모델 로딩 시작: %s", QWEN_TEXT_MODEL_NAME)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(QWEN_TEXT_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_TEXT_MODEL_NAME, device_map="auto", torch_dtype=_resolve_dtype(),
        )
        _text_model = LoadedTextModel(tokenizer=tokenizer, model=model)
        logger.info("텍스트 모델 로딩 완료: %s", QWEN_TEXT_MODEL_NAME)

    if _vl_model is None:
        logger.info("VL 모델 로딩 시작: %s", QWEN_VL_MODEL_NAME)
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(QWEN_VL_MODEL_NAME)
        # bitsandbytes 4bit — 로딩 시점에 즉석 양자화한다. GPTQ(Marlin 커널 JIT 컴파일 필요)
        # 대신 쓰는 이유: Marlin은 Ampere+ 세대에 최적화돼 있어 Colab 무료 T4(Turing)에서
        # 컴파일 후 로딩이 멈추는 문제가 재현됐다. bnb는 커널 컴파일이 없어 이 문제가 없다.
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=_resolve_dtype(),
            bnb_4bit_quant_type="nf4",
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            QWEN_VL_MODEL_NAME, device_map="auto", quantization_config=quantization_config,
        )
        _vl_model = LoadedVLModel(processor=processor, model=model)
        logger.info("VL 모델 로딩 완료: %s", QWEN_VL_MODEL_NAME)


def get_text_model() -> LoadedTextModel:
    if _text_model is None:
        raise RuntimeError("텍스트 모델이 로드되지 않았습니다 — load_models()가 먼저 호출돼야 합니다.")
    return _text_model


def get_vl_model() -> LoadedVLModel:
    if _vl_model is None:
        raise RuntimeError("VL 모델이 로드되지 않았습니다 — load_models()가 먼저 호출돼야 합니다.")
    return _vl_model


def is_ready() -> bool:
    return _text_model is not None and _vl_model is not None
