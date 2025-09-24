import copy
import functools
import os
import sys
from contextlib import contextmanager

import pytest

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import (CudaGraphConfig, EagleDecodingConfig,
                                 KvCacheConfig, MTPDecodingConfig)
from tensorrt_llm.llmapi.tokenizer import TransformersTokenizer
from tensorrt_llm.sampling_params import SamplingParams

from ..conftest import llm_models_root
from .accuracy_core import GSM8K, MMLU, JsonModeEval, LlmapiAccuracyTestHarness


@contextmanager
def add_path(path):
    """
    Add a temp path to import util func.
    """
    sys.path.insert(0, path)
    try:
        yield
    finally:
        try:
            sys.path.remove(path)
        except ValueError:
            pass  # in case, path already been removed.


class TestFeatureCombination(LlmapiAccuracyTestHarness):
    PartialLLM = None
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct"
    ctx_server_config = {}
    gen_server_config = {}

    def test_overlap_scheduler(self):
        if self.PartialLLM == None:
            pytest.skip(
                "LLMs are not well-suited for feature combination testing.")

        with self.PartialLLM(
                model=self.MODEL_PATH,
                kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.5),
                disable_overlap_scheduler=False,
        ) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    def test_cuda_graph(self):
        if self.PartialLLM == None:
            pytest.skip(
                "LLMs are not well-suited for feature combination testing.")

        cuda_graph_config = CudaGraphConfig(batch_sizes=[4])
        with self.PartialLLM(
                model=self.MODEL_PATH,
                kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.5),
                cuda_graph_config=cuda_graph_config,
        ) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    def test_attention_dp(self):
        if self.PartialLLM == None:
            pytest.skip(
                "LLMs are not well-suited for feature combination testing.")

        model_name = "deepseek-ai/DeepSeek-V3-Lite"
        model_path = f"{llm_models_root()}/DeepSeek-V3-Lite/bf16"

        with self.PartialLLM(
                model=model_path,
                kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.5),
                enable_attention_dp=True,
        ) as llm:
            task = MMLU(model_name)
            task.evaluate(llm)

    def test_disaggregated_serving(self):
        if self.PartialLLM == None or not self.ctx_server_config or not self.gen_server_config:
            pytest.skip(
                "LLMs are not well-suited for feature combination testing.")

        from .test_disaggregated_serving import launch_disaggregated_llm
        ctx_server_config = copy.deepcopy(self.ctx_server_config)
        gen_server_config = copy.deepcopy(self.gen_server_config)
        ctx_server_config["cache_transceiver_config"] = {"backend": "DEFAULT"}
        gen_server_config["cache_transceiver_config"] = {"backend": "DEFAULT"}
        disaggregated_server_config = {
            "hostname": "localhost",
            "port": 8000,
            "backend": "pytorch",
            "context_servers": {
                "num_instances": 1,
                "urls": ["localhost:8001"]
            },
            "generation_servers": {
                "num_instances": 1,
                "urls": ["localhost:8002"]
            }
        }
        with launch_disaggregated_llm(disaggregated_server_config,
                                      ctx_server_config, gen_server_config,
                                      self.MODEL_PATH) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    def test_chunked_prefill(self):
        if self.PartialLLM == None:
            pytest.skip(
                "LLMs are not well-suited for feature combination testing.")

        with self.PartialLLM(
                model=self.MODEL_PATH,
                kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.5),
                enable_chunked_prefill=True,
        ) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    def test_mtp(self):
        if self.PartialLLM == None:
            pytest.skip(
                "LLMs are not well-suited for feature combination testing.")

        model_name = "deepseek-ai/DeepSeek-V3-Lite"
        model_path = f"{llm_models_root()}/DeepSeek-V3-Lite/bf16"
        mtp_nextn = 2
        mtp_config = MTPDecodingConfig(num_nextn_predict_layers=mtp_nextn)
        with self.PartialLLM(
                model=model_path,
                kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.75),
                speculative_config=mtp_config,
        ) as llm:
            task = GSM8K(model_name)
            task.evaluate(llm)

    def test_eagle3_one_model(self):
        if self.PartialLLM == None:
            pytest.skip(
                "LLMs are not well-suited for feature combination testing.")

        eagle_model_dir = f"{llm_models_root()}/EAGLE3-LLaMA3.1-Instruct-8B"
        draft_len = 4
        spec_config = EagleDecodingConfig(max_draft_len=draft_len,
                                          speculative_model_dir=eagle_model_dir,
                                          eagle3_one_model=True)
        with self.PartialLLM(
                model=self.MODEL_PATH,
                kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.5),
                speculative_config=spec_config,
        ) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    def test_eagle3_two_model(self):
        if self.PartialLLM == None:
            pytest.skip(
                "LLMs are not well-suited for feature combination testing.")

        eagle_model_dir = f"{llm_models_root()}/EAGLE3-LLaMA3.1-Instruct-8B"
        draft_len = 4
        spec_config = EagleDecodingConfig(max_draft_len=draft_len,
                                          speculative_model_dir=eagle_model_dir,
                                          eagle3_one_model=False)
        with self.PartialLLM(
                model=self.MODEL_PATH,
                kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.5),
                speculative_config=spec_config,
        ) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    def test_torch_sampler(self):
        if self.PartialLLM == None:
            pytest.skip(
                "LLMs are not well-suited for feature combination testing.")

        with self.PartialLLM(
                model=self.MODEL_PATH,
                kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.5),
                sampler_type="TorchSampler",
        ) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    def test_trtllm_sampler(self):
        if self.PartialLLM == None:
            pytest.skip(
                "LLMs are not well-suited for feature combination testing.")

        with self.PartialLLM(
                model=self.MODEL_PATH,
                kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.5),
                sampler_type="TRTLLMSampler",
        ) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    def test_kvcache_reuse(self):
        if self.PartialLLM == None:
            pytest.skip(
                "LLMs are not well-suited for feature combination testing.")

        with self.PartialLLM(
                model=self.MODEL_PATH,
                kv_cache_config=KvCacheConfig(enable_block_reuse=True,
                                              free_gpu_memory_fraction=0.5),
        ) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

    def test_slide_window_attention(self):
        if self.PartialLLM == None:
            pytest.skip(
                "LLMs are not well-suited for feature combination testing.")

        model_name = "google/gemma-3-1b-it"
        model_path = f"{llm_models_root()}/gemma/gemma-3-1b-it/"

        with self.PartialLLM(
                model=model_path,
                kv_cache_config=KvCacheConfig(enable_block_reuse=False,
                                              enable_partial_reuse=False,
                                              free_gpu_memory_fraction=0.5),
        ) as llm:
            task = MMLU(model_name)
            task.evaluate(llm)

    def test_logits_post_processor(self):
        if self.PartialLLM == None:
            pytest.skip(
                "LLMs are not well-suited for feature combination testing.")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        target_dir = os.path.join(current_dir, '../../../', 'unittest')

        with add_path(target_dir):
            from utils.util import MyLogitsProcessor, check_output

            model_path = f"{llm_models_root()}/llama-models-v2/TinyLlama-1.1B-Chat-v1.0"
            tokenizer = TransformersTokenizer.from_pretrained(model_path)
            biased_word_id = tokenizer.encode("Z", add_special_tokens=False)[-1]
            sampling_params = SamplingParams(
                max_tokens=6,
                logits_processor=MyLogitsProcessor(biased_word_id))
            prompts = ["A B C"]
            references = ["Z Z Z Z Z Z"]
            similar_threshold = 0.8
            with self.PartialLLM(
                    model=model_path,
                    tokenizer=model_path,
                    kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
            ) as llm:
                outputs = llm.generate(prompts, sampling_params=sampling_params)
                check_output(outputs,
                             references,
                             similar_threshold=similar_threshold)

    def test_guided_decoding(self, mocker):
        if self.PartialLLM == None:
            pytest.skip(
                "LLMs are not well-suited for feature combination testing.")
        mocker.patch.dict(os.environ, {"TRTLLM_XGUIDANCE_LENIENT": "1"})

        with self.PartialLLM(
                model=self.MODEL_PATH,
                guided_decoding_backend="xgrammar",
        ) as llm:
            task = JsonModeEval(self.MODEL_NAME)
            task.evaluate(llm)


class TestOverlapScheduler(TestFeatureCombination):
    PartialLLM = functools.partial(LLM, disable_overlap_scheduler=False)
    ctx_server_config = {"disable_overlap_scheduler": True}
    gen_server_config = {"disable_overlap_scheduler": False}
