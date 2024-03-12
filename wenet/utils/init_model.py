# Copyright (c) 2022 Binbin Zhang (binbzha@qq.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from wenet.transducer.joint import TransducerJoint
from wenet.transducer.predictor import (ConvPredictor, EmbeddingPredictor,
                                        RNNPredictor)
from wenet.transducer.transducer import Transducer
from wenet.transformer.asr_model import ASRModel
from wenet.transformer.cmvn import GlobalCMVN
from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import BiTransformerDecoder, TransformerDecoder
from wenet.transformer.encoder import ConformerEncoder, TransformerEncoder
from wenet.squeezeformer.encoder import SqueezeformerEncoder
from wenet.efficient_conformer.encoder import EfficientConformerEncoder
from wenet.utils.cmvn import load_cmvn

from wenet.baichuan.configuration_baichuan import BaiChuanConfig, BaiChuanConfig_small
from wenet.baichuan.modeling_baichuan import BaiChuanForCausalLM

#from wenet.cllama2.modeling_cllama2 import LlamaModel, LlamaForCausalLM
from wenet.cllama2.modeling_cllama2 import LlamaModel, LlamaForCausalLM
from wenet.cllama2.configuration_cllama2 import LlamaConfig

from wenet.qwen.modeling_qwen import QWenLMHeadModel
from wenet.qwen.modeling_qwen2 import Qwen2ForCausalLM
from wenet.qwen.configuration_qwen import QWenConfig
from wenet.qwen.configuration_qwen2 import Qwen2Config

from transformers import AutoModelForCausalLM

def init_model(configs):
    input_dim = configs['input_dim']
    vocab_size = configs['output_dim']

    llm_type = configs.get('llm', 'baichuan')
    
    encoder = ConformerEncoder(512,
                         global_cmvn=None,
                         **configs['encoder_conf'])

    if llm_type == 'baichuan':
        llm_conf = BaiChuanConfig()
        llm = BaiChuanForCausalLM(llm_conf)
    elif llm_type == 'baichuan_small':
        llm_conf = BaiChuanConfig_small()
        llm = BaiChuanForCausalLM(llm_conf)
    elif llm_type == 'cllama2':
        llm_conf = LlamaConfig()
        llm = LlamaForCausalLM(llm_conf)
    elif llm_type == 'llama2':
        llm_conf = LlamaConfig()
        llm = LlamaForCausalLM(llm_conf)
    elif llm_type == 'qwen':
        llm_conf = Qwen2Config()
        llm = Qwen2ForCausalLM(llm_conf)
    else:
        llm = None
    model = ASRModel(vocab_size=vocab_size,
                     encoder=encoder,
                     llm=llm,
                     llm_conf=llm_conf,
                     **configs['model_conf'])
    return model
