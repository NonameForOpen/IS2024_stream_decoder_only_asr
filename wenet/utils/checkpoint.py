# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang)
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

import logging
import os
import re

import yaml
import torch
from collections import OrderedDict

import datetime
import safetensors

def load_checkpoint(model: torch.nn.Module, path: str) -> dict:
    #if torch.cuda.is_available():
    #    logging.info('Checkpoint: loading from checkpoint %s for GPU' % path)
    #    checkpoint = torch.load(path)
    #else:
    #    logging.info('Checkpoint: loading from checkpoint %s for CPU' % path)
    #    checkpoint = torch.load(path, map_location='cpu')
    #model.load_state_dict(checkpoint, strict=True)
    part = {}
    ###load for qwen0.5B
    with safetensors.safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if "wte" in k or "lm_head" in k or "embed_tokens" in k:
                pass
            else:
                part[k] = f.get_tensor(k)

    model_dict = model.state_dict()
    model_dict.update(part)
    model.load_state_dict(model_dict)
    #info_path = re.sub('.pt$', '.yaml', path)
    configs = {}
    return configs

def load_checkpoint_for_test(model: torch.nn.Module, path: str) -> dict:
    if torch.cuda.is_available():
        checkpoint = torch.load(path, map_location='cpu')
    else:
        checkpoint = torch.load(path, map_location='cpu')
    logging.info('Checkpoint: loading from checkpoint %s for GPU' % path)
    part = {}
    model_dict = model.state_dict()
    for k,v in checkpoint.items():
        if "linear.weight" in k or "linear.bias" in k:
            pass
            #part[k] = v
        else:
            part[k] = v
    model_dict.update(part)
    model.load_state_dict(model_dict)
    #model.load_state_dict(checkpoint, strict=True)
    info_path = re.sub('.pt$', '.yaml', path)
    configs = {}
    return configs

def save_checkpoint(model: torch.nn.Module, path: str, infos=None):
    '''
    Args:
        infos (dict or None): any info you want to save.
    '''
    logging.info('Checkpoint: save to checkpoint %s' % path)
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, path)
    info_path = re.sub('.pt$', '.yaml', path)
    if infos is None:
        infos = {}
    infos['save_time'] = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    with open(info_path, 'w') as fout:
        data = yaml.dump(infos)
        fout.write(data)


def filter_modules(model_state_dict, modules):
    new_mods = []
    incorrect_mods = []
    mods_model = model_state_dict.keys()
    for mod in modules:
        if any(key.startswith(mod) for key in mods_model):
            new_mods += [mod]
        else:
            incorrect_mods += [mod]
    if incorrect_mods:
        logging.warning(
            "module(s) %s don't match or (partially match) "
            "available modules in model.",
            incorrect_mods,
        )
        logging.warning("for information, the existing modules in model are:")
        logging.warning("%s", mods_model)

    return new_mods


def load_trained_modules(model: torch.nn.Module, args: None):
    # Load encoder modules with pre-trained model(s).
    enc_model_path = args.enc_init
    enc_modules = args.enc_init_mods
    main_state_dict = model.state_dict()
    logging.warning("model(s) found for pre-initialization")
    if os.path.isfile(enc_model_path):
        logging.info('Checkpoint: loading from checkpoint %s for CPU' %
                     enc_model_path)
        model_state_dict = torch.load(enc_model_path, map_location='cpu')
        modules = filter_modules(model_state_dict, enc_modules)
        partial_state_dict = OrderedDict()
        for key, value in model_state_dict.items():
            if any(key.startswith(m) for m in modules):
                partial_state_dict[key] = value
        main_state_dict.update(partial_state_dict)
    else:
        logging.warning("model was not found : %s", enc_model_path)

    model.load_state_dict(main_state_dict)
    configs = {}
    return configs
