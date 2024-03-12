# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
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
# Modified from ESPnet(https://github.com/espnet/espnet)

from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import logging
import torch

from torch.nn.utils.rnn import pad_sequence
from torch.nn import CrossEntropyLoss
from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import TransformerDecoder
from wenet.transformer.encoder import TransformerEncoder
from wenet.transformer.label_smoothing_loss import LabelSmoothingLoss
from wenet.utils.common import (IGNORE_ID, add_sos_eos, log_add,
                                remove_duplicates_and_blank, th_accuracy,
                                reverse_pad_list)
from wenet.utils.mask import (make_pad_mask, mask_finished_preds,
                              mask_finished_scores, subsequent_mask)
from wenet.transformer.embedding import PositionalEncoding
from wenet.baichuan.modeling_baichuan import BaiChuanForCausalLM
from wenet.baichuan.configuration_baichuan import BaiChuanConfig
from wenet.qwen.tokenization_qwen import QWenTokenizer
import sentencepiece as spm
from wenet.utils.file_utils import read_symbol_table
symbol_table = read_symbol_table("./dict")
char_dict = {v: k for k, v in symbol_table.items()}
import random

class ASRModel(torch.nn.Module):
    """CTC-attention hybrid Encoder-Decoder model"""
    def __init__(
        self,
        vocab_size: int,
        encoder: Optional[TransformerEncoder],
        llm: Optional[BaiChuanForCausalLM],
        llm_conf: Optional[BaiChuanConfig],
        neg_weight: float = 0.0,
        ignore_id: int = 0,
        audio_mask_weight: float = 0.0,
        text_mask_weight: float = 0.0,
        length_normalized_loss: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.neg_weight = neg_weight
        self.audio_mask_weight = audio_mask_weight
        self.text_mask_weight = text_mask_weight

        self.llm = llm
        self.llm_conf = llm_conf
        self.emb_size = 2050
        ### only emb
        self.input_embedding = torch.nn.Embedding(self.emb_size, self.llm_conf.hidden_size)
        self.audio_head = torch.nn.Linear(self.llm_conf.hidden_size, self.emb_size, bias=False)
        self.vocab_size = self.llm.config.vocab_size
        self.smooth_factor = 1
        self.noise_threshold = 0.5
        self.trigger_id = 2001
        self.criterion_audio = LabelSmoothingLoss(
            size=self.emb_size,
            padding_idx=self.llm_conf.pad_token_id,
            smoothing=0.1,
            normalize_length=True,
        )
        self.criterion_text = LabelSmoothingLoss(
            size=self.llm.config.vocab_size,
            padding_idx=self.llm_conf.pad_token_id,
            smoothing=0.1,
            normalize_length=True,
        )


    def forward(
        self,
        inputs: torch.Tensor,
        inputs_lengths: torch.Tensor,
        target: torch.Tensor,
        target_lengths: torch.Tensor,
        att_mask: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            inputs: (Batch, L + T)
            inputs_lengths: (Batch, )
            target: (Batch, L + T)
            target_lengths: (Batch,)
        example:
            inputs: [0, 1, 2, 0, 3, 4, 0, 5, 6, 0]
            target: [eos, 0, 0, 1, 0, 0, 2, 0, 0, eos]
        
        """
        ### get emb mask
        B, L = inputs.shape
        device = inputs.device
        num_elements = int(self.audio_mask_weight * L)
        idx = torch.where(inputs == self.trigger_id)
        inputs_mask = torch.clone(inputs)
        for b in range(B):
            selected_indices = torch.randperm(L)[:num_elements]
            inputs_mask[b,selected_indices] = 2002
        inputs_mask[idx] = self.trigger_id
        ### inputs emb
        encoder_out = self.input_embedding(inputs_mask)
        encoder_mask = ~make_pad_mask(inputs_lengths)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        #concat
        inputs_embeds, labels, llm_out_masks, audio_out_masks = self.concat_data(
            encoder_out, encoder_out_lens, target, target_lengths,inputs)

        ### get mask 
        mask = ~make_pad_mask(inputs_lengths + target_lengths - 1)
        B, L = mask.shape
        mask_causal = subsequent_mask(L+1, device).repeat(B,1,1,1)
        llm_attention_mask = torch.ones(B, 1, L+1, L+1).to(torch.bool).to(device)
        pos_id1 = ~make_pad_mask(inputs_lengths + target_lengths)
        for b in range(B):
            pos_id1[b,:inputs_lengths[b]] = 0
            for i in range(target_lengths[b]):
                llm_attention_mask[b,0,inputs_lengths[b]+i,:inputs_lengths[b]+i+1] = torch.where(att_mask[b,:inputs_lengths[b]+i+1] > att_mask[b,inputs_lengths[b]+i], False, True)
            llm_attention_mask[b,0,inputs_lengths[b]+target_lengths[b]:,:] = False
        attention_mask = (llm_attention_mask & mask_causal).to(torch.int32)
        position_ids = pos_id1.long().cumsum(-1) - 1
        position_ids = torch.where(position_ids < 0, 0, position_ids)

        ### compute loss
        loss_llm, loss_audio, loss_text, acc = self._calc_att_loss(inputs_embeds, labels, llm_out_masks,  position_ids, audio_out_masks, attention_mask)
        loss = (loss_llm * 1 + loss_audio * 1 + loss_text * 1) 
        loss_log = {"loss": loss, "loss_llm": loss_llm, "loss_audio": loss_audio,"loss_text": loss_text, "acc": torch.tensor(acc)}
        return loss_log

    def concat_data(
        self,
        encoder_out,
        encoder_out_lens,
        text,
        text_lengths,inputs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        B, T, D = encoder_out.size()
        #import pdb;pdb.set_trace()
        text[text == 0] = self.llm_conf.pad_token_id
        text_mask = torch.clone(text)
        L = text_mask.size(1)
        num_elements = int(self.audio_mask_weight * L)
        for b in range(B):
            selected_indices = torch.randperm(L)[:num_elements]
            text_mask[b,selected_indices] = 4232
        text_mask = text_mask.to(torch.int)
        text_embed = self.llm.get_input_embeddings()(text_mask)

        llm_input = [torch.concat((encoder_out[b, :encoder_out_lens[b], :],
                         text_embed[b, :text_lengths[b], :]), dim=0)
                     for b in range(B)]

        llm_input = pad_sequence(llm_input, batch_first=True, padding_value=0)
        B, T, D = llm_input.size()
        device, dtype = llm_input.device, llm_input.dtype
        llm_out_mask = torch.zeros(B, T-1).to(torch.bool).to(device)
        audio_out_mask = torch.zeros(B, T-1).to(torch.bool).to(device)
        for b in range(B):
            llm_out_mask[b, encoder_out_lens[b]:encoder_out_lens[b]+text_lengths[b]-1] = True
            audio_out_mask[b, :encoder_out_lens[b]-1] = True

        pad = torch.ones(T).to(device) * self.llm_conf.pad_token_id
        labels = [torch.concat((inputs[b,:encoder_out_lens[b]],
                      text[b, :text_lengths[b]]), dim=0)
                  for b in range(B)]
        #import pdb;pdb.set_trace()
        labels = pad_sequence(labels, batch_first=True,
            padding_value=self.llm_conf.pad_token_id).to(torch.int64)
        return llm_input, labels, llm_out_mask, audio_out_mask


    def _calc_att_loss(
        self,
        inputs_emb: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        position_ids: torch.Tensor,
        audio_mask: torch.Tensor,
        att_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float]:
        B,T,D = inputs_emb.size()
        ###llm
        #import pdb;pdb.set_trace()
        if att_mask is not None:
            llm_out = self.llm(inputs_embeds=inputs_emb, labels=target,attention_mask=att_mask,position_ids=position_ids,output_hidden_states=True)
        else:
            llm_out = self.llm(inputs_embeds=inputs_emb, labels=target,position_ids=position_ids,output_hidden_states=True)
        lm_logits = llm_out.logits[..., :-1, :].contiguous()
        #a=torch.log_softmax(lm_logits.view(B,-1,5000),dim=-1)

        shift_labels = target[..., 1:].contiguous()
        loss_fct = self.criterion_text
        loss_llm = loss_fct(lm_logits,shift_labels).reshape(B,-1)
        loss_text = llm_out.loss.reshape(B,-1)
        loss_text = loss_text.masked_fill(~mask, 0.0)
        loss_text = (loss_text.sum(-1) / mask.sum(-1)).sum()/B
        loss_llm = loss_llm.masked_fill(~mask, 0.0)
        loss_llm = (loss_llm.sum(-1) / mask.sum(-1)).sum()/B

        ###audio
        hidden_states = llm_out.hidden_states[-1]
        audio_logits = self.audio_head(hidden_states)
        shift_logits = audio_logits[..., :-1, :].contiguous()
        audio_target = torch.where(target > self.trigger_id + 1,0,target)
        shift_labels = audio_target[..., 1:].contiguous()
        loss_fct = self.criterion_audio  #CrossEntropyLoss(reduction='none')
        loss_audio = loss_fct(shift_logits,shift_labels).reshape(B,-1)
        #loss_audio = loss_fct(
        #    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        #    ).reshape(B,-1)
        #loss_audio = llm_out.loss.reshape(B,-1)
        loss_audio = loss_audio.masked_fill(~audio_mask, 0.0)
        loss_audio = (loss_audio.sum(-1) / audio_mask.sum(-1)).sum()/B
        
        acc_target=target[...,1:]
        acc_target=acc_target.masked_fill(~mask, self.ignore_id)
        acc = th_accuracy(
            llm_out.logits[..., :-1, :].reshape(B * (T-1), -1),
            acc_target,
            ignore_label=self.ignore_id,
        )

        return loss_llm, loss_audio, loss_text, acc



    def recognize(
        self,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
    ) -> torch.Tensor:
        assert tokens.shape[0] == token_lengths.shape[0]
        device = tokens.device
        batch_size = tokens.shape[0]
        max_len = tokens.shape[2]
        tks: Optional[torch.Tensor] = None
        hyps = ''
        past_key_value_audio: Optional[torch.Tensor] = None
        past_key_value_audio_past: Optional[torch.Tensor] = None
        past_key_value_audio_last: Optional[torch.Tensor] = None
        past_key_value_text: Optional[torch.Tensor] = None
        past_key_value: Optional[torch.Tensor] = None
        last_token: Optional[torch.Tensor] = None
        idx_now = 0
        step_now = 0
        beam = 1
        flag = False
        out = []
        pos = -1
        tokens_a = []
        tokens_t = []
        num = 0
        la = 0
        lt = 0
        cnt = -1
        scores = 0
        for i in range(max_len):
            #inputs_emb = self.input_embedding(tokens[:,:,i])
            inputs_emb = self.llm.get_input_embeddings()(tokens[:,:,i])
            tokens_a.append(tokens[0,0,i].item())
            pos_id = torch.zeros(batch_size,1).to(torch.long).to(device)
            llm_out = self.llm(inputs_embeds=inputs_emb, past_key_values=past_key_value_audio,position_ids=pos_id,output_hidden_states=True)
            la = la + 1
            hidden_states = llm_out.hidden_states[-1]
            audio_logits = self.audio_head(hidden_states)
            logp = audio_logits[-1].reshape(batch_size,-1,self.emb_size)
            past_key_value_audio = llm_out.past_key_values
            topk_logp, topk_indexs = logp[:,-1,:].topk(k=beam)
            topk_indexs = topk_indexs.view(batch_size, beam)
            topk_index = topk_indexs[0,0]
            if i == max_len-1:
                topk_index = torch.tensor(self.trigger_id)
            if topk_index == self.trigger_id :
                flag = True
                char_emb = self.input_embedding(topk_index.to(device).unsqueeze(0).unsqueeze(0))
                #char_emb = self.llm.get_input_embeddings()(topk_index.to(device).unsqueeze(0).unsqueeze(0))
                tokens_a.append(topk_index.item())
                llm_out = self.llm(inputs_embeds=char_emb, past_key_values=past_key_value_audio,position_ids=pos_id)
                la = la + 1
                past_key_value_audio_past = past_key_value_audio_last
                past_key_value_audio_last = llm_out.past_key_values
                past_key_value_audio = llm_out.past_key_values
                if num == 0:
                    num = num + 1
                    pos = pos - 1
                    flag = False
                if past_key_value_text == None:
                    topk_index = torch.tensor(self.sos)
                    tokens_t.append(topk_index)
                else:
                    topk_index = ll
                    tokens_t.append(topk_index)
                pos = pos + 1
                p_len = 0
            while flag:
                pos_ids = pos_id + pos
                char_emb = self.llm.get_input_embeddings()(topk_index.to(device).unsqueeze(0).unsqueeze(0))
                past_key_value_audio = past_key_value_audio_last
                p_len = p_len + 1
                if past_key_value_text == None:
                    past_key_value = past_key_value_audio
                else:
                    past_key_value = tuple((torch.cat((past_key_value_audio[i][0],past_key_value_text[i][0]),2),torch.cat((past_key_value_audio[i][1],past_key_value_text[i][1]),2)) for i in range(self.llm_conf.num_hidden_layers))
                llm_out = self.llm(inputs_embeds=char_emb, past_key_values=past_key_value,position_ids=pos_ids)
                lt = lt + 1
                logp = llm_out.logits[-1].reshape(batch_size,-1,self.vocab_size)
                past_key_value_text = tuple((llm_out.past_key_values[i][0][:,:,-pos:,:],llm_out.past_key_values[i][1][:,:,-pos:,:]) for i in range(self.llm_conf.num_hidden_layers))
                logp = torch.log_softmax(logp.float(),dim=-1)
                topk_logp, topk_indexs = logp[:,-1,:].topk(k=beam)
                scores = scores + topk_logp[0,0]
                topk_indexs = topk_indexs.view(batch_size, beam)
                topk_index = topk_indexs[0,0]
                pos = pos + 1
                if p_len > 0:
                    ll = topk_index
                    if topk_index != self.sos:
                        hyps=topk_index.item()
                    topk_index = torch.tensor(self.sos)
                if topk_index == self.sos: #151647:
                    tokens_t.append(topk_index)
                    out.append(hyps)
                    pos = pos -1
                    flag = False
                    break
                else:
                    tokens_t.append(topk_index)
                    hyps=topk_index.item()
        return [out]

    def recognize_beam(
        self,
        tokens: torch.Tensor,
        token_lengths: torch.Tensor,
    ) -> torch.Tensor:
        assert tokens.shape[0] == token_lengths.shape[0]
        device = tokens.device
        batch_size = tokens.shape[0]
        beam_size = 5
        running_size = batch_size * beam_size
        max_len = tokens.shape[2]
        
        past_key_value: Optional[torch.Tensor] = None
        past_key_value_audio: Optional[torch.Tensor] = None
        past_key_value_audio_past: Optional[torch.Tensor] = None
        past_key_value_audio_last: Optional[torch.Tensor] = None
        past_key_value_text: Optional[torch.Tensor] = None
        last_token: Optional[torch.Tensor] = None
        
        hyps = torch.ones([running_size, 1], dtype=torch.long, device=device).fill_(4999) # (B*N, 1)
        prefix_len = hyps.size(1)
        scores = torch.tensor([0.0] + [-float('inf')] * (beam_size - 1), dtype=torch.float)
        scores = scores.to(device).repeat([batch_size]).unsqueeze(1).to(device)  # (B*N, 1)
        cache = [] 
        new_cache = []
        end_flag = torch.zeros_like(scores, dtype=torch.bool, device=device)
        
        flag = False
        pos = -1
        tokens_a = []
        tokens_t = []
        la = 0
        lt = 0
        num = 0
        final = []
        for i in range(max_len):
            inputs_emb = self.input_embedding(tokens[:,:,i])
            tokens_a.append(tokens[0,0,i].item())
            pos_id = torch.zeros(batch_size,1).to(torch.long).to(device)
            llm_out = self.llm(inputs_embeds=inputs_emb, past_key_values=past_key_value,position_ids=pos_id,output_hidden_states=True)
            la = la + 1
            hidden_states = llm_out.hidden_states[-1]
            audio_logits = self.audio_head(hidden_states)
            logp = audio_logits[-1].reshape(batch_size,-1,self.emb_size)
            past_key_value = llm_out.past_key_values
            topk_logp, topk_indexs = logp[:,-1,:].topk(k=1)
            topk_indexs = topk_indexs.view(batch_size, 1)
            topk_index = topk_indexs[0,0]
            if i == max_len - 1:
                topk_index = torch.tensor(self.trigger_id)
            if topk_index == self.trigger_id :
                flag = True
                char_emb = self.input_embedding(topk_index.to(device).unsqueeze(0).unsqueeze(0))
                tokens_a.append(topk_index.item())
                llm_out = self.llm(inputs_embeds=char_emb, past_key_values=past_key_value,position_ids=pos_id)
                la = la + 1
                past_key_value_audio_past = past_key_value_audio_last
                past_key_value_audio_last = llm_out.past_key_values
                past_key_value_audio = llm_out.past_key_values
                past_key_value = past_key_value_audio_last
                if num == 0:
                    num = num + 1
                    pos = pos -1
                    flag = False
                pos = pos + 1
            if flag:
                ### 1 forward
                pos_len = hyps.size(1)
                char_emb = self.llm.get_input_embeddings()(hyps[:,-1]).unsqueeze(1)
                pos_ids = pos_id + pos

                llm_out = torch.zeros([batch_size, beam_size, self.vocab_size], dtype=torch.float, device=device)
                for i in range(beam_size):
                    if len(cache) < beam_size:
                        past_key_value_beam = past_key_value
                    else:
                        past_key_value_beam= tuple((torch.cat((past_key_value[l][0],cache[i][l][0]),2),torch.cat((past_key_value[l][1],cache[i][l][1]),2)) for l in range(self.llm_conf.num_hidden_layers))
                    llm_outs = self.llm(inputs_embeds=char_emb[i,:,:].unsqueeze(0), past_key_values=past_key_value_beam,position_ids=pos_ids)
                    llm_out[:,i,:] = llm_outs.logits[:,-1,:].reshape(1,1,self.vocab_size)
                    new_cache.append(tuple((llm_outs.past_key_values[i][0][:,:,-pos:,:],llm_outs.past_key_values[i][1][:,:,-pos:,:]) for i in range(self.llm_conf.num_hidden_layers)))
                lt = lt + 1

                #import pdb;pdb.set_trace()
                ### 2 logp
                logp = llm_out.squeeze(0)
                logp = torch.log_softmax(logp.float(),dim=-1)
                top_k_logp, top_k_index = logp.topk(beam_size)  # (B*N, N)
                ### 3 score
                scores = scores + top_k_logp
                scores = scores.view(batch_size, beam_size * beam_size)  # (B, N*N)
                scores, offset_k_index = scores.topk(k=beam_size)  # (B, N)

                ### 4 update topk
                cache_index = (offset_k_index // beam_size).view(-1)  # (B*N)
                base_cache_index = (torch.arange(batch_size, device=device).view(-1, 1).repeat([1, beam_size]) * beam_size).view(-1)  # (B*N)
                cache_index = base_cache_index + cache_index
                cache = [new_cache[i] for i in cache_index]
                new_cache = []
                #cache = [torch.index_select(c, dim=0, index=cache_index) for c in cache]
                scores = scores.view(-1, 1)  # (B*N, 1)

                base_k_index = torch.arange(batch_size, device=device).view( -1, 1).repeat([1, beam_size])  # (B, N)
                base_k_index = base_k_index * beam_size * beam_size
                best_k_index = base_k_index.view(-1) + offset_k_index.view(-1)  # (B*N)

                ### 5 update hyps
                best_k_pred = torch.index_select(top_k_index.view(-1), dim=-1, index=best_k_index)  # (B*N)
                best_hyps_index = best_k_index // beam_size
                last_best_k_hyps = torch.index_select(hyps, dim=0, index=best_hyps_index)  # (B*N, i)
                hyps = torch.cat((last_best_k_hyps, best_k_pred.view(-1, 1)),dim=1)  # (B*N, i+1)
                tmp = hyps[:,1:].cpu().numpy().tolist()
                #for i in range(len(tmp)):
                #    content = []
                #    for w in tmp[i]:
                #        if w == 4999:
                #            break
                #        content.append(char_dict[w])
                #    logging.info('{} {}'.format(content,scores[i]))

                #print(hyps,scores)
        scores = scores.view(batch_size, beam_size)
        best_scores, best_index = scores.max(dim=-1)
        best_hyps_index = best_index + torch.arange(batch_size, dtype=torch.long, device=device) * beam_size
        best_hyps = torch.index_select(hyps, dim=0, index=best_hyps_index)
        best_hyps = best_hyps[:, prefix_len:]
        out = best_hyps.squeeze(0)
        for i in out:
            final.append(i.item())
        return [hyps[best_hyps_index.item(),1:].cpu().numpy().tolist()]
