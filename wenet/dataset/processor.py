# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
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
import json
import random
import re
import tarfile
from subprocess import PIPE, Popen
from urllib.parse import urlparse
#import librosa

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torch.nn.utils.rnn import pad_sequence

from wenet.baichuan.configuration_baichuan import BaiChuanConfig
from wenet.cllama2.configuration_cllama2 import LlamaConfig
from transformers import AutoTokenizer
from wenet.qwen.tokenization_qwen import QWenTokenizer
from wenet.qwen.configuration_qwen import QWenConfig
qwen_conf = QWenConfig()
AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])


def url_opener(data):
    """ Give url or local file, return file descriptor
        Inplace operation.

        Args:
            data(Iterable[str]): url or local file list

        Returns:
            Iterable[{src, stream}]
    """
    for sample in data:
        assert 'src' in sample
        # TODO(Binbin Zhang): support HTTP
        url = sample['src']
        try:
            pr = urlparse(url)
            # local file
            if pr.scheme == '' or pr.scheme == 'file':
                stream = open(url, 'rb')
            # network file, such as HTTP(HDFS/OSS/S3)/HTTPS/SCP
            else:
                cmd = f'wget -q -O - {url}'
                process = Popen(cmd, shell=True, stdout=PIPE)
                sample.update(process=process)
                stream = process.stdout
            sample.update(stream=stream)
            yield sample
        except Exception as ex:
            logging.warning('Failed to open {}'.format(url))


def tar_file_and_group(data):
    """ Expand a stream of open tar files into a stream of tar file contents.
        And groups the file with same prefix

        Args:
            data: Iterable[{src, stream}]

        Returns:
            Iterable[{key, wav, txt, sample_rate}]
    """
    for sample in data:
        assert 'stream' in sample
        stream = tarfile.open(fileobj=sample['stream'], mode="r|*")
        prev_prefix = None
        example = {}
        valid = True
        for tarinfo in stream:
            name = tarinfo.name
            pos = name.rfind('.')
            assert pos > 0
            prefix, postfix = name[:pos], name[pos + 1:]
            if prev_prefix is not None and prefix != prev_prefix:
                example['key'] = prev_prefix
                if valid:
                    yield example
                example = {}
                valid = True
            with stream.extractfile(tarinfo) as file_obj:
                try:
                    if postfix == 'txt':
                        example['txt'] = file_obj.read().decode('utf8').strip() #.replace(' ', '')
                    elif postfix in AUDIO_FORMAT_SETS:
                        waveform, sample_rate = torchaudio.load(file_obj)
                        example['wav'] = waveform
                        example['sample_rate'] = sample_rate
                    else:
                        example[postfix] = file_obj.read()
                except Exception as ex:
                    valid = False
                    logging.warning('error to parse {}'.format(name))
            prev_prefix = prefix
        if prev_prefix is not None:
            example['key'] = prev_prefix
            yield example
        stream.close()
        if 'process' in sample:
            sample['process'].communicate()
        sample['stream'].close()


def parse_raw(data):
    """ Parse key/wav/txt from json line

        Args:
            data: Iterable[str], str is a json line has key/wav/txt

        Returns:
            Iterable[{key, wav, txt, sample_rate}]
    """
    for sample in data:
        assert 'src' in sample
        json_line = sample['src']
        obj = json.loads(json_line)
        assert 'key' in obj
        assert 'txt' in obj
        key = obj['key']
        txt = obj['txt']
        try:
            example = dict(key=key,
                           txt=txt)
            yield example
        except Exception as ex:
            logging.warning('Failed to read {}'.format(wav_file))


def filter(data,
           max_length=10240,
           min_length=10,
           token_max_length=200,
           token_min_length=1,
           min_output_input_ratio=0.0005,
           max_output_input_ratio=1):
    """ Filter sample according to feature and label length
        Inplace operation.

        Args::
            data: Iterable[{key, wav, label, sample_rate}]
            max_length: drop utterance which is greater than max_length(10ms)
            min_length: drop utterance which is less than min_length(10ms)
            token_max_length: drop utterance which is greater than
                token_max_length, especially when use char unit for
                english modeling
            token_min_length: drop utterance which is
                less than token_max_length
            min_output_input_ratio: minimal ration of
                token_length / feats_length(10ms)
            max_output_input_ratio: maximum ration of
                token_length / feats_length(10ms)

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'label' in sample
        # sample['wav'] is torch.Tensor, we have 100 frames every second
        if len(sample['label']) < token_min_length:
            continue
        if len(sample['label']) > token_max_length:
            continue
        yield sample


def resample(data, resample_rate=16000):
    """ Resample data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            resample_rate: target resample rate

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        if sample_rate != resample_rate:
            sample['sample_rate'] = resample_rate
            sample['wav'] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=resample_rate)(waveform)
        yield sample


def speed_perturb(data, speeds=None):
    """ Apply speed perturb to the data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            speeds(List[float]): optional speed

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    if speeds is None:
        speeds = [0.9, 1.0, 1.1]
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        speed = random.choice(speeds)
        if speed != 1.0:
            wav, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, sample_rate,
                [['speed', str(speed)], ['rate', str(sample_rate)]])
            sample['wav'] = wav

        yield sample


def volume_perturb(data, volume=None):
    """ Apply volume perturb to the data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            volume(List[float]): optional volume

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    if volume is None:
        volume = [316, 3160]
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        gain = random.randint(volume[0], volume[1]) / 1000.
        if volume != 1.0:
            sample['wav'] *= gain

        yield sample


def compute_fbank(data,
                  num_mel_bins=23,
                  frame_length=25,
                  frame_shift=10,
                  dither=0.0):
    """ Extract fbank

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'key' in sample
        assert 'label' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        waveform = waveform * (1 << 15)
        # Only keep key, feat, label
        mat = kaldi.fbank(waveform,
                          num_mel_bins=num_mel_bins,
                          frame_length=frame_length,
                          frame_shift=frame_shift,
                          dither=dither,
                          energy_floor=0.0,
                          sample_frequency=sample_rate)
        yield dict(key=sample['key'], label=sample['label'], feat=mat)


def compute_mfcc(data,
                 num_mel_bins=23,
                 frame_length=25,
                 frame_shift=10,
                 dither=0.0,
                 num_ceps=40,
                 high_freq=0.0,
                 low_freq=20.0):
    """ Extract mfcc

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'key' in sample
        assert 'label' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        waveform = waveform * (1 << 15)
        # Only keep key, feat, label
        mat = kaldi.mfcc(waveform,
                         num_mel_bins=num_mel_bins,
                         frame_length=frame_length,
                         frame_shift=frame_shift,
                         dither=dither,
                         num_ceps=num_ceps,
                         high_freq=high_freq,
                         low_freq=low_freq,
                         sample_frequency=sample_rate)
        yield dict(key=sample['key'], label=sample['label'], feat=mat)


def __tokenize_by_bpe_model(sp, txt):
    tokens = []
    # CJK(China Japan Korea) unicode range is [U+4E00, U+9FFF], ref:
    # https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    pattern = re.compile(r'([\u4e00-\u9fff])')
    pattern_num = re.compile(r'\d+')
    # Example:
    #   txt   = "你好 ITS'S OKAY 的"
    #   chars = ["你", "好", " ITS'S OKAY ", "的"]
    chars = pattern.split(txt.upper())
    mix_chars = [w for w in chars if len(w.strip()) > 0]
    for ch_or_w in mix_chars:
        # ch_or_w is a single CJK charater(i.e., "你"), do nothing.
        if pattern_num.fullmatch(ch_or_w) is not None:
            tokens.append(ch_or_w)
        # ch_or_w contains non-CJK charaters(i.e., " IT'S OKAY "),
        # encode ch_or_w using bpe_model.
        else:
            for p in sp.encode_as_pieces(ch_or_w):
                tokens.append(p)

    return tokens


def convert_to_sy(data, id2sy):
    for sample in data:
        try:
            key = sample['key']
            sample['txt'] = id2sy[key]
        except KeyError:
            continue
        yield sample

def shift_elements(lst, element, l):
    index_list = [idx for idx,t in enumerate(lst) if t == element]
    count = len(index_list)
    if l > 0:
        r = l #random.randint(1,l)
        for i in range(count-1,-1,-1):  
            index = index_list[i]
            new_index = index + r
            lst.insert(new_index, lst.pop(index)) 
    elif l < 0:
        for i in range(count):
            index = index_list[i]
            new_index = index + l
            lst.insert(new_index, lst.pop(index))
    else:
        return lst
    return lst 

def tokenize(data,
             symbol_table,
             bpe_model=None,
             non_lang_syms=None,
             split_with_space=False):
    """ Decode text to chars or BPE
        Inplace operation

        Args:
            data: Iterable[{key, wav, txt, sample_rate}]

        Returns:
            Iterable[{key, wav, txt, tokens, label, sample_rate}]
    """
    if non_lang_syms is not None:
        non_lang_syms_pattern = re.compile(r"(\[[^\[\]]+\]|<[^<>]+>|{[^{}]+})")
    else:
        non_lang_syms = {}
        non_lang_syms_pattern = None

    #tokenizer = QWenTokenizer(bpe_model)

    if bpe_model is not None:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        #sp.load(bpe_model)

    for sample in data:
        assert 'txt' in sample
        txt = sample['txt'].strip()
        if non_lang_syms_pattern is not None:
            parts = non_lang_syms_pattern.split(txt.upper())
            parts = [w for w in parts if len(w.strip()) > 0]
        else:
            parts = [txt]

        label_in = []
        label_out = []
        tokens = []
        token = []
        id_a = 0
        id_t = 1
        l_a = -1
        l_t = 0
        for part in parts:
            if part in non_lang_syms:
                tokens.append(part)
            else:
                part = part.split(" ")
                for ch in part:
                    tokens.append(ch)
        #shift =  random.randint(1, 10)
        #if shift < 4 and shift > -4:
        #    shift_elements(tokens, '<b>', shift)
        for i in range(len(tokens)):
            ch = tokens[i]
            ### get token
            if ch.isdigit():
                if i < len(tokens) + 1:# -1 and ch != tokens[i+1]:
                    label_in.append(int(ch)+1)
                    token.append(id_a)
                    l_a = l_a + 1
            ### get text
            elif ch == '<b>':
                token.append(id_a)
                id_a = id_a + 1
                label_in.append(2001)
                l_a = l_a + 1
            elif ch == '<s>':
                #label_in.append(0)
                #l_a = l_a + 1
                #token.append(id_a)
                l_t = l_t + 1
                token.append(id_t)
                id_t = id_t + 1
                label_out.append(4999)
            else:
                if ch in symbol_table:
                    l_t = l_t + 1
                    label_out.append(symbol_table[ch])
                    token.extend([id_t])
                    id_t = id_t + 1
                else:
                    l_t = l_t + 1
                    label_out.append(symbol_table['<unk>'])
                    token.extend([id_t])
                    id_t = id_t + 1
        label_out.append(4999)
        token.append(id_t)
        #assert len(label_in) == len(label_out), len(label_in) == len(token)
        sample['label_in'] = label_in
        sample['label_out'] = label_out
        sample['tokens'] = token
        sample['len_audio'] = l_a
        sample['len_text'] = l_t
        yield sample

def token_mask(data, symbal_table, p):
    """Do token mask inplace
        data: [{key,label_in,label_out}]
        p: prob of mask
    """
    for sample in data:
        assert 'label_in' in sample
        x = sample['label_in']
        #y = sample['label_out'].copy()
        label_mask = [] #x.copy()
        #num = random.randint(1, 5)
        for i in range(len(x)):
            num = random.randint(1, 10)
            num = min(num, len(x) - i -1)
            #if x[i] == 2001:
            #    for j in range(num):
            #        label_mask[i+j] = x[i+j+1]
            #    label_mask[i+num] = 2001
            #if y[i] != 4999 and num > 4:
            #    y[i] = 4232


            if num > 7 and x[i] != 2001:
                label_mask.append(2002)
            else:
                label_mask.append(x[i])
        sample['label_in'] = label_mask.copy()
        #sample['label_out'] = y
        yield sample



def spec_aug(data, num_t_mask=2, num_f_mask=2, max_t=50, max_f=10, max_w=80):
    """ Do spec augmentation
        Inplace operation

        Args:
            data: Iterable[{key, feat, label}]
            num_t_mask: number of time mask to apply
            num_f_mask: number of freq mask to apply
            max_t: max width of time mask
            max_f: max width of freq mask
            max_w: max width of time warp

        Returns
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'feat' in sample
        x = sample['feat']
        assert isinstance(x, torch.Tensor)
        y = x.clone().detach()
        max_frames = y.size(0)
        max_freq = y.size(1)
        # time mask
        for i in range(num_t_mask):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, max_t)
            end = min(max_frames, start + length)
            y[start:end, :] = 0
        # freq mask
        for i in range(num_f_mask):
            start = random.randint(0, max_freq - 1)
            length = random.randint(1, max_f)
            end = min(max_freq, start + length)
            y[:, start:end] = 0
        sample['feat'] = y
        yield sample


def spec_sub(data, max_t=20, num_t_sub=3):
    """ Do spec substitute
        Inplace operation

        Args:
            data: Iterable[{key, feat, label}]
            max_t: max width of time substitute
            num_t_sub: number of time substitute to apply

        Returns
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'feat' in sample
        x = sample['feat']
        assert isinstance(x, torch.Tensor)
        y = x.clone().detach()
        max_frames = y.size(0)
        for i in range(num_t_sub):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, max_t)
            end = min(max_frames, start + length)
            # only substitute the earlier time chosen randomly for current time
            pos = random.randint(0, start)
            y[start:end, :] = x[start - pos:end - pos, :]
        sample['feat'] = y
        yield sample


def spec_trim(data, max_t=20):
    """ Trim tailing frames. Inplace operation.
        ref: TrimTail [https://arxiv.org/abs/2211.00522]

        Args:
            data: Iterable[{key, feat, label}]
            max_t: max width of length trimming

        Returns
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'feat' in sample
        x = sample['feat']
        assert isinstance(x, torch.Tensor)
        max_frames = x.size(0)
        length = random.randint(1, max_t)
        if length < max_frames / 2:
            y = x.clone().detach()[:max_frames - length]
            sample['feat'] = y
        yield sample


def shuffle(data, shuffle_size=10000):
    """ Local shuffle the data

        Args:
            data: Iterable[{key, feat, label}]
            shuffle_size: buffer size for shuffle

        Returns:
            Iterable[{key, feat, label}]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    random.shuffle(buf)
    for x in buf:
        yield x


def sort(data, sort_size=500):
    """ Sort the data by feature length.
        Sort is used after shuffle and before batch, so we can group
        utts with similar lengths into a batch, and `sort_size` should
        be less than `shuffle_size`

        Args:
            data: Iterable[{key, feat, label}]
            sort_size: buffer size for sort

        Returns:
            Iterable[{key, feat, label}]
    """

    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= sort_size:
            buf.sort(key=lambda x: len(x['label_in']))
            for x in buf:
                yield x
            buf = []
    # The sample left over
    buf.sort(key=lambda x: len(x['label_in']))
    for x in buf:
        yield x


def static_batch(data, batch_size=16):
    """ Static batch the data by `batch_size`

        Args:
            data: Iterable[{key, feat, label}]
            batch_size: batch size

        Returns:
            Iterable[List[{key, feat, label}]]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


def dynamic_batch(data, max_frames_in_batch=12000):
    """ Dynamic batch the data until the total frames in batch
        reach `max_frames_in_batch`

        Args:
            data: Iterable[{key, feat, label}]
            max_frames_in_batch: max_frames in one batch

        Returns:
            Iterable[List[{key, feat, label}]]
    """
    buf = []
    longest_frames = 0
    for sample in data:
        new_sample_frames = len(sample['label_in'])
        longest_frames = max(longest_frames, new_sample_frames)
        frames_after_padding = longest_frames * (len(buf) + 1)
        if frames_after_padding > max_frames_in_batch:
            yield buf
            buf = [sample]
            longest_frames = new_sample_frames
        else:
            buf.append(sample)
    if len(buf) > 0:
        yield buf


def batch(data, batch_type='static', batch_size=16, max_frames_in_batch=12000):
    """ Wrapper for static/dynamic batch
    """
    if batch_type == 'static':
        return static_batch(data, batch_size)
    elif batch_type == 'dynamic':
        return dynamic_batch(data, max_frames_in_batch)
    else:
        logging.fatal('Unsupported batch type {}'.format(batch_type))


def padding(data):
    """ Padding the data into training data

        Args:
            data: Iterable[List[{key, feat, label}]]

        Returns:
            Iterable[Tuple(keys, feats, labels, feats lengths, label lengths)]
    """
    for sample in data:
        assert isinstance(sample, list)
        label_lengths = torch.tensor([len(x['label_in']) + len(x['label_out']) for x in sample],
                                    dtype=torch.int32)
        order = torch.argsort(label_lengths, descending=True)
        sorted_keys = [sample[i]['key'] for i in order]
        sorted_labels_in = [torch.tensor(sample[i]['label_in'], dtype=torch.int64) for i in order]
        sorted_labels_out = [torch.tensor(sample[i]['label_out'], dtype=torch.int64) for i in order]
        label_lengths_in = torch.tensor([sample[i]['len_audio']+1 for i in order], dtype=torch.int32)
        label_lengths_out = torch.tensor([sample[i]['len_text']+1 for i in order], dtype=torch.int32)
        att_mask = [torch.tensor(sample[i]['tokens'], dtype=torch.int64) for i in order]

        padding_labels_in = pad_sequence(sorted_labels_in,
                                      batch_first=True,
                                      padding_value=0)

        padding_labels_out = pad_sequence(sorted_labels_out,
                                      batch_first=True,
                                      padding_value=4999)
        
        padding_att_mask = pad_sequence(att_mask,
                                      batch_first=True,
                                      padding_value=9999)

        yield (sorted_keys, padding_labels_in, padding_labels_out, label_lengths_in, label_lengths_out, padding_att_mask)
