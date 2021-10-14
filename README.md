## Description

This repo contains a pre-trained model using [icefall][icefall]
with the following command (it uses 7 GPUs):

```
cd egs/librispeech/ASR
./conformer_ctc/train.py \
  --world-size 7 \
  --max-duration 150 \
  --bucketing-sampler 1 \
  --full-libri 1 \
  --start-epoch 0 \
  --num-epochs 50 \
  --exp-dir conformer_ctc/exp \
  --lang-dir data/lang_bpe_500
```

The model `exp/pretrained.pt` is obtained by the following command:

```
cd egs/librispeech/ASR
./conformer_ctc/export.py \
  --epoch 49 \
  --avg 15 \
  --exp-dir conformer_ctc/exp \
  --lang-dir data/lang_bpe_500 \
  --jit 0
```

You can use `exp/pretrained.pt` to compute the WER for `test-clean` and `test-other`:

```
cd egs/librispeech/ASR
cd conformer_ctc/exp
ln -s pretrained.pt epoch-99.pt
cd ../..
./conformer_ctc/decode.py \
  --exp-dir ./conformer_ctc/exp \
  --lang-dir ./data/lang_bpe_500 \
  --epoch 99 \
  --avg 1 \
  --max-duration 30 \
  --nbest-scale 0.5
```

You will get the following log:

```
2021-10-14 20:01:39,246 INFO [decode.py:474] batch 0/804, cuts processed until now is 6
2021-10-14 20:03:19,789 INFO [decode.py:474] batch 100/804, cuts processed until now is 398
2021-10-14 20:04:53,371 INFO [decode.py:474] batch 200/804, cuts processed until now is 792
2021-10-14 20:06:22,611 INFO [decode.py:474] batch 300/804, cuts processed until now is 1160
2021-10-14 20:07:50,482 INFO [decode.py:474] batch 400/804, cuts processed until now is 1574
2021-10-14 20:09:20,129 INFO [decode.py:474] batch 500/804, cuts processed until now is 1965
2021-10-14 20:10:53,932 INFO [decode.py:474] batch 600/804, cuts processed until now is 2308
2021-10-14 20:12:36,852 INFO [decode.py:474] batch 700/804, cuts processed until now is 2498
2021-10-14 20:14:26,477 INFO [decode.py:474] batch 800/804, cuts processed until now is 2614
2021-10-14 20:16:16,519 INFO [decode.py:523]
For test-clean, WER of different settings are:
ngram_lm_scale_1.1_attention_scale_0.9  2.56    best for test-clean
ngram_lm_scale_1.1_attention_scale_1.0  2.56
ngram_lm_scale_1.2_attention_scale_1.0  2.56
ngram_lm_scale_0.9_attention_scale_0.6  2.57
ngram_lm_scale_0.9_attention_scale_0.7  2.57
ngram_lm_scale_1.0_attention_scale_0.6  2.57
ngram_lm_scale_1.0_attention_scale_0.9  2.57
ngram_lm_scale_1.1_attention_scale_0.7  2.57
ngram_lm_scale_1.1_attention_scale_1.1  2.57
ngram_lm_scale_1.2_attention_scale_0.7  2.57
ngram_lm_scale_1.2_attention_scale_0.9  2.57
ngram_lm_scale_1.2_attention_scale_1.1  2.57
ngram_lm_scale_1.2_attention_scale_1.2  2.57
ngram_lm_scale_1.3_attention_scale_0.9  2.57
ngram_lm_scale_1.3_attention_scale_1.1  2.57
ngram_lm_scale_1.5_attention_scale_1.0  2.57
ngram_lm_scale_1.5_attention_scale_1.1  2.57
ngram_lm_scale_1.5_attention_scale_1.2  2.57
ngram_lm_scale_1.5_attention_scale_1.3  2.57
ngram_lm_scale_2.0_attention_scale_1.7  2.57

2021-10-14 20:16:17,979 INFO [decode.py:474] batch 0/782, cuts processed until now is 6
2021-10-14 20:17:54,037 INFO [decode.py:474] batch 100/782, cuts processed until now is 434
2021-10-14 20:19:26,335 INFO [decode.py:474] batch 200/782, cuts processed until now is 885
2021-10-14 20:20:52,910 INFO [decode.py:474] batch 300/782, cuts processed until now is 1327
2021-10-14 20:22:15,968 INFO [decode.py:474] batch 400/782, cuts processed until now is 1807
2021-10-14 20:23:42,595 INFO [decode.py:474] batch 500/782, cuts processed until now is 2238
2021-10-14 20:25:06,502 INFO [decode.py:474] batch 600/782, cuts processed until now is 2584
2021-10-14 20:26:46,868 INFO [decode.py:474] batch 700/782, cuts processed until now is 2785
2021-10-14 20:30:11,556 INFO [decode.py:523]
For test-other, WER of different settings are:
ngram_lm_scale_1.5_attention_scale_1.9  5.8     best for test-other
ngram_lm_scale_1.7_attention_scale_1.9  5.8
ngram_lm_scale_1.7_attention_scale_2.0  5.8
ngram_lm_scale_1.9_attention_scale_2.0  5.8
ngram_lm_scale_1.3_attention_scale_1.5  5.81
ngram_lm_scale_1.5_attention_scale_1.5  5.81
ngram_lm_scale_1.5_attention_scale_1.7  5.81
ngram_lm_scale_1.7_attention_scale_1.7  5.81
ngram_lm_scale_1.5_attention_scale_1.3  5.82
ngram_lm_scale_1.9_attention_scale_1.9  5.82
ngram_lm_scale_1.2_attention_scale_1.2  5.83
ngram_lm_scale_1.3_attention_scale_1.3  5.83
ngram_lm_scale_1.3_attention_scale_1.7  5.83
ngram_lm_scale_1.5_attention_scale_1.2  5.83
ngram_lm_scale_1.5_attention_scale_2.0  5.83
ngram_lm_scale_2.0_attention_scale_2.0  5.83
```

[icefall]: https://github.com/k2-fsa/icefall

### Note

This repo uses `git lfs`. See <https://git-lfs.github.com/>
