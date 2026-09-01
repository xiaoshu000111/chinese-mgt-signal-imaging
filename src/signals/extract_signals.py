"""① 用中文因果 LM 提取逐 token 概率信号。

一次前向取 logits -> log-softmax 分布，向量化算 5 条通道，
落盘 .npz（含信号矩阵 + 缩放元信息），供转图阶段读取。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import config
from src.data.load_data import load_split


class SignalExtractor:
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(config.SEED)
        np.random.seed(config.SEED)
        self.tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            config.BASE_MODEL_NAME, torch_dtype=torch.float16 if config.DEVICE == "cuda" else torch.float32
        ).to(config.DEVICE)
        self.model.eval()
        self.vocab = self.model.config.vocab_size

    @torch.no_grad()
    def extract_batch(self, texts: list[str]) -> list[dict]:
        """返回每个样本的 {logits分布相关通道: np.ndarray}，长度随 token 数变化。"""
        enc = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
            max_length=config.MAX_SEQ_LEN, add_special_tokens=True,
        ).to(config.DEVICE)
        logits = self.model(**enc).logits  # [B, T, V]
        # 前向用 fp16 省时省显存，但后处理必须升 fp32：rank 依赖逐元素概率比较，
        # fp16 的舍入会让近平局结果抖动、低概率 exp 后下溢为 0，破坏信号确定性（docs/01 §1）。
        logits = logits.float()
        log_probs = torch.log_softmax(logits, dim=-1)
        del logits  # 及时释放 [B,T,V] 大张量
        probs = log_probs.exp()

        input_ids = enc["input_ids"]          # [B, T]
        # 逐 token 真实概率/对数概率
        logp = log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)   # [B, T]
        p_true = probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
        # rank：分布中比真实 token 概率更高的 token 数量 + 1
        rank = (probs > p_true.unsqueeze(-1)).sum(-1) + 1                  # [B, T]
        rank_norm = rank.float() / self.vocab
        # entropy / top_prob
        entropy = -(probs * log_probs).sum(-1)                             # [B, T]
        top_prob = probs.max(-1).values                                    # [B, T]

        seq_lens = enc["attention_mask"].sum(-1).tolist()
        results = []
        for i in range(len(texts)):
            n = seq_lens[i]
            channels = {
                "logp": logp[i, :n].cpu().numpy(),
                "rank": rank[i, :n].cpu().numpy().astype(np.float32),
                "rank_norm": rank_norm[i, :n].cpu().numpy(),
                "entropy": entropy[i, :n].cpu().numpy(),
                "top_prob": top_prob[i, :n].cpu().numpy(),
            }
            results.append(channels)
        return results

    def run_split(self, split: str):
        """对某个 split 逐批提取并落盘 .npz。"""
        out_dir = config.SIGNALS_DIR / split
        out_dir.mkdir(parents=True, exist_ok=True)
        recs = load_split(split)
        if config.MAX_SAMPLES:
            recs = recs[: config.MAX_SAMPLES]

        for start in tqdm(range(0, len(recs), config.BATCH_SIZE), desc=f"extract {split}"):
            batch = recs[start : start + config.BATCH_SIZE]
            # 断点续跑：已落盘的样本跳过（OOM/中断后重跑只补缺口；
            # .npz 只依赖文本+基础模型，跳过不会引入不一致）
            batch = [r for r in batch
                     if not (out_dir / f"{r['sample_id']}.npz").exists()]
            if not batch:
                continue
            channels_list = self.extract_batch([r["text"] for r in batch])
            for rec, channels in zip(batch, channels_list):
                np_path = out_dir / f"{rec['sample_id']}.npz"
                # 保存信号 + 元信息（label/model 一并存，方便后续免查 json）
                np.savez(
                    np_path,
                    label=rec["label"] if rec["label"] is not None else -1,
                    model=str(rec["model"]) if rec["model"] else "",
                    **channels,
                )
        print(f"[extract] {split} 完成 -> {out_dir}")


if __name__ == "__main__":
    ext = SignalExtractor()
    ext.run_split("train")
    ext.run_split("dev")
    ext.run_split("test_with_label")