#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train KG-LMF for CODE tasks using CodeBERT embeddings (d=768).

Only trains on code datasets (KodCode + BigCodeBench).
Uses CodeBERT for meaningful code embeddings so the latent graph
stores useful code patterns.

Usage:
    cd D:/KGLM/KGLMF
    /d/anaconda3/envs/kglmf/python experiments/scripts/train_code_composer.py \
        --epochs 50 --batch_size 8 --device cpu
"""

import os, sys, json, random, argparse
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.graph.graph import LatentReasoningGraph
from src.graph.node import NodeType
from src.composer.composer import MemoryComposer
from src.router.router import GraphRouter
from src.injection.injector import LatentInjector
from src.api.embedders import CodeBERTEmbedder


class CodeDataset(Dataset):
    """Load code samples from KodCode + BigCodeBench."""

    def __init__(self, data_dir: str, embedder: CodeBERTEmbedder, max_samples: int = 400):
        self.embedder = embedder
        self.samples = []

        for fname in ['kodcode_real.json', 'bigcodebench_real.json']:
            fpath = os.path.join(data_dir, fname)
            if not os.path.exists(fpath):
                continue
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for item in data[:max_samples//2]:
                q = item.get('prompt') or item.get('question', '')
                a = item.get('answer') or item.get('solution', '')
                if q and str(a) and len(str(q)) > 10:
                    self.samples.append((str(q), str(a)))

        print(f"Code training samples: {len(self.samples)}")
        self.embeddings = []
        print("Computing CodeBERT embeddings...")
        for q, a in tqdm(self.samples):
            q_emb = torch.from_numpy(embedder.encode(q))
            a_emb = torch.from_numpy(embedder.encode(a[:1000]))
            self.embeddings.append((q_emb, a_emb))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        q, a = self.samples[idx]
        q_emb, a_emb = self.embeddings[idx]
        neg_idx = random.choice([i for i in range(len(self)) if i != idx])
        _, neg_emb = self.embeddings[neg_idx]
        return q_emb, a_emb, neg_emb, q, a


class CodeComposerTrainer:
    """Trains KG-LMF components for code tasks with d_model=768 (CodeBERT dimension)."""

    def __init__(self, d_model=768, device='cpu', lr=1e-3):
        self.device = device
        self.d_model = d_model

        # Components with CodeBERT dimension
        self.composer = MemoryComposer(d_model=d_model, n_heads=8, n_layers=4,
                                       n_learnable_queries=16).to(device)
        self.router = GraphRouter(d_model=d_model).to(device)
        self.injector = LatentInjector(d_model=d_model,
                                       injection_layers=[0, 4, 8]).to(device)
        for key in self.injector.gates:
            nn.init.constant_(self.injector.gates[key], 0.0)  # Start open

        self.graph = LatentReasoningGraph(d_model=d_model)

        self.all_params = (list(self.composer.parameters()) +
                          list(self.router.parameters()) +
                          list(self.injector.parameters()))
        self.optimizer = torch.optim.AdamW(self.all_params, lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=150, T_mult=2)
        self.cos = nn.CosineSimilarity(dim=-1)

    def train_step(self, batch):
        q_embs, a_embs, neg_embs, questions, answers = batch
        B = len(questions)
        losses = []

        for i in range(B):
            q_emb = q_embs[i].to(self.device)
            a_emb = a_embs[i].to(self.device)
            neg_emb = neg_embs[i].to(self.device)

            interaction = (q_emb + a_emb) / 2
            self.graph.add_node(interaction, NodeType.CAPSULE, role='code')

            k = min(8, len(self.graph))
            if k < 2: continue

            scores = self.router.score_nodes(q_emb, self.graph)
            node_ids = self.graph.get_node_ids()
            k = min(k, len(node_ids))
            top_idx = torch.topk(scores, k).indices
            retrieved = torch.stack([
                self.graph.get_node(node_ids[j]).embedding.to(self.device)
                for j in top_idx
            ])

            mem = self.composer.compose(q_emb, self.graph, k, 'code')
            if mem.shape[0] == 0: continue

            pos_sim = self.cos(mem.mean(0).unsqueeze(0), a_emb.unsqueeze(0))
            neg_sim = self.cos(mem.mean(0).unsqueeze(0), neg_emb.unsqueeze(0))
            loss = F.relu(0.3 - pos_sim + neg_sim).mean()
            nce = F.cross_entropy(
                torch.cat([pos_sim, neg_sim]).unsqueeze(0) / 0.07,
                torch.zeros(1, dtype=torch.long, device=self.device))
            losses.append(loss + 0.5 * nce)

        if not losses: return 0.0
        total = torch.stack(losses).mean()
        self.optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.all_params, 2.0)
        self.optimizer.step()
        self.scheduler.step()
        return total.item()

    def train(self, dataloader, epochs=50, save_dir='checkpoints_code/'):
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"Code Composer Training — {epochs} epochs, d_model=768")
        print(f"Trainable params: {sum(p.numel() for p in self.all_params):,}")
        print(f"{'='*60}\n")

        best_loss = float('inf')
        for epoch in range(epochs):
            ep_losses = []
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                loss = self.train_step(batch)
                ep_losses.append(loss)
                pbar.set_postfix({'loss': f'{np.mean(ep_losses[-10:]):.4f}',
                                  'nodes': len(self.graph)})

            avg = np.mean(ep_losses)
            print(f"  Epoch {epoch+1}: loss={avg:.4f}, nodes={len(self.graph)}")
            if avg < best_loss:
                best_loss = avg
                self.save(os.path.join(save_dir, 'best_model.pt'))
            if (epoch+1) % 10 == 0:
                self.save(os.path.join(save_dir, f'checkpoint_{epoch+1}.pt'))

        print(f"\nTraining complete. Best loss: {best_loss:.4f}")
        return best_loss

    def save(self, path):
        torch.save({
            'composer': self.composer.state_dict(),
            'router': self.router.state_dict(),
            'injector': self.injector.state_dict(),
            'graph_nodes': [(nid, node.embedding, node.node_type.value)
                           for nid, node in self.graph.nodes.items()],
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.composer.load_state_dict(ckpt['composer'])
        self.router.load_state_dict(ckpt['router'])
        self.injector.load_state_dict(ckpt['injector'])
        for nid, emb, ntype in ckpt['graph_nodes']:
            self.graph.add_node(emb, NodeType(ntype))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_samples', type=int, default=400)
    parser.add_argument('--save_dir', type=str, default='checkpoints_code/')
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'data'))
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    print("=" * 60)
    print("KG-LMF Code Composer Training (CodeBERT d=768)")
    print("=" * 60)

    embedder = CodeBERTEmbedder(device=args.device)
    dataset = CodeDataset(args.data_dir, embedder, args.max_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    trainer = CodeComposerTrainer(d_model=768, device=args.device, lr=args.lr)
    trainer.train(dataloader, epochs=args.epochs, save_dir=args.save_dir)


if __name__ == '__main__':
    main()
