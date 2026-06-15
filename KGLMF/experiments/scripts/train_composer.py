#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train the KG-LMF Memory Composer using contrastive learning on real QA data.

Training strategy:
  1. Load QA pairs from real datasets (TriviaQA, StrategyQA, PopQA)
  2. For each question, embed with sentence-transformers
  3. Write interaction (question+answer) into the latent graph
  4. For similar questions, retrieve graph nodes and compose memory tokens
  5. Contrastive loss: push composed memory toward successful answer embeddings,
     pull away from random/irrelevant node embeddings
  6. Train: composer (queries, cross-attn, role_emb), router (score_proj),
     injector (gates, memory_proj)

Usage:
    /d/WorkSoftware/python D:/KGLM/KGLMF/experiments/scripts/train_composer.py \
        --epochs 20 --batch_size 16 --lr 1e-3 --save_dir checkpoints/
"""

import os, sys, json, random, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

# Add project root to path (D:/KGLM/KGLMF)
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.graph.graph import LatentReasoningGraph
from src.graph.node import NodeType
from src.composer.composer import MemoryComposer
from src.router.router import GraphRouter
from src.router.tts import AdaptiveTTSController
from src.injection.injector import LatentInjector


# ═══════════════════════════════════════════════════════════
# TRAINING DATASET
# ═══════════════════════════════════════════════════════════

class QATrainingDataset(Dataset):
    """Load QA pairs from dataset JSON files for contrastive training."""

    def __init__(self, data_dir: str, embedder, max_samples: int = 500):
        self.embedder = embedder
        self.pairs = []
        self.embeddings = []

        # Load datasets
        dataset_files = [
            'trivia_qa_real.json', 'strategyqa_real.json',
            'akariasai_PopQA_real.json'
        ]
        for fname in dataset_files:
            fpath = os.path.join(data_dir, fname)
            if os.path.exists(fpath):
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for item in data[:max(1, max_samples // len(dataset_files))]:
                    q = item.get('question', item.get('input', ''))
                    a = item.get('answer', item.get('target', item.get('output', '')))
                    if q and a:
                        self.pairs.append((q, a))

        print(f"Loaded {len(self.pairs)} QA pairs for training")

        # Pre-compute question embeddings
        print("Computing question embeddings...")
        for q, a in tqdm(self.pairs):
            q_emb = torch.from_numpy(
                embedder.encode(q, normalize_embeddings=True).astype(np.float32))
            a_emb = torch.from_numpy(
                embedder.encode(a[:500], normalize_embeddings=True).astype(np.float32))
            self.embeddings.append((q_emb, a_emb))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        q, a = self.pairs[idx]
        q_emb, a_emb = self.embeddings[idx]
        # Random negative: a different QA pair's answer
        neg_idx = random.choice([i for i in range(len(self)) if i != idx])
        _, neg_emb = self.embeddings[neg_idx]
        return q_emb, a_emb, neg_emb, q, a


# ═══════════════════════════════════════════════════════════
# CONTRASTIVE LOSS
# ═══════════════════════════════════════════════════════════

class ContrastiveMemoryLoss(nn.Module):
    """Contrastive loss for memory composition training.

    Pulls composed memory toward positive examples (correct answers)
    and pushes away from negative examples (random/irrelevant nodes).
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, memory_tokens, pos_embedding, neg_embedding):
        """
        Args:
            memory_tokens: (n_queries, d_model) — composed memory
            pos_embedding: (d_model,) — correct answer embedding
            neg_embedding: (d_model,) — random answer embedding
        """
        # Pool memory tokens by mean
        mem_pooled = memory_tokens.mean(dim=0)  # (d_model,)

        # Positive similarity
        pos_sim = self.cos(mem_pooled.unsqueeze(0), pos_embedding.unsqueeze(0))
        # Negative similarity
        neg_sim = self.cos(mem_pooled.unsqueeze(0), neg_embedding.unsqueeze(0))

        # InfoNCE-style contrastive loss
        logits = torch.cat([pos_sim, neg_sim]) / self.temperature
        labels = torch.zeros(1, dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits.unsqueeze(0), labels)


# ═══════════════════════════════════════════════════════════
# TRAINER
# ═══════════════════════════════════════════════════════════

class ComposerTrainer:
    def __init__(self, d_model=384, device='cpu'):
        self.device = device
        self.d_model = d_model

        # Initialize all trainable components
        self.composer = MemoryComposer(d_model=d_model, n_heads=8, n_layers=4,
                                       n_learnable_queries=16).to(device)
        self.router = GraphRouter(d_model=d_model).to(device)
        self.tts = AdaptiveTTSController()
        self.injector = LatentInjector(d_model=d_model,
                                       injection_layers=[0, 4, 8]).to(device)
        self.graph = LatentReasoningGraph(d_model=d_model)

        # Loss
        self.contrastive_loss = ContrastiveMemoryLoss(temperature=0.07)

        # Collect all trainable parameters
        self.all_params = list(self.composer.parameters()) + \
                          list(self.router.parameters()) + \
                          list(self.injector.parameters())

        self.optimizer = torch.optim.AdamW(self.all_params, lr=1e-3, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=500)

        # Injector auxiliary loss: encourage gates to be meaningful
        self.gate_reg_weight = 0.01

    def train_step(self, batch):
        """Single training step."""
        q_embs, a_embs, neg_embs, questions, answers = batch

        B = len(questions)
        losses = []
        gate_values = []

        for i in range(B):
            q_emb = q_embs[i].to(self.device)
            a_emb = a_embs[i].to(self.device)
            neg_emb = neg_embs[i].to(self.device)
            question = questions[i]
            answer = answers[i]

            # ---- Write interaction to graph ----
            interaction_emb = (q_emb + a_emb) / 2
            role_idx = i % 3  # distribute across planner/coder/critic
            role = ["planner", "coder", "critic"][role_idx]
            node_type = [NodeType.ROLE, NodeType.TRAJECTORY, NodeType.CONTEXT][role_idx]
            self.graph.add_node(interaction_emb, node_type, role=role,
                               task_id=f"train_{i}")

            # ---- Retrieve from graph ----
            k = min(8, len(self.graph))
            if k < 2:
                continue

            scores = self.router.score_nodes(q_emb, self.graph)
            node_ids = self.graph.get_node_ids()
            k = min(k, len(node_ids))
            top_idx = torch.topk(scores, k).indices
            retrieved_embs = torch.stack([
                self.graph.get_node(node_ids[j]).embedding.to(self.device)
                for j in top_idx
            ])

            # ---- Compose memory ----
            memory_tokens = self.composer.compose(
                q_emb, self.graph, k, role)

            # ---- Contrastive loss ----
            if memory_tokens.shape[0] > 0:
                cont_loss = self.contrastive_loss(memory_tokens, a_emb, neg_emb)
                losses.append(cont_loss)

                # ---- Gate auxiliary loss ----
                # Encourage gates to produce non-trivial values (not all 0 or all 1)
                for layer_key in self.injector.gates:
                    gate = torch.sigmoid(self.injector.gates[layer_key])
                    # Penalize extreme gate values
                    gate_loss = ((gate - 0.5) ** 2).mean()
                    losses.append(self.gate_reg_weight * gate_loss)
                    gate_values.append(gate.mean().item())

        if not losses:
            return None, None

        total_loss = torch.stack(losses).mean()

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.all_params, 1.0)
        self.optimizer.step()
        self.scheduler.step()

        avg_gate = np.mean(gate_values) if gate_values else 0.0
        return total_loss.item(), avg_gate

    def train(self, dataloader, epochs=20, save_dir='checkpoints/'):
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"Training Composer — {epochs} epochs, {len(dataloader)} batches/epoch")
        print(f"Trainable params: {sum(p.numel() for p in self.all_params):,}")
        print(f"{'='*60}\n")

        best_loss = float('inf')
        for epoch in range(epochs):
            epoch_losses = []
            epoch_gates = []
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

            for batch in pbar:
                loss, gate_avg = self.train_step(batch)
                if loss is not None:
                    epoch_losses.append(loss)
                    epoch_gates.append(gate_avg)
                    pbar.set_postfix({
                        'loss': f'{np.mean(epoch_losses[-20:]):.4f}',
                        'gate': f'{np.mean(epoch_gates[-20:]):.3f}',
                        'nodes': len(self.graph)
                    })

            avg_loss = np.mean(epoch_losses) if epoch_losses else 0
            avg_gate = np.mean(epoch_gates) if epoch_gates else 0
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, gate_mean={avg_gate:.3f}, "
                  f"graph_nodes={len(self.graph)}, lr={self.scheduler.get_last_lr()[0]:.2e}")

            # Save checkpoint
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save(os.path.join(save_dir, 'best_model.pt'))
                print(f"  -> Best model saved (loss={best_loss:.4f})")

            # Periodic save
            if (epoch + 1) % 5 == 0:
                self.save(os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt'))

        print(f"\nTraining complete. Best loss: {best_loss:.4f}")
        self.print_gate_analysis()
        return best_loss

    def save(self, path):
        torch.save({
            'composer': self.composer.state_dict(),
            'router': self.router.state_dict(),
            'injector': self.injector.state_dict(),
            'graph_nodes': [(nid, node.embedding, node.node_type.value)
                           for nid, node in self.graph.nodes.items()],
            'graph_edges': [(src, dst, edge.weight, edge.edge_type.value)
                           for src, dst, edge in self.graph.edges.items()],
        }, path)
        print(f"  Saved checkpoint to {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.composer.load_state_dict(ckpt['composer'])
        self.router.load_state_dict(ckpt['router'])
        self.injector.load_state_dict(ckpt['injector'])
        # Restore graph
        for nid, emb, ntype in ckpt['graph_nodes']:
            self.graph.add_node(emb, NodeType(ntype), node_id=nid)
        print(f"  Loaded checkpoint from {path}")

    def print_gate_analysis(self):
        print("\n--- Gate Analysis ---")
        for layer_key in self.injector.gates:
            gate_val = torch.sigmoid(self.injector.gates[layer_key]).item()
            print(f"  Layer {layer_key}: gate = {gate_val:.4f} "
                  f"({'open' if gate_val > 0.5 else 'closed'})")
        print("  (Gates > 0.5 = memory actively used at that layer)")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_samples', type=int, default=500)
    parser.add_argument('--save_dir', type=str, default='checkpoints/')
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'data'))
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    print("=" * 60)
    print("KG-LMF Memory Composer Training")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Data dir: {args.data_dir}")
    print(f"Max samples: {args.max_samples}")

    # Load embedder
    print("\nLoading sentence-transformer embedder...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # Load dataset
    dataset = QATrainingDataset(args.data_dir, embedder, args.max_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Train
    trainer = ComposerTrainer(d_model=384, device=args.device)
    trainer.train(dataloader, epochs=args.epochs, save_dir=args.save_dir)

    print("\nDone! Trained model saved to", args.save_dir)


if __name__ == '__main__':
    main()
