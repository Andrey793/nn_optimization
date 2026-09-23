import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import os
import urllib.request
from tqdm import tqdm

from torch.profiler import profile, ProfilerActivity, record_function, schedule
import contextlib

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler



class TromptCell(nn.Module):
    def __init__(self, n_columns, n_prompts, d_model):
        super().__init__()
        # Embeddings (Figure 3.2)
        self.feature_emb_weight = nn.Parameter(torch.empty(n_columns, d_model))
        self.feature_emb_bias = nn.Parameter(torch.empty(n_columns, d_model))
        self.ln_emb = nn.LayerNorm(d_model)

        # Importance Getter (Figure 3.1)
        self.ln_col = nn.LayerNorm(d_model)
        self.ln_prompt = nn.LayerNorm(d_model)
        self.dense_imp = nn.Linear(2 * d_model, d_model) 

        self.emb_column = nn.Parameter(torch.empty(n_columns, d_model))
        self.emb_prompt = nn.Parameter(torch.empty(n_prompts, d_model))

        # Modified expansion block (Figure 3.3)
        # Without non-linearities! This is important to make significant speed-ups possible.
        self.dense_expand = nn.Linear(1, n_prompts)

        self.reset_parameters()

    def reset_parameters(self):
        d_rsqrt = self.feature_emb_weight.shape[1] ** -0.5
        nn.init.uniform_(self.feature_emb_weight, -d_rsqrt, d_rsqrt)
        nn.init.uniform_(self.feature_emb_bias, -d_rsqrt, d_rsqrt)
        nn.init.normal_(self.emb_column, std=0.01)
        nn.init.normal_(self.emb_prompt, std=0.01)

    def forward(self, x: torch.Tensor, prev_cell_out: torch.Tensor) -> torch.Tensor:
        x_emb = x.unsqueeze(-1) * self.feature_emb_weight + self.feature_emb_bias.unsqueeze(0)
        x_emb = F.relu(x_emb)
        x_emb = self.ln_emb(x_emb)

        x_prompt = self.emb_prompt.unsqueeze(0)
        x_prompt = self.dense_imp(torch.cat([self.ln_prompt(x_prompt), prev_cell_out], dim=-1)) + x_prompt
        x_column = self.ln_col(self.emb_column)
        mask = torch.softmax(x_prompt @ x_column.transpose(0, 1), dim=-1)

        # x_emb = x_emb.unsqueeze(1) + self.dense_expand(x_emb.unsqueeze(-1)).permute(0, 3, 1, 2)
        # x_out = (mask.unsqueeze(-1) * x_emb).sum(dim=2)

        w = self.dense_expand.weight#.unsqueeze(-1)
        b = self.dense_expand.bias#.unsqueeze(-1).unsqueeze(-1)
        #x_emb = x_emb.unsqueeze(1) + x_emb.unsqueeze(1) * w + b
        #x_out = (mask.unsqueeze(-1) * x_emb).sum(dim=2)
        x_out = mask @ x_emb * (1 + w).view(1, -1, 1) + b.view(1, -1, 1)


        return x_out


class TromptDownstream(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.dense0 = nn.Linear(d_model, 1)
        self.dense1 = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.dense_out = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pw = torch.softmax(self.dense0(x).squeeze(-1), dim=-1)
        #xnew = (pw.unsqueeze(-1) * x).sum(dim=-2)
        xnew = (pw.unsqueeze(-2) @ x).squeeze(-2)
        return self.dense_out(self.ln(F.relu(self.dense1(xnew))))


class Trompt(nn.Module):
    def __init__(self, n_columns, n_prompts, d_model, n_cycles):
        super().__init__()
        self.tcells = nn.ModuleList([TromptCell(n_columns, n_prompts, d_model) for _ in range(n_cycles)])
        self.tdown = TromptDownstream(d_model)
        self.prompt = nn.Parameter(torch.empty(n_prompts, d_model))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.prompt, std=0.01)

    def forward(self, x):
        x_prompt = self.prompt.unsqueeze(0)
        outputs = []
        for cell in self.tcells:
            outputs.append(self.tdown(cell(x, x_prompt)))
        return torch.stack(outputs, dim=1).squeeze(-1)


def load_from_url(url, cache_dir='.'):
    filename = os.path.join(cache_dir, url.split('/')[-1])
    if not os.path.exists(filename):
        with tqdm(unit='B', unit_scale=True, desc=filename) as pbar:
            urllib.request.urlretrieve(url, filename, reporthook=lambda _, b, t: pbar.update(b))
    return torch.load(filename, map_location=torch.device('cpu'), weights_only=True)


TRAIN_DATA = "https://huggingface.co/datasets/puhsu/hw01-data/resolve/main/train_dataset.pt"
VAL_DATA = "https://huggingface.co/datasets/puhsu/hw01-data/resolve/main/val_dataset.pt"

if __name__ == "__main__":
    torch.manual_seed(0)

    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    is_main = rank == 0

    train_dataset = torch.utils.data.TensorDataset(*map(torch.nan_to_num, load_from_url(TRAIN_DATA)))
    val_dataset = torch.utils.data.TensorDataset(*map(torch.nan_to_num, load_from_url(VAL_DATA)))

    Y_mean = train_dataset.tensors[1].mean()
    Y_std = train_dataset.tensors[1].std()
    train_dataset.tensors = (train_dataset.tensors[0], (train_dataset.tensors[1] - Y_mean) / Y_std)

    model = Trompt(n_columns=train_dataset.tensors[0].shape[1], n_prompts=128, d_model=128, n_cycles=6)
    device = torch.device(f'cuda:{local_rank}')
    model.to(device)
    model.compile()
    model = DDP(model, device_ids=[local_rank])
    n_cycles = len(model.module.tcells)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    train_dl = torch.utils.data.DataLoader(train_dataset, num_workers=0, batch_size=1024, sampler=train_sampler)
    val_dl = torch.utils.data.DataLoader(val_dataset, num_workers=0, batch_size=1024, sampler=val_sampler)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

    AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=False)#(AMP_DTYPE == torch.float16))

    EPOCHS = 5

    USE_PROFILER = False
    sort_by_keyword = "self_" + str(device) + "_time_total"

    prof_ctx = (
        profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=6, warmup=4, active=2, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('tmp'),
            record_shapes=True,
        )
        if USE_PROFILER
        else contextlib.nullcontext()
    )
    
    with prof_ctx as p:
        for e in range(1, EPOCHS + 1):
            model.train()
            train_sampler.set_epoch(e)
            for step, batch in enumerate(tqdm(train_dl, disable=not is_main)):
                x, y = batch
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=AMP_DTYPE):
                    with torch.profiler.record_function("forward pass"):
                        pred = model(x.to(device))
                    with torch.profiler.record_function("loss"):
                        loss = F.mse_loss(pred, y.unsqueeze(1).repeat(1, n_cycles).to(device))
                with torch.profiler.record_function("backward pass"):
                    scaler.scale(loss).backward()
                with torch.profiler.record_function("optimizer step"):
                    scaler.step(optimizer)
                    scaler.update()
                if USE_PROFILER:
                    p.step()
                if USE_PROFILER and step >= (1+1+2)*3:
                    break

            model.eval()
            mae = torch.zeros((), device=device)
            with torch.inference_mode():
                for batch in val_dl:
                    x, y = batch
                    with torch.autocast('cuda', dtype=AMP_DTYPE):
                        pred = model(x.to(device))
                    mae += (pred.float().mean(dim=-1) * Y_std + Y_mean - y.to(device)).abs().sum()

                dist.all_reduce(mae, op=dist.ReduceOp.SUM)
                mae = mae.item() / len(val_dataset)

                if is_main:
                    print(f'>>> Epoch {e:>02}')
                    print(f'Validation MAE = {mae:.5f}')
                    print('>>>\n')

    dist.destroy_process_group()
