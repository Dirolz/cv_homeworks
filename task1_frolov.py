import statistics
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def prepare_data() -> TensorDataset:
    X = torch.randn(10000, 128, device="cuda")
    y = torch.randint(0, 2, (10000,), device="cuda")
    return TensorDataset(X, y)

def train():
    dataloader = DataLoader(prepare_data(), batch_size=256, shuffle=True)
    model = nn.Sequential(
        nn.Linear(128, 512), nn.ReLU(),
        nn.Linear(512, 128), nn.ReLU(),
        nn.Linear(128, 2)
    ).cuda().train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    fwd_events, bwd_events, losses = [], [], []
    log_interval = 5

    for batch_idx, (data, target) in enumerate(dataloader):
        noise = torch.randn_like(data)
        data = data + noise

        optimizer.zero_grad()

        fwd_start, fwd_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        fwd_start.record()
        output = model(data)
        loss = criterion(output, target)
        fwd_end.record()

        bwd_start, bwd_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        bwd_start.record()
        loss.backward()
        bwd_end.record()
        optimizer.step()

        fwd_events.append((fwd_start, fwd_end))
        bwd_events.append((bwd_start, bwd_end))
        losses.append(loss.detach())

        if batch_idx % log_interval == 0:
            print(f"Batch {batch_idx} loss: {loss.item():.4f}")

    torch.cuda.synchronize()

    fwd_times = [s.elapsed_time(e) / 1000.0 for s, e in fwd_events]
    bwd_times = [s.elapsed_time(e) / 1000.0 for s, e in bwd_events]
    avg_loss = torch.stack(losses).mean().item()

    print(f"Epoch finished, avg loss: {avg_loss:.4f}")
    print(f"Avg forward time: {statistics.mean(fwd_times):.4f} s")
    print(f"Avg backward time: {statistics.mean(bwd_times):.4f} s")

if __name__ == '__main__':
    train()