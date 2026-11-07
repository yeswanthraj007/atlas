import signal, time, torch
signal.alarm(300)  # hard limit: never hang silently
print("torch", torch.__version__, "mps", torch.backends.mps.is_available(), flush=True)
from transformers import AutoModel
import sys
name = sys.argv[1]
rev = sys.argv[2] if len(sys.argv) > 2 else None
t0 = time.time()
model = AutoModel.from_pretrained(name, revision=rev, dtype=torch.float32)
print(f"loaded in {time.time()-t0:.0f}s; dtype {next(model.parameters()).dtype}", flush=True)
dev = torch.device("mps")
model = model.to(dev); model.train()
head = torch.nn.Linear(768, 3).to(dev)
opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=2e-5)
for L, B in ((480, 16), (384, 16), (256, 16)):
    x = torch.randint(5, 20000, (B, L), device=dev); m = torch.ones_like(x)
    times = []
    for i in range(4):
        t = time.time()
        out = model(input_ids=x, attention_mask=m).last_hidden_state[:, 0]
        loss = head(out).pow(2).mean(); opt.zero_grad(); loss.backward(); opt.step(); torch.mps.synchronize()
        times.append(time.time() - t)
    s = sum(times[1:]) / 3
    print(f"len {L} batch {B}: {s:.2f}s/step -> {B/s:.1f} samples/s", flush=True)
