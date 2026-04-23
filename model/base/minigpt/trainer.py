import torch


class Trainer:
    def __init__(self, model, data_processor, optimizer, eval_iters):
        self.model = model
        self.data_processor = data_processor
        self.optimizer = optimizer
        self.eval_iters = eval_iters

    @torch.no_grad()
    def estimate_loss(self):
        print("Calculating loss..")
        out = {}
        self.model.eval()

        for split in ("train", "val"):
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                percentage = (k + 1) / self.eval_iters * 100
                print(f"\r\033[K{split} Validation Progress: {percentage:.2f}", end="", flush=True)
                x, y = self.data_processor.get_batch(split)
                _, loss = self.model(x, y)
                losses[k] = loss.item()
            print()
            out[split] = losses.mean().item()

        self.model.train()
        return out
    
    def train(self, max_iters, eval_interval):
        for step in range(max_iters):
            percentage = (step + 1) / max_iters * 100
            print(f"\r\033[KTrain Progress: {percentage:.2f}", end="", flush=True)
            if step % eval_interval == 0 or step == max_iters - 1:
                print()
                losses = self.estimate_loss()
                print(
                    f"step {step}: "
                    f"train loss {losses['train']:.4f}, "
                    f"val loss {losses['val']:.4f}"
                )
            
            x, y = self.data_processor.get_batch("train")
            _, loss = self.model(x, y)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
