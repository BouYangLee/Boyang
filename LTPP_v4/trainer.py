# trainer.py
import torch
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, model, lr=1e-3, device='cpu'):
        """
        model: LA_TPP_Model instance
        """
        self.device = device
        self.model = model.to(device)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.Adam(params, lr=lr)

    def seqs_from_batch(self, action_time, action_type, pad_mask):
        """
        action_time: [B, L] tensor
        action_type: [B, L] tensor
        pad_mask: [B, L] binary tensor
        returns: list of sequences, each seq is list of (action_id(int), time(float))
        """
        seqs = []
        B, L = action_time.shape
        for i in range(B):
            seq = []
            for j in range(L):
                if pad_mask[i, j] == 0:
                    continue
                seq.append((int(action_type[i, j].item()), float(action_time[i, j].item())))
            seqs.append(seq)
        return seqs

    def compute_batch_loss(self, batch):
        """
        batch is the tuple returned by dataset.collate_fn:
        (mental_time, mental_type, action_time, action_type, pad_mask_action, action_time_to_event)
        """
        _, _, action_time, action_type, pad_mask, _ = batch
        action_time = action_time.to(self.device)
        action_type = action_type.to(self.device)
        pad_mask = pad_mask.to(self.device)

        seqs = self.seqs_from_batch(action_time, action_type, pad_mask)
        losses = []
        for seq in seqs:
            loglik, integral = self.model.handle_event_sequence(seq)
            losses.append(-loglik + integral)
        return torch.stack(losses).mean()

    def train_epoch(self, dataloader: DataLoader):
        """
        Runs one training epoch over dataloader.
        Returns average loss (scalar).
        """
        self.model.train()
        total_loss = 0.0
        n = 0
        for batch in dataloader:
            self.opt.zero_grad()
            loss = self.compute_batch_loss(batch)
            loss.backward()
            # # ---------- debug grads after backward ----------
            # print("---- debug grads (after loss.backward()) ----")
            # cnt = 0
            # for param_name in list(self.model.Thetas.keys())[:6]:  # show up to 6 Thetas
            #     theta = self.model.Thetas[param_name]
            #     g = theta.grad
            #     print("Theta:", param_name, " grad is None?", g is None, " norm=",
            #           None if g is None else float(g.norm().item()))
            #     cnt += 1
            #     if cnt >= 6:
            #         break
            # # also check K grad and other params
            # print("K requires_grad:", self.model.K.requires_grad, "K.grad is None?",
            #       self.model.K.grad is None if hasattr(self.model.K, 'grad') else 'no grad attr')
            # # check one dynamics param
            # print("alpha_un grad is None?", self.model.alpha_un.grad is None)
            # print("---------------------------------------------")

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.opt.step()
            bs = batch[2].shape[0]
            total_loss += float(loss.item()) * bs
            n += bs
        return total_loss / n if n > 0 else 0.0
