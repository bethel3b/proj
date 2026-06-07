"""Generic step-based training loop with per-step scheduling and MLflow tracking."""

import copy
import os
from typing import Literal

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.utils import print_header


class Trainer:
    """Drives training off `total_steps`; evaluates every `eval_every_n_steps` and saves best ckpt."""

    def __init__(self, args: dict):
        """Unpack the flat `args` dict into the trainer's attributes."""
        # dataloader
        self.train_loader = args["train_loader"]
        self.valid_loader = args["valid_loader"]
        self.test_loader = args["test_loader"]
        # model
        self.model = args["model"]
        # optim
        self.lr = args["lr"]
        self.optimizer = args["optimizer"]
        # loss
        self.criterion = args["criterion"]
        # scheduler
        self.scheduler = args["scheduler"]
        # training
        self.total_steps = args["total_steps"]
        self.eval_every_n_steps = args["eval_every_n_steps"]
        self.log_every_n_steps = args["log_every_n_steps"]
        self.device = args["device"]
        self.checkpoint_dir = args["checkpoint_dir"]
        # mlflow
        self.run_name = args["run_name"]
        self.experiment_name = args["experiment_name"]
        self.tracker_url = args["tracker_url"]

        self.model.to(self.device)
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.batch_size = self.train_loader.batch_size

        # Track the best-by-val-loss weights so the final test uses them.
        self.best_state = copy.deepcopy(self.model.state_dict())
        self.best_val_loss = float("inf")
        self.best_step = 0
        self.val_loss = float("inf")

    def train_step(self, batch: dict, step: int) -> tuple[float, float, float]:
        """Run one optimizer step on `batch`; log per-step metrics; return (loss, lr, grad_norm)."""
        self.optimizer.zero_grad()

        logits = self.model(
            input_ids=batch["input_ids"].to(self.device),
            attention_mask=batch["attention_mask"].to(self.device),
        )
        labels = batch["labels"].to(self.device)

        # Drop the last logit (no next-token target) so logits[:, t] is
        # paired with the token at position t+1, which `labels` already holds.
        logits_shifted = logits[:, :-1, :]
        batch_size, seq_len, vocab_size = logits_shifted.shape
        logits_shifted = logits_shifted.reshape(batch_size * seq_len, vocab_size)
        labels = labels.reshape(batch_size * seq_len)

        loss = self.criterion(logits_shifted, labels)
        loss.backward()
        # max_norm=inf measures the global grad norm without clipping.
        grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=float("inf")
        )
        self.optimizer.step()
        self.scheduler.step()

        train_loss = loss.item()
        grad_norm_val = grad_norm.item()
        current_lr = self.optimizer.param_groups[0]["lr"]
        mlflow.log_metrics(
            {
                "step_level/Train_loss": train_loss,
                "step_level/lr": current_lr,
                "step_level/grad_norm": grad_norm_val,
            },
            step=step,
        )
        return train_loss, current_lr, grad_norm_val

    @torch.no_grad()
    def evaluate(
        self, loader: DataLoader, mode: Literal["Val", "Init", "Test"], step: int
    ) -> float:
        """Run a full pass over `loader` in eval mode; log eval/{mode}_loss; return average loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(loader, desc=f"[Step: {mode}]"):
            logits = self.model(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
            )
            labels = batch["labels"].to(self.device)

            logits_shifted = logits[:, :-1, :]
            batch_size, seq_len, vocab_size = logits_shifted.shape
            logits_shifted = logits_shifted.reshape(batch_size * seq_len, vocab_size)
            labels = labels.reshape(batch_size * seq_len)

            loss = self.criterion(logits_shifted, labels)
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"{mode} Loss: {avg_loss:.4f}")
        mlflow.log_metrics({f"eval/{mode}_loss": avg_loss}, step=step)
        return avg_loss

    def train(self) -> None:
        """Run initial validation, the step-based training loop, then a final test on best ckpt."""
        # set up mlflow
        mlflow.set_tracking_uri(self.tracker_url)
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name=self.run_name):
            mlflow.enable_system_metrics_logging()
            mlflow.log_params(
                {
                    "total_steps": self.total_steps,
                    "eval_every_n_steps": self.eval_every_n_steps,
                    "log_every_n_steps": self.log_every_n_steps,
                    "batch_size": self.batch_size,
                    "lr": self.lr,
                    "checkpoint_dir": self.checkpoint_dir,
                }
            )

            # Initial Validation (step=0 — model hasn't been touched yet).
            print_header(text="Initial Validation")
            init_loss = self.evaluate(loader=self.valid_loader, mode="Init", step=0)

            print_header(text="Started Training")
            self.model.train()
            train_iter = iter(self.train_loader)

            for step in tqdm(range(1, self.total_steps + 1), desc="Train"):
                # Fetch one batch. When the loader exhausts, rebuild the
                # iterator — this re-shuffles automatically (DataLoader behaviour).
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)

                train_loss, lr_val, grad_norm_val = self.train_step(
                    batch=batch, step=step
                )

                # Per-step stdout summary (MLflow logs every step regardless).
                if step % self.log_every_n_steps == 0:
                    print(
                        f"step {step:>7d}/{self.total_steps} | "
                        f"train loss {train_loss:.4f} | "
                        f"lr {lr_val:.2e} | "
                        f"grad_norm {grad_norm_val:.2f}"
                    )

                # Eval + best-checkpoint cadence. The `or step == total_steps`
                # clause guarantees a closing val_loss even if total_steps isn't
                # a multiple of eval_every_n_steps.
                if step % self.eval_every_n_steps == 0 or step == self.total_steps:
                    self.val_loss = self.evaluate(
                        loader=self.valid_loader, mode="Val", step=step
                    )
                    self.model.train()  # evaluate() flipped us into eval mode

                    if self.val_loss <= self.best_val_loss:
                        self.best_state = copy.deepcopy(self.model.state_dict())
                        self.best_step = step
                        self.best_val_loss = self.val_loss

                        if self.checkpoint_dir:
                            ckpt_path = os.path.join(self.checkpoint_dir, "best.pt")
                            torch.save(
                                {
                                    "step": self.best_step,
                                    "model_state_dict": self.best_state,
                                    "optimizer_state_dict": self.optimizer.state_dict(),
                                    "val_loss": self.best_val_loss,
                                },
                                ckpt_path,
                            )
                            print(f"Saved best checkpoint to {ckpt_path}")
                            mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")

            print(
                f"\nInitial Loss [step 0]={init_loss:.4f}"
                f"\nFinal Loss [step {self.total_steps}]={self.val_loss:.4f}"
                f"\nBest Loss [step {self.best_step}]={self.best_val_loss:.4f}"
            )

            # Final test — restore the best validation checkpoint first.
            self.model.load_state_dict(self.best_state)
            print_header(text=f"Final Test at step {self.best_step}")
            self.evaluate(loader=self.test_loader, mode="Test", step=self.best_step)
