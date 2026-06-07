"""Generic training loop with MLflow tracking, checkpointing and grad-norm logging."""

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
    """Runs train/val/test epochs, logging metrics and saving the best checkpoint."""

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
        self.device = args["device"]
        self.checkpoint_dir = args["checkpoint_dir"]
        # mlflow
        self.run_name = args["run_name"]
        self.experiment_name = args["experiment_name"]
        self.tracker_url = args["tracker_url"]

        self.model.to(self.device)
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Step-based knobs
        self.step_per_epoch = len(self.train_loader)  # affected by drop_last
        self.batch_size = self.train_loader.batch_size
        self.epochs = self.total_steps // self.step_per_epoch

        # Track the best-by-val-loss weights so the final test uses them.
        self.best_state = copy.deepcopy(self.model.state_dict())
        self.best_val_loss = float("inf")
        self.best_step = 0
        self.epoch_one_loss = None
        self.global_step = 0

    def train_step(self, batch, mode: Literal["Train", "Val", "Test"]) -> float:
        """."""
        if mode == "Train":
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

        if mode == "Train":
            loss.backward()
            # max_norm=inf measures the global grad norm without clipping.
            # Manual comp: compute_grad_norm(model=self.model)
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=float("inf")
            )
            self.optimizer.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            mlflow.log_metrics(
                {"step_level/lr": current_lr, "step_level/grad_norm": grad_norm},
                step=self.global_step,
            )
            # Step-level logging
            mlflow.log_metrics(
                {f"step_level/{mode}_loss": loss.item()}, step=self.global_step
            )

            self.scheduler.step()

        return loss.item()

    def run_eval(self, mode) -> None:
        with torch.no_grad():
            self.model.eval()
            total_loss = 0
            loader = self.valid_loader if mode == "Val" else self.test_loader
            for i, batch in enumerate(loader):
                val_loss = self.train_step(batch=batch, mode=mode)
                total_loss += val_loss
            mlflow.log_metrics({f"{mode}_loss": total_loss}, step=self.global_step)
            print(f"{mode} Loss at step {self.global_step} ={total_loss:.4f}")

        return total_loss

    def get_next_batch(self, data_loader):
        while True:
            for batch in data_loader:
                yield batch

    def train(self) -> None:
        """Run initial validation, the checkpointing epoch loop, then a final test."""
        # Training with MLflow logging
        print_header(text="Setting up mlflow")
        mlflow.set_tracking_uri(self.tracker_url)
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name=self.run_name):
            # Enable system metrics logging
            mlflow.enable_system_metrics_logging()
            mlflow.log_params(
                {
                    "epochs": self.epochs,
                    "batch_size": len(self.train_loader),
                    "total_steps": self.total_steps,
                    "lr": self.lr,
                    "checkpoint_dir": self.checkpoint_dir,
                }
            )

            # Initial Validation
            print_header(text="Initial Validation")
            init_loss = self.run_eval(mode="Val")

            print_header(text="Startsteped Training")

            self.model.train()
            train_generator = iter(self.get_next_batch(self.train_loader))
            for _ in tqdm(range(self.total_steps), desc="Train"):
                batch = next(train_generator)
                # Training
                train_loss = self.train_step(batch=batch, mode="Train")
                # TODO: Log every N steps
                if ((self.global_step + 1) % 5) == 0:
                    print(f"Train Loss at step {self.global_step} = {train_loss:.4f}")

                if (
                    (self.global_step + 1) % self.step_per_epoch
                ) == 0 or self.global_step == self.total_steps - 1:
                    val_loss = self.run_eval(mode="Val")
                    if self.epoch_one_loss is None:
                        self.epoch_one_loss = val_loss
                    self.model.train()

                    if val_loss <= self.best_val_loss:
                        self.best_state = copy.deepcopy(self.model.state_dict())
                        self.best_step = self.global_step
                        self.best_val_loss = val_loss

                        if self.checkpoint_dir:
                            ckpt_path = os.path.join(self.checkpoint_dir, "best.pt")
                            torch.save(
                                {
                                    "epoch": self.best_step,
                                    "model_state_dict": self.best_state,
                                    "optimizer_state_dict": self.optimizer.state_dict(),
                                    "val_loss": self.best_val_loss,
                                    "global_step": self.global_step,
                                },
                                ckpt_path,
                            )
                            print(f"Saved best checkpoint to {ckpt_path}")
                            mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")
                self.global_step += 1

            print(
                f"\nInitial Loss [epoch 0]={init_loss:.4f}"
                f"\nLoss [epoch 1]={self.epoch_one_loss}"
                f"\nFinal Loss [epoch {self.epochs}]={val_loss:.4f}"
                f"\nBest Loss [ step {self.best_step} epoch {(self.best_step + 1) / self.step_per_epoch}]={self.best_val_loss:.4f}"
            )

            # Final test — restore the best validation checkpoint first.
            # Ideally, read both of these from checkpoint using torch.load
            self.model.load_state_dict(self.best_state)
            print_header(text="Final Test")
            self.global_step = self.best_step
            self.run_eval(mode="Test")
