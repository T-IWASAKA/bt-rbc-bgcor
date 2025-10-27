# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

trainer class for training a model

@author: tadahaya
"""
import torch
import numpy as np

from .utils import save_experiment, save_checkpoint

class Trainer:
    def __init__(self, config, model, optimizer, scheduler, loss_fn, exp_name, device, wandb_run):
        self.config = config
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device
        self.wandb_run = wandb_run


    def train(self, trainloader, testloader, save_model_evry_n_epochs=0):
        """
        train the model for the specified number of epochs.
        
        """
        # configの確認
        config = self.config
        assert config["hidden_size"] % config["num_attention_heads"] == 0
        assert config["intermediate_size"] == 4 * config["hidden_size"]
        assert config["image_size"] % config["patch_size"] == 0
        # keep track of the losses and accuracies
        base_dir = "/workspace/data" #後で追加
        train_losses, test_losses = [], []
        
        # prep_earlystop
        patience = 10 # 性能が改善しないのを何エポックまで待つか
        patience_counter = 0 # カウンター
        best_loss = np.inf # 過去最良の損失を保存する変数（最初は無限大に設定）
        best_epoch = 0

        # training
        for i in range(config["epochs"]):
            train_loss = self.train_epoch(trainloader)
            test_loss = self.evaluate(testloader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()

            print(
                f"Epoch: {i + 1}, Train_loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Learning Rate: {current_lr}"
                )
            
            # ▼▼▼ エポックごとのメトリクスを記録 ▼▼▼
            if self.wandb_run:
                self.wandb_run.log({
                    "epoch": i + 1,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "learning_rate": current_lr[0]
                })
            # ▲▲▲ 変更ここまで ▲▲▲

            if save_model_evry_n_epochs > 0 and (i + 1) % save_model_evry_n_epochs == 0 and i + 1 != config["epochs"]:
                print("> Save checkpoint at epoch", i + 1)
                save_checkpoint(self.exp_name, self.model.backbone.net, i + 1, base_dir)
            
            if test_loss < best_loss:
                best_loss = test_loss
                save_checkpoint(self.exp_name, self.model.backbone.net, i + 1, base_dir) #おそらくBTの場合はバックボーンだけを保存した方がいい
                patience_counter = 0 # カウンターをリセット
                best_epoch = i + 1

                # ▼▼▼ Best ScoreをWandBに記録 ▼▼▼
                if self.wandb_run:
                    # best_scoreをWandBのサマリーに保存すると、後で比較しやすくなる
                    self.wandb_run.summary["best_test_loss"] = best_loss
                    self.wandb_run.summary["best_epoch"] = best_epoch
                # ▲▲▲ 変更ここまで ▲▲▲

            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered.")
                print("best epoch = ", str(best_epoch))
                save_experiment(
                    self.exp_name, base_dir, config, self.model, train_losses, test_losses
                    )
                break # 学習ループを抜ける
        
        self.best_epoch = best_epoch


    def train_epoch(self, trainloader):
        """ train the model for one epoch """
        self.model.train()
        total_loss = 0
        for batch in trainloader:
            batch = [x.to(self.device) for x in batch] # batchをdeviceへ
            if len(batch) == 2:
                y1, y2 = batch
                # 勾配を初期化
                self.optimizer.zero_grad()
                # forward / loss
                loss = self.model(y1, y2) # attentionもNoneで返るので
            elif len(batch) == 4:
                y1, y2, b1, b2 = batch
                # 勾配を初期化
                self.optimizer.zero_grad()
                # forward / loss
                loss = self.model(y1, y2, b1, b2) # attentionもNoneで返るので
            else:
                raise ValueError('!! This is unexpected data format !!')
            # backpropagation
            loss.backward()
            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) # 外部からさわれるように
            # パラメータ更新
            self.optimizer.step()
            total_loss += loss.item() # 本当なbatchsizeによる補正があったほうがいいかも 250714

        return total_loss / len(trainloader.dataset) # 全データセットのうちのいくらかという比率になっている
    

    @torch.no_grad()
    def evaluate(self, testloader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in testloader:
                batch = [x.to(self.device) for x in batch] # batchをdeviceへ
                if len(batch) == 2:
                    y1, y2 = batch
                    loss = self.model(y1, y2)
                elif len(batch) == 4:
                    y1, y2, b1, b2 = batch
                    # loss
                    loss = self.model(y1, y2, b1, b2)
                total_loss += loss.item()
        avg_loss = total_loss / len(testloader.dataset)
        return avg_loss