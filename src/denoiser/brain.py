import os
import torch
import torchaudio
from pesq import pesq
import speechbrain as sb
from speechbrain.nnet.loss.stoi_loss import stoi_loss
from speechbrain.utils.metric_stats import MetricStats

class SEBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        noisy_wavs, lens = batch.predictor # B, C, L

        return self.modules.model(noisy_wavs)


    def compute_objectives(self, predictions, batch, stage):
        # Get clean targets
        targets, lens = batch.target # B, L, C
        loss = self.modules.loss(predictions, targets, lens)

        if stage != sb.Stage.TRAIN:
            # Evaluate speech quality/intelligibility
            self.stoi_metric.append(
                batch.id, predictions, targets, lens, reduction="batch"
            )

            predictions = predictions.squeeze(-1)
            targets = targets.squeeze(-1)

            self.pesq_metric.append(
                batch.id,
                predict=predictions.squeeze(-1),
                target=targets.squeeze(-1),
                lengths=lens
            )

            # Write wavs to file
            if stage == sb.Stage.TEST:
                if not os.path.exists(self.hparams.audio_result):
                    os.mkdir(self.hparams.audio_result)

                lens = lens * targets.shape[1]
                for name, pred_wav, length in zip(batch.id, predictions, lens):
                    name += ".wav"
                    enhance_path = os.path.join(
                        self.hparams.audio_result, name
                    )
                    torchaudio.save(
                        enhance_path,
                        torch.unsqueeze(pred_wav[: int(length)].cpu(), 0),
                        self.hparams.sample_rate,
                    )

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch"""
        self.loss_metric = MetricStats(metric=self.modules.loss)
        self.stoi_metric = MetricStats(metric=stoi_loss)

        # Define function taking (prediction, target) for parallel eval
        def pesq_eval(pred_wav, target_wav):
            """Computes the PESQ evaluation metric"""
            return pesq(
                fs=16000,
                ref=target_wav.numpy(),
                deg=pred_wav.numpy(),
                mode="wb",
            )

        if stage != sb.Stage.TRAIN:
            self.pesq_metric = MetricStats(
                metric=pesq_eval, n_jobs=1, batch_eval=False
            )

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            self.train_stats = {"loss": self.loss_metric.scores}
        else:
            stats = {
                "loss": stage_loss,
                "pesq": self.pesq_metric.summarize("average"),
                "stoi": -self.stoi_metric.summarize("average"),
            }

        if stage == sb.Stage.VALID:
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )
            self.checkpointer.save_and_keep_only(meta=stats, max_keys=["pesq"])
