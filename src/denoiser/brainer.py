import os
import torch
import torchaudio
from pesq import pesq
import speechbrain as sb
from speechbrain.nnet.loss.stoi_loss import stoi_loss
from speechbrain.utils.metric_stats import MetricStats

class SEBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the enhanced output."""
        batch = batch.to(self.device)
        noisy_wavs, lens = batch.noisy_sig
        noisy_feats = self.compute_feats(noisy_wavs)

        # mask with "signal approximation (SA)"
        mask = self.modules.model(noisy_feats)
        mask = torch.squeeze(mask, 2)
        predict_spec = torch.mul(mask, noisy_feats)

        # Also return predicted wav
        predict_wav = self.hparams.resynth(
            torch.expm1(predict_spec), noisy_wavs
        )

        return predict_spec, predict_wav


    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs"""
        predict_spec, predict_wav = predictions
        clean_wavs, lens = batch.clean_sig

        if getattr(self.hparams, "waveform_target", False):
            loss = self.hparams.compute_cost(predict_wav, clean_wavs, lens)
            self.loss_metric.append(
                batch.id, predict_wav, clean_wavs, lens, reduction="batch"
            )
        else:
            clean_spec = self.compute_feats(clean_wavs)
            loss = self.hparams.compute_cost(predict_spec, clean_spec, lens)
            self.loss_metric.append(
                batch.id, predict_spec, clean_spec, lens, reduction="batch"
            )

        if stage != sb.Stage.TRAIN:

            # Evaluate speech quality/intelligibility
            self.stoi_metric.append(
                batch.id, predict_wav, clean_wavs, lens, reduction="batch"
            )
            self.pesq_metric.append(
                batch.id, predict=predict_wav, target=clean_wavs, lengths=lens
            )

            # Write wavs to file
            if stage == sb.Stage.TEST:
                lens = lens * clean_wavs.shape[1]
                for name, pred_wav, length in zip(batch.id, predict_wav, lens):
                    name += ".wav"
                    enhance_path = os.path.join(
                        self.hparams.enhanced_folder, name
                    )
                    torchaudio.save(
                        enhance_path,
                        torch.unsqueeze(pred_wav[: int(length)].cpu(), 0),
                        16000,
                    )

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch"""
        self.loss_metric = MetricStats(metric=self.hparams.compute_cost)
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
            if self.hparams.use_tensorboard:
                valid_stats = {
                    "loss": self.loss_metric.scores,
                    "stoi": self.stoi_metric.scores,
                    "pesq": self.pesq_metric.scores,
                }
                self.hparams.tensorboard_train_logger.log_stats(
                    {"Epoch": epoch}, self.train_stats, valid_stats
                )
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )
            self.checkpointer.save_and_keep_only(meta=stats, max_keys=["pesq"])

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

