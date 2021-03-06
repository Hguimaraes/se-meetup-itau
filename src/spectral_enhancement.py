import sys
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml

from denoiser.dataset import prep_librispeech
from denoiser.dataset import create_datasets
from denoiser.brain import SEBrain


def main(hparams, hparams_file, run_opts, overrides):
    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Data prep to run on the main thread
    sb.utils.distributed.run_on_main(
        prep_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_json_train": hparams["train_annotation"],
            "save_json_valid": hparams["valid_annotation"],
            "train_folder": hparams["train_folder"],
            "valid_folder": hparams["valid_folder"],
        },
    )

    # Create dataset objects "train" and "valid"
    datasets = create_datasets(hparams)

    # Initialize the Trainer
    brain = SEBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Call the training loop
    brain.fit(
        epoch_counter=brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_options"],
        valid_loader_kwargs=hparams["valid_dataloader_options"],
    )

    # Load best checkpoint for evaluation
    test_stats = brain.evaluate(
        test_set=datasets["valid"],
        max_key="pesq",
        test_loader_kwargs=hparams["valid_dataloader_options"],
    )


if __name__ == "__main__":
    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as f:
        hparams = load_hyperpyyaml(f, overrides)
    
    main(hparams, hparams_file, run_opts, overrides)