from jsonargparse import CLI

import os
from transformers import (
    WhisperTokenizer,
    WhisperProcessor,
    T5Tokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from sl_whisper_config import SignLanguageWhisperConfig
from sl_whisper_modeling import SignLanguageWhisperForConditionalGeneration
from sl_t5_config import SignLanguageT5Config
from sl_t5_modeling import SignLanguageT5ForConditionalGeneration

from features_dataset import TranslationFeatures
from collators import (
    SignLanguageWhisperCollator,
    SignLanguageWhisperCollatorBicubic,
    SignLanguageT5Collator,
)
from eval import wer_and_bleu


def train(
    model_name: str,
    run_dir: str,
    model_config: str,
    model_checkpoint: str,
    tokenizer_checkpoint: str,
    processor_checkpoint: str,
    target_language: str,
    train_files: str,
    train_labels: str,
    valid_files: str,
    valid_labels: str,
    features_dir: str,
    expand_time: bool,
    freeze_model: bool,
    freeze_adapter: bool,
    force_target_language: bool,
    batch_size_per_gpu: int,
    accumulation_steps: int,
    learning_rate: float,
    warmup_steps: int,
    train_steps: int,
    fp16: bool,
    test_every: int,
    save_every: int,
    log_every: int,
    num_workers: int,
    auto_resume: bool,
    weight_decay: float,
    schedule: str,
    optimizer: str,
):
    """Main training loop for Whisper.

    Args:
        model_name: Model of choice. Currently supports 't5' or 'whisper'.
        run_dir: Directory from which to load/save model for training.
        model_config: Path to HuggingFace config file or to directory containing one.
        model_checkpoint: Path to local directory or model saved on HuggingFace.
        tokenizer_checkpoint: Path to local directory or model saved on HuggingFace.
        processor_checkpoint: Path to local directory or model saved on HuggingFace.
        target_language: Target written language for translation.
        train_files: Path to .tsv file containing the names of the original .mp4 files in the training set.
        train_labels: Path to .wrd file with the labels corresponding to train_files.
        valid_files: Path to .tsv file containing the names of the original .mp4 files in the validation set.
        valid_labels: Path to .wrd file with the labels corresponding to valid_files.
        features_dir: Path to the directory where the output of the feature extractor was saved.
        expand_time: Whether or not to use bicubic interpolation to get 50 frames per second.
        freeze_model: Whether or not to freeze the SLWhisper model.
        freeze_adapter: Whether or not to freeze the adapter inside SLWhisper. Overrides freeze_model on the adapter.
        force_target_language: Whether or not to set the model forced decoder ids to target_language.
    """
    os.makedirs(run_dir, exist_ok=True)
    os.environ["WANDB_PROJECT"] = f"sl-{model_name}"
    os.environ["WANDB_RUN_ID"] = os.path.basename(run_dir)
    os.environ["WANDB_ALLOW"] = "allow"

    if model_name == "whisper":
        config = SignLanguageWhisperConfig.from_pretrained(model_config)
        model = SignLanguageWhisperForConditionalGeneration.from_pretrained(
            model_checkpoint,
            config=config,
            cache_dir=os.path.join(run_dir, "cache"),
        )
        tokenizer = WhisperTokenizer.from_pretrained(
            tokenizer_checkpoint,
            language=target_language, task="transcribe",
        )
        processor = WhisperProcessor.from_pretrained(
            processor_checkpoint,
            language=target_language, task="transcribe",
        )

        if expand_time:
            data_collator = SignLanguageWhisperCollatorBicubic(processor)
        else:
            data_collator = SignLanguageWhisperCollator(processor)

        if freeze_model:
            for param in model.parameters():
                param.requires_grad = False
        else:
            for param in model.parameters():
                param.requires_grad = True
    
        if freeze_adapter:
            for param in model.model.encoder.adapter.parameters():
                param.requires_grad = False
        else:
            for param in model.model.encoder.adapter.parameters():
                param.requires_grad = True

        if force_target_language:
            model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
                language=target_language,
                task="transcribe"
            )
        else:
            model.config.forced_decoder_ids = None
            model.config.suppress_tokens = []
        
    elif model_name == "t5":
        config = SignLanguageT5Config.from_pretrained(model_config)
        model = SignLanguageT5ForConditionalGeneration.from_pretrained(
            model_checkpoint,
            config=config,
            cache_dir=os.path.join(run_dir, "cache"),
        )
        tokenizer = T5Tokenizer.from_pretrained(
            tokenizer_checkpoint,
        )

        if expand_time:
            raise NotImplementedError
        else:
            data_collator = SignLanguageT5Collator(model=model)

        if freeze_model:
            for param in model.parameters():
                param.requires_grad = False
        else:
            for param in model.parameters():
                param.requires_grad = True
    
        if freeze_adapter:
            for param in model.encoder.adapter.parameters():
                param.requires_grad = False
        else:
            for param in model.encoder.adapter.parameters():
                param.requires_grad = True

        if force_target_language:
            raise NotImplementedError

    else:
        raise NotImplementedError

    train_dataset = TranslationFeatures(
        train_files,
        train_labels,
        tokenizer,
        features_dir,
    )
    valid_dataset = TranslationFeatures(
        valid_files,
        valid_labels,
        tokenizer,
        features_dir,
    )

    compute_wer_and_bleu = lambda x: wer_and_bleu(x, tokenizer=tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir=run_dir,
        per_device_train_batch_size=batch_size_per_gpu,
        gradient_accumulation_steps=accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        max_steps=train_steps,
        fp16=fp16,
        eval_steps=test_every,
        save_steps=save_every,
        evaluation_strategy="steps",
        per_device_eval_batch_size=batch_size_per_gpu,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=log_every,
        load_best_model_at_end=True,
        metric_for_best_model="bleu-4",
        greater_is_better=False,
        lr_scheduler_type=schedule,
        report_to="wandb" if os.environ["LOCAL_RANK"] == 0 else None,
        dataloader_num_workers=num_workers,
        weight_decay=weight_decay,
        optim=optimizer,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_wer_and_bleu,
    )

    if auto_resume:
        try:
            trainer.train(resume_from_checkpoint=True)
        except ValueError:
            trainer.train()
        # trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()


if __name__ == "__main__":
    CLI(train)

