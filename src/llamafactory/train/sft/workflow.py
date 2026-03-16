# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from typing import TYPE_CHECKING, List, Optional

from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer, calculate_length
from ...extras.constants import IGNORE_INDEX
from ...extras.misc import get_logits_processor
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer, expand_lm_head_for_mllama
from ..trainer_utils import create_modelcard_and_push
from .metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from .trainer import CustomSeq2SeqTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments

from llamafactory.extras.logging import get_logger
logger = get_logger(__name__)


def highlight_comment(comment, training_args: "Seq2SeqTrainingArguments"):
    if training_args.should_log:
        logger.info("*************************************")
        logger.info(comment)
        logger.info("*************************************")


def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    if data_args.calculate_length:
        data_args.cutoff_len = 1000000000
        
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    origin_train_dataset = data_args.dataset
    origin_eval_dataset = data_args.eval_dataset

    # load eval dataset
    eval_datasets = {}
    if origin_eval_dataset is not None:
        highlight_comment("Loading separate eval datasets...", training_args)
        data_args.dataset = None
        
        for eval_ds in origin_eval_dataset:
            data_args.eval_dataset = [eval_ds]
            eval_ds_module = get_dataset(template, model_args, data_args, training_args, stage="sft", disable_train_shuffling=finetuning_args.disable_train_shuffling, **tokenizer_module)
            eval_ds_name = eval_ds.split("__eval__")[-1]
            eval_datasets[eval_ds_name] = eval_ds_module["eval_dataset"]
        
        data_args.dataset = origin_train_dataset
        data_args.eval_dataset = origin_eval_dataset
        highlight_comment("Separate eval datasets loaded successfully.", training_args)
    
    # load train dataset
    dataset_module = {}
    if training_args.do_train and origin_train_dataset is not None:
        highlight_comment("Loading train && eval datasets...", training_args)
        data_args.eval_dataset = None
        
        dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", disable_train_shuffling=finetuning_args.disable_train_shuffling, **tokenizer_module)
        train_dataset = dataset_module["train_dataset"]
        
        data_args.eval_dataset = origin_eval_dataset
        highlight_comment("Train && eval datasets loaded successfully.", training_args)        
    else:
        train_dataset = None
    
    if "eval_dataset" not in dataset_module and len(eval_datasets) == 0:
        eval_datasets = None
    elif "eval_dataset" in dataset_module and len(eval_datasets) > 0:
        logger.warning("Provides multiple ways to construct evaluation datasets, with the default being the use of separate multiple eval datasets.")
    elif "eval_dataset" not in dataset_module and len(eval_datasets) > 0:
        logger.info("Using separate multiple eval datasets.")
    else:
        eval_datasets["eval"] = [dataset_module["eval_dataset"]]
    
    # load prediction dataset
    if training_args.do_predict and data_args.predict_dataset is not None:
        highlight_comment("Loading prediction datasets...", training_args)
        data_args.dataset = None
        data_args.eval_dataset = data_args.predict_dataset

        predict_dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", disable_train_shuffling=finetuning_args.disable_train_shuffling, **tokenizer_module)
        predict_dataset = predict_dataset_module["eval_dataset"]
        
        data_args.dataset = origin_train_dataset
        data_args.eval_dataset = origin_eval_dataset

        highlight_comment("Prediction datasets loaded successfully.", training_args)
    else:
        predict_dataset = None
    
    # calculate sample length
    if data_args.calculate_length:
        with training_args.main_process_first(desc="Calculating length..."):
            if training_args.should_log:
                calculate_length(dataset_module["train_dataset"], training_args.output_dir)
        return
    
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    
    # To enable the mllama model to be compatible with repetition_penalty during inference.
    expand_lm_head_for_mllama(model, tokenizer, training_args, data_args)

    # Huawei: This line of code is used to address the issue of slow data loading.
    import gc
    gc.set_threshold(700, 10, 10000)
    
    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    model_type = getattr(model.config, "model_type", None)
    if model_type == "mllama":
        mllama_num_tiles = data_args.mllama_num_tiles
        if mllama_num_tiles is not None:
            assert mllama_num_tiles == 1, "mllama_num_tiles can only be reset to 1!"
    else:
        mllama_num_tiles = None

    data_collator = SFTDataCollatorWith4DAttentionMask(
        is_training=training_args.do_train,
        template=template,
        mllama_num_tiles=mllama_num_tiles, # for mllama
        model_type=model_type,
        add_image_token_in_input_ids=model_args.add_image_token_in_input_ids,
        add_video_position_encoding=model_args.mllama_add_video_position_encoding,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False  # important for multimodal dataset

    # Metric utils
    metric_module = {}
    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        # **dataset_module,
        **tokenizer_module,
        **metric_module,
    )

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "eval_accuracy"])

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        if training_args.predict_with_generate:  # eval_loss will be wrong if predict_with_generate is enabled
            metrics.pop("eval_loss", None)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        with torch.no_grad():
            predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict", **gen_kwargs)
        if training_args.predict_with_generate:  # predict_loss will be wrong if predict_with_generate is enabled
            predict_results.metrics.pop("predict_loss", None)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(predict_dataset, predict_results, tokenizer)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
