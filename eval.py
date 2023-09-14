import evaluate
from transformers import WhisperTokenizer


def wer_and_bleu(pred, tokenizer):
    wer = evaluate.load("wer")
    bleu = evaluate.load("bleu")
    
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer_val = 100 * wer.compute(predictions=pred_str, references=label_str)
    b1_val = 100 * bleu.compute(predictions=pred_str, references=label_str, max_order=1)["bleu"]
    b2_val = 100 * bleu.compute(predictions=pred_str, references=label_str, max_order=2)["bleu"]
    b3_val = 100 * bleu.compute(predictions=pred_str, references=label_str, max_order=3)["bleu"]
    b4_val = 100 * bleu.compute(predictions=pred_str, references=label_str, max_order=4)["bleu"]

    return {
        "wer": wer_val,
        "bleu-1": b1_val,
        "bleu-2": b2_val,
        "bleu-3": b3_val,
        "bleu-4": b4_val,
    }

