import json

from evaluate import load_eval_data, load_model

crf_model = load_model("crf", "checkpoints/crf_window2")
bilstm_model = load_model("bilstm", "checkpoints/bilstm_1.00")

sentences, gold_labels = load_eval_data("test")
crf_predictions = crf_model.predict(sentences)
bilstm_predictions = bilstm_model.predict(
    sentences, bilstm_model.vocab
)

for filename, predictions in [
    ("crf_predictions.jsonl", crf_predictions),
    ("bilstm_predictions.jsonl", bilstm_predictions),
]:
    with open(filename, "w", encoding="utf-8") as output_file:
        for index, (tokens, slots) in enumerate(zip(sentences, predictions)):
            json.dump(
                {"index": index, "tokens": tokens, "slots": slots},
                output_file,
            )
            output_file.write("\n")

crf_only, bilstm_only, both_wrong = [], [], []

for index, (gold, crf, bilstm) in enumerate(
    zip(gold_labels, crf_predictions, bilstm_predictions)
):
    crf_wrong = crf != gold
    bilstm_wrong = bilstm != gold

    if crf_wrong and not bilstm_wrong:
        crf_only.append(index)
    elif bilstm_wrong and not crf_wrong:
        bilstm_only.append(index)
    elif crf_wrong and bilstm_wrong:
        both_wrong.append(index)

selected = crf_only[:5] + bilstm_only[:5]
selected += both_wrong[: max(0, 10 - len(selected))]

for index in selected:
    print(f"\nTest example {index}")
    print("Tokens: ", " ".join(sentences[index]))
    print("Gold:   ", " ".join(gold_labels[index]))
    print("CRF:    ", " ".join(crf_predictions[index]))
    print("BiLSTM: ", " ".join(bilstm_predictions[index]))