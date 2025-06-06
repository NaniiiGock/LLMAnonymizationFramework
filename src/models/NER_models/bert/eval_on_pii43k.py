import pandas as pd
import ast
import spacy

df = pd.read_csv("data/PII43k.csv")
df['tokens'] = df['Tokenised Filled Template'].apply(ast.literal_eval)
df['ner_tags'] = df['Tokens'].apply(ast.literal_eval)
nlp = spacy.load("en_core_web_sm")

def merge_spans(spans):
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: x[0])
    merged = [spans[0]]
    for start, end in spans[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged

def get_char_span_set(spans):
    char_set = set()
    for start, end in spans:
        char_set.update(range(start, end))
    return char_set

total_gt_chars = 0
total_overlap_chars = 0

i = 0
for _, row in df.iterrows():
    i += 1
    if i == 100:
        break
    text = row["Filled Template"]
    tokens = row['tokens']
    labels = row['ner_tags']
    char_spans_gt = []
    idx = 0
    for token, label in zip(tokens, labels):
        token_clean = token.replace("##", "")
        token_len = len(token_clean)
        if label != 'O':
            char_spans_gt.append((idx, idx + token_len))
        idx += token_len + 1 

    gt_spans = merge_spans(char_spans_gt)
    gt_chars = get_char_span_set(gt_spans)
    total_gt_chars += len(gt_chars)
    doc = nlp(text)
    pred_spans = merge_spans([(ent.start_char, ent.end_char) for ent in doc.ents])
    pred_chars = get_char_span_set(pred_spans)
    overlap = gt_chars.intersection(pred_chars)
    total_overlap_chars += len(overlap)

overlap_pct = (total_overlap_chars / total_gt_chars) * 100 if total_gt_chars > 0 else 0.0