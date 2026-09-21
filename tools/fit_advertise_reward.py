"""Fit a pairwise logistic reward on independently selected ad candidates."""
import argparse, json, math, re
from pathlib import Path

FEATURES = ["bias", "attr_coverage", "length_quality", "char_repeat", "token_repeat", "eos_complete", "malformed"]
BAD = re.compile(r"�|嘿嘿|sos|mrang|不锈钢|遗传|危机|透视效果|防晒外套|冬天的冬天")

def feats(text, attrs):
    text = text or ""
    # candidates.jsonl stores attributes as pairs such as ["类型", "裤"]
    keys = []
    for x in (attrs or []):
        if isinstance(x, list):
            keys.extend(str(v) for v in x)
        else:
            keys.append(str(x))
    cov = sum(k in text for k in keys) / max(1, len(keys))
    n = len(text)
    length = 1.0 if 80 <= n <= 260 else max(0.0, 1 - abs(n-170)/170)
    chars = [c for c in text if not c.isspace()]
    cr = 1 - len(set(chars))/max(1,len(chars))
    toks = re.findall(r"[\u4e00-\u9fff]|[A-Za-z]+|\d+", text)
    tr = 1 - len(set(toks))/max(1,len(toks))
    eos = 0.0 if BAD.search(text) else 1.0
    malformed = 1.0 if BAD.search(text) else 0.0
    return [1.0, cov, length, cr, tr, eos, malformed]

def sigmoid(x):
    return 1/(1+math.exp(-max(-40,min(40,x))))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--candidates', type=Path, default=Path('grpo_candidates/pilot200/candidates.jsonl'))
    ap.add_argument('--labels', type=Path, default=Path('grpo_candidates/pilot200/ai_preference_labels_independent.jsonl'))
    ap.add_argument('--output', type=Path, default=Path('grpo_candidates/pilot200/fitted_reward.json'))
    ap.add_argument('--epochs', type=int, default=3000); ap.add_argument('--lr', type=float, default=.08); ap.add_argument('--l2', type=float, default=.1)
    a=ap.parse_args(); groups={json.loads(x)['group_id']:json.loads(x) for x in a.candidates.read_text(encoding='utf-8-sig').splitlines() if x.strip()}
    labels=[json.loads(x) for x in a.labels.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
    pairs=[]
    for l in labels:
        g=groups[l['group_id']]; cs={c['candidate_id']:c for c in g['candidates']}; ch=cs[l['chosen_candidate_id']]
        for rid in l['rejected_candidate_ids']: pairs.append((feats(ch['text'],g.get('attributes')),feats(cs[rid]['text'],g.get('attributes'))))
    cut=max(1,int(.8*len(pairs))); w=[0.0]*len(FEATURES)
    for _ in range(a.epochs):
        grad=[0.0]*len(w)
        for fa,fb in pairs[:cut]:
            d=[x-y for x,y in zip(fa,fb)]; p=sigmoid(sum(x*y for x,y in zip(w,d)))
            for i,x in enumerate(d): grad[i]+=(1-p)*x
        for i in range(len(w)): w[i]+=a.lr*(grad[i]/cut-a.l2*w[i]); w[i]=max(-10,min(10,w[i]))
    def acc(ps): return sum(sigmoid(sum(x*y for x,y in zip(w,[u-v for u,v in zip(fa,fb)])))>.5 for fa,fb in ps)/max(1,len(ps))
    out={'features':FEATURES,'weights':w,'pairs':len(pairs),'train_pairs':cut,'val_pairs':len(pairs)-cut,'train_accuracy':acc(pairs[:cut]),'val_accuracy':acc(pairs[cut:]),'fit_policy':'pairwise logistic with L2; content features only'}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
