"""JSONL stdin/stdout bridge for the external reward model.
Run with the evaluation environment's Python interpreter."""
import argparse, json, sys, os, torch, contextlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer.train_grpo import load_reward_components, get_checked_reward_score

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model-path',required=True); p.add_argument('--device',default='cuda:0'); a=p.parse_args()
    # stdout is a machine-readable JSONL protocol; route model-loader logs away.
    with contextlib.redirect_stdout(sys.stderr):
        model,tok=load_reward_components(a.model_path,a.device)
    for line in sys.stdin:
        try:
            req=json.loads(line); value=get_checked_reward_score(model,tok,req['conversation'])
            if isinstance(value,torch.Tensor): value=value.detach().float().cpu().item()
            print(json.dumps({'score':float(value)}),flush=True)
        except Exception as e:
            print(json.dumps({'error':f'{type(e).__name__}: {e}'}),flush=True)
if __name__=='__main__': main()
