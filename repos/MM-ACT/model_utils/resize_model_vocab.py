import argparse, torch, torch.nn as nn
from transformers import AutoModelForCausalLM,AutoTokenizer
import json, os

def get_mod_attr(m, paths):
    for p in paths:
        try:
            obj = m
            for seg in p.split('.'):
                obj = getattr(obj, seg)
            return obj, p
        except AttributeError:
            pass
    return None, None

def set_by_path(m, path, new_obj):
    parent = m
    if "." in path:
        *ps, last = path.split(".")
        for seg in ps: parent = getattr(parent, seg)
        setattr(parent, last, new_obj)
    else:
        setattr(m, path, new_obj)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--num-new", type=int, default=2048)
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype="auto",
        low_cpu_mem_usage=True, device_map=None
    )
    model.to("cpu")
    cfg = model.config

    total_old = int(getattr(cfg, "embedding_size", getattr(cfg, "vocab_size")))
    llm_vocab = int(getattr(cfg, "llm_vocab_size", total_old))
    codebook_size = max(0, total_old - llm_vocab)
    print(f"[Info] total_old={total_old}, llm_vocab_size={llm_vocab}, codebook_size={codebook_size}")

    wte = model.get_input_embeddings() #output embedding 
    if wte is None:
        wte, wte_path = get_mod_attr(model, [
            "model.transformer.wte", "transformer.wte", "wte", "model.wte"
        ])
        assert wte is not None and hasattr(wte, "weight"), "Could not find embedding"
    else:
        wte_path = None

    out = model.get_output_embeddings()
    if out is None:
        out, out_path = get_mod_attr(model, [
            "lm_head", "ff_out", "model.lm_head", "model.ff_out",
            "transformer.lm_head", "transformer.ff_out",
        ])
    else:
        out_path = None

    assert wte.weight.shape[0] == total_old, "wte line not match config.embedding_size"
    d_model = wte.weight.shape[1]
    std = float(getattr(cfg, "initializer_range", getattr(cfg, "init_std", 0.02)))

    with torch.no_grad():
        new_wte = nn.Embedding(total_old + args.num_new, d_model, dtype=wte.weight.dtype)
        new_wte.weight[:total_old].copy_(wte.weight)
        nn.init.normal_(new_wte.weight[total_old:], mean=0.0, std=std)

        if wte_path is None and hasattr(model, "set_input_embeddings"):
            model.set_input_embeddings(new_wte)
        else:
            set_by_path(model, wte_path, new_wte)

        if out is not None and hasattr(out, "weight"):
            new_out = nn.Linear(out.in_features, total_old + args.num_new, bias=False, dtype=out.weight.dtype)
            new_out.weight[:total_old].copy_(out.weight)
            nn.init.normal_(new_out.weight[total_old:], mean=0.0, std=std)

            if out_path is None and hasattr(model, "set_output_embeddings"):
                model.set_output_embeddings(new_out)
            else:
                set_by_path(model, out_path, new_out)
        else:
            print("[Warn] use tied weight")

    total_new = total_old + args.num_new
    cfg.embedding_size = total_new
    cfg.vocab_size = total_new
    if hasattr(cfg, "new_vocab_size"): cfg.new_vocab_size = total_new
    setattr(cfg, "action_vocab_size", args.num_new)

    model.save_pretrained(args.out)
    print(f"[Done] {args.out} | total {total_old} -> {total_new}; "
          f"llm_vocab_size:{llm_vocab},codebook_size:{codebook_size}")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    tok.save_pretrained(args.out)

    tk_cfg_path = os.path.join(args.out, "tokenizer_config.json")
    with open(tk_cfg_path, "r", encoding="utf-8") as f:
        tk_cfg = json.load(f)
    tk_cfg["mmada_action_vocab_size"] = args.num_new
    tk_cfg["mmada_total_embedding_size"] = int(model.config.vocab_size)
    tk_cfg["llm_vocab_size"] = int(getattr(model.config, "llm_vocab_size", 0))
    with open(tk_cfg_path, "w", encoding="utf-8") as f:
        json.dump(tk_cfg, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
