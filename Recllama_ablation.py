import os, json, csv, ast, math, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gensim.models import KeyedVectors
from sklearn.ensemble import RandomForestClassifier
import joblib

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI

from reasoner.reasoner import Reasoner


INPUT_CSV      = "questions.csv"            
TEXT_COL       = "patient_description"


PROC_LEXICON   = "D_ICD_PROCEDURES.csv"    
DIAG_LEXICON   = ""                         


CE_DIR         = "proc_linker_biobert"     


NODE2VEC       = "node2vec_embeddings.txt"
RF_PATH        = "rf_model.pkl"

# LLM endpoint (DeepSeek/OpenAI-compatible)
MODEL_NAME     = "deepseek-r1"
OPENAI_BASE    = os.getenv("OPENAI_BASE_URL", "https://api.lkeap.cloud.tencent.com/v1")
OPENAI_KEY     = os.getenv("OPENAI_API_KEY", "")


def parse_list(s):
    try:
        v = ast.literal_eval(str(s))
        return [str(x) for x in v]
    except Exception:
        return []

def safe_json(x):
    try:
        return json.dumps(x, ensure_ascii=False)
    except TypeError:
        return json.dumps([str(z) for z in x], ensure_ascii=False)

# -----------------------
# Lexicons
# -----------------------
def load_code_lexicon(path, code_names=("code","icd9","icd9_code"),
                      title_names=("short_title","long_title","title","description")):
    """
    Generic loader for code->title CSVs (procedures or diagnoses).
    Returns (codes_list, title_by_code) with dotless codes.
    """
    if not path or not Path(path).exists():
        return [], {}

    df = pd.read_csv(path, encoding="utf-8-sig")
    code_col  = next((c for c in df.columns if c.lower() in set(code_names)), None)
    title_col = next((c for c in df.columns if c.lower() in set(title_names)), None)
    if not code_col or not title_col:
        raise SystemExit(f"[ERROR] Need code + title columns in {path}")
    df = df[[code_col, title_col]].rename(columns={code_col:"code", title_col:"title"})
    df["code_nodot"] = df["code"].astype(str).str.replace(".","", regex=False)
    df = df.drop_duplicates(subset=["code_nodot"]).reset_index(drop=True)
    codes = df["code_nodot"].tolist()
    title_by_code = dict(zip(df["code_nodot"], df["title"]))
    return codes, title_by_code


def load_ce(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir)
    if torch.cuda.device_count() > 1:
        mdl = torch.nn.DataParallel(mdl)
    mdl.to(device)
    mdl.eval()
    return tok, mdl, device

@torch.no_grad()
def ce_predict_procedures(text, tokenizer, model, device, codes, title_by_code,
                          topk=5, score_threshold=0.5, batch_size=64, max_length=256):
    if not text or not codes:
        return []
    lefts, rights, meta = [], [], []
    for c in codes:
        lefts.append(text)
        rights.append(f"ICD9:{c} {title_by_code.get(c,'')}")
        meta.append(c)

    enc = tokenizer(lefts, rights, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
    enc = {k: v.to(device) for k,v in enc.items()}

    logits_all = []
    for s in range(0, enc["input_ids"].size(0), batch_size):
        sl = slice(s, s+batch_size)
        batch = {k: v[sl] for k,v in enc.items()}
        out = model(**batch)
        logits_all.append(out.logits.detach().cpu())
    logits = torch.cat(logits_all, dim=0)
    probs  = torch.softmax(logits, dim=1)[:,1].numpy()

    scored = [{"code": c, "title": title_by_code.get(c,""), "score": float(p)}
              for c,p in zip(meta, probs)]
    scored = [x for x in scored if x["score"] >= score_threshold]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:topk]

# -----------------------
# LLM extractor (fallback / ablation)
# -----------------------
def llm_map_procedures(client, text, codes, title_by_code, topk=5):
    if not client or not text:
        return []
    # keep prompt small
    rows = [f"{c}\t{title_by_code.get(c,'')}" for c in codes[:2000]]
    kb = "\n".join(rows)
    prompt = f"""
You are a medical coding assistant. Given the patient's description, pick the most relevant ICD-9 procedure codes.
Return a JSON list of objects: [{{"code":"<code>", "score": <0..1>}}] sorted by score desc. Use only these codes:

{kb}

Patient description:
{text}
"""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role":"user","content":prompt}],
            temperature=0.0,
            response_format="json_object",
        )
        data = json.loads(resp.choices[0].message.content)
        out = []
        for item in data:
            c = str(item.get("code","")).replace(".","")
            if c in title_by_code:
                out.append({"code": c, "title": title_by_code[c], "score": float(item.get("score", 0.5))})
        out.sort(key=lambda x: x["score"], reverse=True)
        return out[:topk]
    except Exception as e:
        print(f"[WARN] LLM mapping failed: {e}")
        return []

# -----------------------
# Alignment: procedures -> proteins
# -----------------------
def pair_rf_predict(procedures, w2v, rf, topN=50):
    """RF alignment: select proteins predicted by RF across procs."""
    if not procedures:
        return []
    all_nodes = list(w2v.key_to_index)
    cand_prot = [n for n in all_nodes if n not in procedures]
    prot_scores = []
    for prot in cand_prot:
        s = 0
        for proc in procedures:
            if proc in w2v and prot in w2v:
                feat = np.concatenate((w2v[proc], w2v[prot])).reshape(1,-1)
                s += int(rf.predict(feat)[0])
        if s > 0:
            prot_scores.append((prot, s))
    prot_scores.sort(key=lambda x: x[1], reverse=True)
    return [p for p,_ in prot_scores[:topN]]

def pair_cosine_predict(procedures, w2v, topN=50):
    """Ablation (no RF): cosine similarity fallback."""
    if not procedures:
        return []
    all_nodes = list(w2v.key_to_index)
    cand_prot = [n for n in all_nodes if n not in procedures]
    def cossim(a,b):
        return float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9))
    prot_scores = []
    for prot in cand_prot:
        s = 0.0
        if prot not in w2v: 
            continue
        vp = w2v[prot]
        for proc in procedures:
            if proc in w2v:
                s += cossim(w2v[proc], vp)
        if s != 0:
            prot_scores.append((prot, s))
    prot_scores.sort(key=lambda x: x[1], reverse=True)
    return [p for p,_ in prot_scores[:topN]]

# -----------------------
# Reasoner
# -----------------------
def run_reasoner(protein_list, reasoner):
    if not protein_list or reasoner is None:
        return [], []
    results, inter = reasoner.reason(protein_list)

    def c_val(tv):
        try:
            parts = str(tv).split(";")
            return float(parts[1].split("%")[0])
        except Exception:
            return 0.0

    sr = sorted(results, key=lambda x: c_val(x[1]), reverse=True)
    si = sorted(inter,   key=lambda x: c_val(x[1]), reverse=True)
    return sr, si

# -----------------------
# Final diagnosis extraction + explanation
# -----------------------
def extract_top_diagnoses(reasoner_results, max_n=5):
    """
    Parse diagnoses from reasoner tuples like:
      [('4254', <TruthValue: %1.00;0.99% (k=1)>), ...]
    Returns a list of (code_or_label, confidence_float).
    """
    out = []
    for item in reasoner_results[:max_n]:
        try:
            label = str(item[0]).strip()
            tv = str(item[1])
            # extract 'c' from '%1.00;0.99% (k=1)'
            c = 0.0
            parts = tv.split(";")
            if len(parts) > 1:
                c = float(parts[1].split("%")[0])
            out.append((label, c))
        except Exception:
            continue
    return out

def llm_explain(client, text, procs, prots, diagnoses, diag_title_map=None):
    """
    Build a friendly explanation. If LLM unavailable, fall back to a template.
    """
    # Pretty candidates
    proc_str = ", ".join([f"{c}" for c in procs]) if procs else "None"
    prot_str = ", ".join(prots[:10]) if prots else "None"
    if diag_title_map:
        diag_str = ", ".join([f"{d} ({diag_title_map.get(d,d)})" for d,_ in diagnoses])
    else:
        diag_str = ", ".join([d for d,_ in diagnoses]) if diagnoses else "None"

    if client is None:
        # Template fallback
        return (
            "Possible diagnoses inferred from structured reasoning: "
            f"{diag_str}. "
            "These hypotheses were derived by mapping the patient description to procedures, "
            "aligning them to molecular evidence, and running a rule-based reasoning engine. "
            "Consider clinical validation with a physician."
        )

    # LLM prompt
    prompt = f"""
You are a clinical assistant. Given the patient description and the structured pipeline outputs,
summarize likely diagnoses and explain the reasoning briefly in lay terms. Be concise and actionable.

Patient description:
{text}

Candidate procedures (ICD-9): {proc_str}
Aligned molecular evidence (top proteins): {prot_str}
Reasoner top diagnoses: {diag_str}

Return a short paragraph suitable for a patient summary.
"""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role":"user","content":prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return (
            f"(LLM unavailable) Possible diagnoses: {diag_str}. "
            "Derived via procedures → proteins → rule-based reasoning."
        )

# -----------------------
# Metrics
# -----------------------
def setify(x):
    return set(x) if isinstance(x, (list,tuple,set)) else set([x])

def prf_at_k(pred, gold, k):
    predk = pred[:k]
    gset = setify(gold)
    if len(predk) == 0: return (0.0, 0.0, 0.0)
    tp = len([c for c in predk if c in gset])
    prec = tp / max(len(predk), 1)
    rec  = tp / max(len(gset), 1) if gset else 0.0
    f1   = 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)
    return (prec, rec, f1)

def dcg(labels):
    s = 0.0
    for i, y in enumerate(labels, 1):
        s += (1.0 if y>0 else 0.0) / math.log2(i+1)
    return s

def rank_metrics(pred_order, gold_set, ks=(1,3,5,10)):
    labs = [1 if p in gold_set else 0 for p in pred_order]
    out = {}
    # MAP
    num_pos = sum(labs)
    ap = 0.0
    if num_pos>0:
        c = 0; s = 0.0
        for i, y in enumerate(labs, 1):
            if y==1:
                c += 1
                s += c / i
        ap = s / num_pos
    # MRR
    mrr = 0.0
    for i, y in enumerate(labs,1):
        if y==1:
            mrr = 1.0/i
            break
    # NDCG@k
    for k in ks:
        top = labs[:k]
        nd = 0.0
        idcg = dcg(sorted(labs, reverse=True)[:k])
        if idcg>0: nd = dcg(top)/idcg
        out[f"p@{k}"] = sum(top)/max(k,1)
        out[f"ndcg@{k}"] = nd
    out["MAP"] = ap
    out["MRR"] = mrr
    return out

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", default=INPUT_CSV)
    ap.add_argument("--text_col",  default=TEXT_COL)

    ap.add_argument("--proc_lexicon", default=PROC_LEXICON)
    ap.add_argument("--diag_lexicon", default=DIAG_LEXICON)

    ap.add_argument("--ce_dir", default=CE_DIR)
    ap.add_argument("--node2vec", default=NODE2VEC)
    ap.add_argument("--rf_path",  default=RF_PATH)

    ap.add_argument("--use_llm", action="store_true", help="Use LLM extractor instead of cross-encoder")
    ap.add_argument("--disable_rf", action="store_true", help="Disable RF alignment (use cosine baseline)")
    ap.add_argument("--disable_reasoner", action="store_true", help="Skip reasoning stage (still produce fallback diagnosis/explanation)")
    ap.add_argument("--topk_proc", type=int, default=5)
    ap.add_argument("--proc_thresh", type=float, default=0.5)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    # LLM client (optional)
    client = OpenAI(api_key=OPENAI_KEY, base_url=OPENAI_BASE) if (OPENAI_KEY) else None

    # Load lexicons
    proc_codes, proc_title_by_code = load_code_lexicon(args.proc_lexicon)
    diag_codes, diag_title_by_code = load_code_lexicon(args.diag_lexicon) if args.diag_lexicon else ([], {})

    # Load cross-encoder (if not using LLM)
    if not args.use_llm:
        if not Path(args.ce_dir).exists():
            raise SystemExit(f"[ERROR] Cross-encoder dir not found: {args.ce_dir}")
        ce_tok, ce_mdl, ce_dev = load_ce(args.ce_dir)
    else:
        ce_tok = ce_mdl = ce_dev = None

    # Load RF + node2vec + reasoner
    w2v = KeyedVectors.load_word2vec_format(args.node2vec)
    rf  = None if args.disable_rf else joblib.load(args.rf_path)
    reasoner = None if args.disable_reasoner else Reasoner(5)

    # Read data
    df = pd.read_csv(args.input_csv)
    if args.text_col not in df.columns:
        raise SystemExit(f"[ERROR] Input must have '{args.text_col}'")

    # Optional gold labels for metrics
    gold_proc_col = next((c for c in df.columns if c.lower() in {"procedures","gold_procedures","gold_proc"}), None)
    gold_prot_col = next((c for c in df.columns if c.lower() in {"proteins","gold_proteins","gold_biomarkers"}), None)

    # Prepare output cols (append to input)
    out = pd.DataFrame({
        "icd9_predictions_json": "",
        "icd9_codes": "",
        "reasoner_results": "",
        "reasoner_intermediates": "",
        "final_diagnoses_json": "",
        "final_diagnosis_titles": "",
        "final_explanation": "",
        "ablation_note": "",
    }, index=df.index)

    # For metrics accumulation
    proc_metrics = {k: [] for k in ["p@1","r@1","f1@1","p@3","r@3","f1@3","p@5","r@5","f1@5"]}
    prot_metrics = {k: [] for k in ["p@1","p@3","p@5","p@10","ndcg@1","ndcg@3","ndcg@5","ndcg@10","MAP","MRR"]}

    ablation_note = []
    ablation_note.append("Extractor=LLM" if args.use_llm else "Extractor=CE")
    ablation_note.append("Alignment=cosine" if args.disable_rf else "Alignment=RF")
    ablation_note.append("Reasoner=OFF" if args.disable_reasoner else "Reasoner=ON")
    note_str = "; ".join(ablation_note)

    for i, row in df.iterrows():
        text = str(row[args.text_col]).strip()

        # --- Stage A: Extract procedures ---
        if args.use_llm:
            scored = llm_map_procedures(client, text, proc_codes, proc_title_by_code, topk=args.topk_proc)
        else:
            scored = ce_predict_procedures(
                text, ce_tok, ce_mdl, ce_dev, proc_codes, proc_title_by_code,
                topk=args.topk_proc, score_threshold=args.proc_thresh
            )
        out.at[i,"icd9_predictions_json"] = json.dumps(scored, ensure_ascii=False)
        pred_codes = [s["code"] for s in scored]
        out.at[i,"icd9_codes"] = ";".join(pred_codes)

        # --- Stage B: Align to proteins (RF or cosine) ---
        if len(pred_codes) > 0:
            if rf is not None:
                prot_rank = pair_rf_predict(pred_codes, w2v, rf, topN=50)
            else:
                prot_rank = pair_cosine_predict(pred_codes, w2v, topN=50)
        else:
            prot_rank = []

        # --- Stage C: Reasoner (can be disabled) ---
        if reasoner is not None and len(prot_rank)>0:
            res, inter = run_reasoner(prot_rank, reasoner)
        else:
            res, inter = [], []

        out.at[i,"reasoner_results"] = safe_json(res)
        out.at[i,"reasoner_intermediates"] = safe_json(inter)

        
        scored_for_fallback = scored[:]  

        if res:
            # Reasoner path: parse diagnoses + confidence
            diags = extract_top_diagnoses(res, max_n=5)
            final_diag_codes = [d for d, _ in diags if d.isdigit()]
            final_diag_labels = [d for d, _ in diags if not d.isdigit()]
            out.at[i, "final_diagnoses_json"] = json.dumps(
                [{"label": d, "confidence": c} for d, c in diags],
                ensure_ascii=False
            )

        else:
            topk = min(args.topk_proc, len(scored_for_fallback))
            top_scored = scored_for_fallback[:topk]

            if topk > 0:
                raw = np.array([s["score"] for s in top_scored], dtype=float)
                norm = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
                final_diag_codes = [s["code"] for s in top_scored]
                final_diag_labels = []
                fallback_json = [
                    {"label": c, "confidence": float(p)}
                    for c, p in zip(final_diag_codes, norm.tolist())
                ]
            else:
                final_diag_codes, final_diag_labels = [], []
                fallback_json = []

            out.at[i, "final_diagnoses_json"] = json.dumps(fallback_json, ensure_ascii=False)

    
        titles = []
        if final_diag_codes:
            if diag_title_by_code:
                titles = [diag_title_by_code.get(c, "") for c in final_diag_codes]
            else:
                titles = [proc_title_by_code.get(c, "") for c in final_diag_codes]

        titles_join = ";".join([t for t in titles if t]) if any(titles) else ""
        if final_diag_labels:  
            extra = ";".join(final_diag_labels)
            titles_join = (titles_join + (";" if titles_join and extra else "") + extra).strip(";")
        out.at[i, "final_diagnosis_titles"] = titles_join

        
        out.at[i, "final_diagnoses_codes"] = ";".join(final_diag_codes)


        
        if diag_title_by_code:
            titles = [diag_title_by_code.get(c, "") for c in final_diag_codes]
        else:
            titles = [""] * len(final_diag_codes)
   
        titles_join = ";".join([t for t in titles if t]) if any(titles) else ""
        if final_diag_labels:
            extra = ";".join(final_diag_labels)
            titles_join = (titles_join + (";" if titles_join and extra else "") + extra).strip(";")
        out.at[i,"final_diagnosis_titles"] = titles_join


        final_expl = llm_explain(
            client=client,
            text=text,
            procs=pred_codes,
            prots=prot_rank,
            diagnoses=[(c,1.0) for c in final_diag_codes] + [(lbl,1.0) for lbl in final_diag_labels],
            diag_title_map=diag_title_by_code if diag_title_by_code else None
        )
        out.at[i,"final_explanation"] = final_expl


        out.at[i,"ablation_note"] = note_str


        if gold_proc_col:
            gold_p = [c.replace(".","") for c in parse_list(row[gold_proc_col])]
            for k in [1,3,5]:
                p,r,f = prf_at_k(pred_codes, gold_p, k)
                proc_metrics[f"p@{k}"].append(p)
                proc_metrics[f"r@{k}"].append(r)
                proc_metrics[f"f1@{k}"].append(f)

        if gold_prot_col and len(prot_rank)>0:
            gold_b = [str(x) for x in parse_list(row[gold_prot_col])]
            gset   = set(gold_b)
            if gset:
                rm = rank_metrics(prot_rank, gset, ks=(1,3,5,10))
                for k in ["p@1","p@3","p@5","p@10","ndcg@1","ndcg@3","ndcg@5","ndcg@10","MAP","MRR"]:
                    prot_metrics[k].append(rm[k])


    summary_rows = []
    if any(proc_metrics.values()):
        pm = {k: (np.mean(v) if v else float("nan")) for k,v in proc_metrics.items()}
        summary_rows.append({"group":"Procedures", **pm})
    if any(prot_metrics.values()):
        bm = {k: (np.mean(v) if v else float("nan")) for k,v in prot_metrics.items()}
        summary_rows.append({"group":"Proteins", **bm})
    summary_df = pd.DataFrame(summary_rows)


    cfg_tag = []
    cfg_tag.append("LLM" if args.use_llm else "CE")
    cfg_tag.append("noRF" if args.disable_rf else "RF")
    cfg_tag.append("noRsn" if args.disable_reasoner else "RSN")
    tag = "_".join(cfg_tag)

    out_path = (args.input_csv if args.overwrite
                else str(Path(args.input_csv).with_name(Path(args.input_csv).stem + f"_{tag}_with_results.csv")))

    final_df = pd.concat([df, out], axis=1)
    final_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    sum_path = str(Path(out_path).with_name(Path(out_path).stem + "_metrics_summary.csv"))
    summary_df.to_csv(sum_path, index=False)

    print(f"[DONE] Wrote outputs: {out_path}")
    print(f"[DONE] Wrote metrics: {sum_path}")
    print(f"[Ablation] {note_str}")

if __name__ == "__main__":
    main()
