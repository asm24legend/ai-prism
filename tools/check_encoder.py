"""Run before using ANY encoder (review Appendix A).  python tools/check_encoder.py --model Simonlee711/Clinical_ModernBERT"""
import argparse

from src.encoder_check import inspect_encoder, masked_word_accuracy

PASSAGE = ("FINAL DIAGNOSIS: Lung, right upper lobe, lobectomy: invasive adenocarcinoma, acinar predominant, "
           "moderately differentiated, 3.2 cm, with visceral pleural invasion. Surgical margins are negative for "
           "carcinoma. Metastatic carcinoma is identified in one of twelve regional lymph nodes.")
FILLER = ("GROSS DESCRIPTION: The specimen is received fresh and labeled with the patient name and designated right "
          "upper lobe. It consists of a lobectomy specimen measuring 14 x 9 x 4 cm. The pleural surface is pink and "
          "glistening with focal puckering. The bronchial margin is identified and inked. Sectioning reveals a tan "
          "white firm mass in the parenchyma. Representative sections are submitted in cassettes A through K. ") * 2

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Simonlee711/Clinical_ModernBERT")
    ap.add_argument("--passage_file"); ap.add_argument("--filler_file")
    a = ap.parse_args()
    info = inspect_encoder(a.model)
    print("encoder:", info)
    if info["learned_position_table"]:
        print(f"-> only the first {info['trained_positions']} of {info['max_position_embeddings']} position slots are trained")
    passage = open(a.passage_file).read() if a.passage_file else PASSAGE
    filler = open(a.filler_file).read() if a.filler_file else FILLER
    r = masked_word_accuracy(a.model, passage, filler)
    print(r)
    ok = r["passage_after_filler"] > 0.5 * max(r["passage_first"], 1e-9)
    print("PASS: accuracy holds at later positions" if ok else
          "FAIL: accuracy collapses at later positions -> use windows within the trained positions")
