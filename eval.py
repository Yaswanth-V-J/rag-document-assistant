import os
import json
import time
import chromadb
from openai import OpenAI, RateLimitError

TOP_K = 5
MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
JUDGE = os.getenv("JUDGE_MODEL", MODEL)

SYSTEM = (
    "You answer questions using ONLY the numbered context snippets provided. "
    "Cite snippets inline like [1], [2]. If the answer is not in the context, "
    "say \"I couldn't find that in the documents.\" Do not use outside knowledge."
)
REFUSAL = "couldn't find that in the documents"

col = chromadb.PersistentClient("chroma_db").get_or_create_collection("docs")
llm = OpenAI(
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
)


def chat(model, system, user):
    for attempt in range(6):
        try:
            r = llm.chat.completions.create(
                model=model,
                temperature=0,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            return r.choices[0].message.content.strip()
        except RateLimitError:
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("Rate limited too many times")


def judge(question, reference, candidate):
    out = chat(
        JUDGE,
        "You grade answers. Reply with exactly one word: CORRECT or INCORRECT.",
        f"Question: {question}\nReference answer: {reference}\nCandidate answer: {candidate}\n\n"
        "CORRECT if the candidate conveys the same key facts as the reference and does not contradict it.",
    )
    return out.upper().startswith("CORRECT")


def main():
    with open("eval_set.json", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    hits = rr_sum = correct = refused_ok = n_ans = n_unans = 0

    for i, item in enumerate(data, 1):
        q = item["question"]
        res = col.query(query_texts=[q], n_results=TOP_K)
        docs, metas = res["documents"][0], res["metadatas"][0]
        context = "\n\n".join(
            f"[{k}] ({m['source']}, p.{m['page']})\n{d}" for k, (d, m) in enumerate(zip(docs, metas), 1)
        )
        answer = chat(MODEL, SYSTEM, f"Context:\n{context}\n\nQuestion: {q}")
        row = {"question": q, "answer": answer}

        if item.get("answerable", True):
            n_ans += 1
            pages = item["page"] if isinstance(item["page"], list) else [item["page"]]
            rank = next(
                (k for k, m in enumerate(metas) if m["source"] == item["source"] and m["page"] in pages),
                None,
            )
            hit = rank is not None
            ok = judge(q, item["reference_answer"], answer)
            hits += hit
            rr_sum += 1 / (rank + 1) if hit else 0
            correct += ok
            row.update(type="answerable", retrieval_hit=hit, rank=None if rank is None else rank + 1, correct=ok)
        else:
            n_unans += 1
            ok = REFUSAL in answer.lower().replace("\u2019", "'")
            refused_ok += ok
            row.update(type="unanswerable", correctly_refused=ok)

        results.append(row)
        print(f"[{i}/{len(data)}] {row['type']}: {'PASS' if row.get('correct', row.get('correctly_refused')) else 'FAIL'} - {q[:60]}")
        time.sleep(1)

    summary = {"n_answerable": n_ans, "n_unanswerable": n_unans, "top_k": TOP_K, "llm": MODEL, "judge": JUDGE}
    if n_ans:
        summary["retrieval_hit_rate"] = round(hits / n_ans, 3)
        summary["mrr"] = round(rr_sum / n_ans, 3)
        summary["answer_accuracy"] = round(correct / n_ans, 3)
    if n_unans:
        summary["correct_refusal_rate"] = round(refused_ok / n_unans, 3)

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
