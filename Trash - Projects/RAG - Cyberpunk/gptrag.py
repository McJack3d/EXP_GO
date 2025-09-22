import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss


# -------------------------------
# 1) Lecture HTML + nettoyage
# -------------------------------

def read_and_clean_html(html_path: Path) -> str:
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")

    # cibler le contenu principal (Fandom / MediaWiki)
    main = soup.find(id="mw-content-text") or soup.find("main") or soup.find(attrs={"role": "main"})
    if main:
        soup = BeautifulSoup(str(main), "lxml")

    # virer les blocs bruités
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # heuristique anti-menu : supprimer les très longues listes empilées en haut de page
    # (ça reste safe : notre source est un dump Fandom truffé de menus)
    for ul in soup.find_all("ul")[:50]:
        # si beaucoup de liens récurrents → menu
        links = ul.find_all("a")
        if len(links) >= 10:
            ul.decompose()

    # récupérer le texte
    text = soup.get_text(separator="\n")

    # nettoyage
    text = re.sub(r"\n{2,}", "\n", text)         # compresser lignes vides
    text = re.sub(r"[ \t]{2,}", " ", text)       # compresser espaces
    text = re.sub(r"\u200b", "", text)           # zero-width
    text = text.strip()

    # retirer citations entre crochets, même si elles contiennent des sauts de ligne
    text = re.sub(r"\[\s*(?:citation\s+needed|\d+)\s*\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"READ\s*MORE", "", text, flags=re.IGNORECASE)
    # normaliser les espaces (y compris multiples sauts de ligne)
    text = re.sub(r"\s+", " ", text).strip()

    # supprimer tête/deux trois lignes ultra redondantes si présentes
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # garder quelque chose de raisonnable
    cleaned = "\n".join(lines[:5000])  # coupe si jamais le HTML est gigantesque
    return cleaned


def preview(text: str, min_len=300, max_len=500) -> str:
    snippet = text[:max_len]
    # essayer de couper proprement à la fin d’une phrase
    m = re.search(r"(.{"+str(min_len)+r","+str(max_len)+r"}?[\.!?])", text, re.DOTALL)
    if m:
        snippet = m.group(1)
    return snippet.replace("\n", " ").strip()


# -------------------------------
# 2) Découpage en chunks (overlap)
# -------------------------------
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100, min_chunks: int = 5) -> List[str]:
    # si le texte est court, réduire la taille pour avoir >= 5 chunks
    total = len(text)
    if total > 0 and total < min_chunks * (chunk_size - overlap):
        chunk_size = max(300, total // min_chunks + overlap)

    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk.strip())
        if i + chunk_size >= len(text):
            break
        i += chunk_size - overlap
    # filtrage chunks trop petits / vides
    chunks = [c for c in chunks if len(c) > 20]
    return chunks


# -------------------------------
# 3) Embeddings (normalisés)
# -------------------------------
def embed_chunks(chunks: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Tuple[np.ndarray, SentenceTransformer]:
    model = SentenceTransformer(model_name)
    vecs = model.encode(chunks, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    # contrôle dimensions & normes
    dims = vecs.shape[1]
    norms = np.linalg.norm(vecs, axis=1)
    print(f"[Embeddings] modèle: {model_name} | dim: {dims} | norme moyenne: {norms.mean():.4f} (min {norms.min():.4f}, max {norms.max():.4f})")
    return vecs.astype("float32"), model


# -------------------------------
# 4) Index FAISS + retrieval
# -------------------------------
class Retriever:
    def __init__(self, embeddings: np.ndarray):
        self.dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)  # produit scalaire (cosine si vecteurs normalisés)
        self.index.add(embeddings)

    def search(self, q_emb: np.ndarray, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        scores, idx = self.index.search(q_emb, k)
        return scores, idx


def topk_with_scores(query: str, model: SentenceTransformer, retriever: Retriever, k: int = 3) -> Tuple[List[int], List[float]]:
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, idx = retriever.search(q, k=k)
    return idx[0].tolist(), scores[0].tolist()


# -------------------------------
# 5) Réponse MVP
# -------------------------------
def answer_mvp(query: str, chunks: List[str], model: SentenceTransformer, retriever: Retriever,
               k: int = 3, threshold: float = 0.30) -> Tuple[str, str]:
    idxs, scores = topk_with_scores(query, model, retriever, k=k)

    print("\n[Retrieval] Top-k résultats")
    for rank, (i, s) in enumerate(zip(idxs, scores), start=1):
        print(f"{rank:>2}. chunk#{i} | score(cosine)={s:.3f} | len={len(chunks[i])}")

    if not scores or scores[0] < threshold:
        return "Je ne sais pas.", ""

    best_i = idxs[0]
    source = f"chunk#{best_i}"
    extrait = chunks[best_i]
    # on renvoie un petit extrait du meilleur chunk
    extrait_short = extract_snippet(extrait, query, max_len=500)
    return extrait_short, source


# -------------------------------
# Utilitaire : extraire un extrait pertinent autour de la requête
# -------------------------------

def extract_snippet(text: str, query: str, max_len: int = 500) -> str:
    # normaliser l'espace pour éviter les fragments cassés par des retours à la ligne
    t = " ".join(text.split())
    q = query.strip()
    if not t:
        return ""

    # 1) priorité à une fenêtre autour de la requête exacte (meilleur pour noms propres)
    pos = t.lower().find(q.lower())
    if pos != -1:
        start = max(0, pos - max_len // 2)
        end = min(len(t), pos + len(q) + max_len // 2)
        return t[start:end].strip()

    # 2) fallback: découpe en phrases et score sur recouvrement des mots-clés
    sentences = re.split(r'(?<=[\.!?])\s+', t)
    if not sentences:
        return t[:max_len].strip()

    q_words = [w.lower() for w in re.findall(r"\w+", q) if len(w) > 2]
    best_idx, best_score = 0, -1
    for i, s in enumerate(sentences):
        s_low = s.lower()
        # score: présence des mots + bonus si tous présents
        score = sum(1 for w in q_words if w in s_low)
        if q_words and all(w in s_low for w in q_words):
            score += len(q_words)
        if score > best_score:
            best_idx, best_score = i, score

    snippet = sentences[best_idx]
    if len(snippet) < max_len and best_idx + 1 < len(sentences):
        snippet = (snippet + " " + sentences[best_idx + 1]).strip()
    return snippet[:max_len].strip()


# -------------------------------
# 6) (Optionnel) LLM avec garde-fous
#    -> ici un stub, prêt à brancher si tu veux
# -------------------------------
SAFETY_PROMPT = """Tu es un assistant francophone. Réponds UNIQUEMENT à partir du contexte fourni.
Si l'information n'est pas dans le contexte, dis: "Je ne sais pas".
Sois concis, cite l'extrait utile et résume en 2-3 phrases max.
"""

def llm_guardrailed_response(context: str, question: str) -> str:
    """
    Stub : à brancher avec le LLM de ton choix (OpenAI, etc.).
    Idée:
      - prompt = SAFETY_PROMPT + "\\n\\n[CONTEXTE]\\n" + context + "\\n\\n[QUESTION]\\n" + question
      - appeler l'API, récupérer la réponse
      - retourner la réponse.
    """
    # Pour cette démo on ne fait rien et on renvoie juste une réponse templatisée.
    return f"(LLM désactivé) Contexte trouvé:\n{context[:400]}..."


# -------------------------------
# Utilitaire : trouver le HTML si le nom contient des espaces/| etc.
# -------------------------------
def guess_html_path(user_path: str) -> Path:
    p = Path(user_path)
    if p.exists():
        return p
    # si on nous donne juste un dossier, chercher un .html à l’intérieur
    if p.is_dir():
        cands = sorted(p.glob("*.html"))
        if not cands:
            raise FileNotFoundError(f"Aucun .html trouvé dans {p}")
        return cands[0]
    # essayer une recherche large si le chemin contient des caractères spéciaux
    parent = p.parent if p.parent.as_posix() != "" else Path(".")
    pattern = p.name.replace("|", "*").replace(" ", "*")
    cands = list(parent.glob(pattern))
    if cands:
        return cands[0]
    raise FileNotFoundError(f"Fichier HTML introuvable : {user_path}")


# -------------------------------
# Demo end-to-end + tests
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Mini-pipeline RAG (HTML → chunks → embeddings → retrieval → réponse MVP).")
    parser.add_argument("--html", type=str, required=True, help="Chemin vers le fichier HTML.")
    parser.add_argument("--k", type=int, default=3, help="top-k")
    parser.add_argument("--threshold", type=float, default=0.12, help="Seuil de confiance (cosine) pour répondre.")
    parser.add_argument("--chunk_size", type=int, default=800, help="Taille d'un chunk (caractères).")
    parser.add_argument("--overlap", type=int, default=100, help="Overlap entre chunks (caractères).")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Modèle d'embedding.")
    parser.add_argument("--multilingual", action="store_true", help="Utilise un modèle d'embedding multilingue.")
    parser.add_argument("--ask", type=str, default=None, help="Question libre (sinon la démo lance 2 tests).")
    args = parser.parse_args()

    html_path = guess_html_path(args.html)
    print(f"[Input] {html_path}")

    text = read_and_clean_html(html_path)
    print("\n[Aperçu 300–500 caractères]")
    print(preview(text))

    chunks = chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap, min_chunks=5)
    print(f"\n[Chunking] nb_chunks={len(chunks)} | chunk_size≈{args.chunk_size} | overlap={args.overlap}")
    for i, c in enumerate(chunks[:5]):
        print(f"  - chunk#{i} (len={len(c)})")

    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" if args.multilingual else args.model
    embeddings, model = embed_chunks(chunks, model_name=model_name)
    retriever = Retriever(embeddings)

    # mode question libre
    if args.ask:
        extrait, source = answer_mvp(args.ask, chunks, model, retriever, k=args.k, threshold=args.threshold)
        print("\n[Réponse MVP]")
        if extrait == "Je ne sais pas.":
            print("Je ne sais pas.")
        else:
            print(extrait)
            print(f"\nSource: {source}")
        sys.exit(0)

    # 4) Retrieval tests
    test_ok = "Quel événement met fin à la Quatrième Guerre des Corporations ?"
    test_oos = "Quelle est la capitale de la France ?"

    print("\n[Test facile]")
    extrait, source = answer_mvp(test_ok, chunks, model, retriever, k=args.k, threshold=args.threshold)
    print("\n[Réponse MVP]")
    if extrait == "Je ne sais pas.":
        print("Je ne sais pas.")
    else:
        print(extrait)
        print(f"\nSource: {source}")

    print("\n[Test hors sujet]")
    extrait, source = answer_mvp(test_oos, chunks, model, retriever, k=args.k, threshold=args.threshold)
    print("\n[Réponse MVP]")
    if extrait == "Je ne sais pas.":
        print("Je ne sais pas.")
    else:
        print(extrait)
        print(f"\nSource: {source}")


if __name__ == "__main__":
    main()