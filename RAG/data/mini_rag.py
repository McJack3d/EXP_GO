from bs4 import BeautifulSoup
import re

path = "/Users/alexandrebredillot/Documents/GitHub/EXP/RAG/data/Timeline | Cyberpunk Wiki | Fandom.html"

# 1) Ne plus imprimer l'aperçu brut
# fichier = open(path, "r", encoding="utf-8")
# aperçu = fichier.read(500)
# print(aperçu)
# fichier.close()

with open(path, "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")

# 2) Supprimer le bruit structurel courant
for tag in ["script", "style", "noscript", "nav", "footer", "header", "aside", "form"]:
    for el in soup.find_all(tag):
        el.decompose()

# (Optionnel) Fandom/MediaWiki : enlever quelques conteneurs fréquents si présents
for sel in [
    "#mw-navigation", ".mw-footer", ".vector-header", ".vector-toc",
    ".mw-editsection", ".global-notice"
]:
    for el in soup.select(sel):
        el.decompose()

# 3) Récupérer uniquement le texte (sans balises)
clean_text = soup.get_text(separator=" ", strip=True)

# 4) Normaliser les espaces
clean_text = re.sub(r"\s+", " ", clean_text).strip()

# 5) (Optionnel et agressif) enlever TOUT ce qui est entre chevrons dans le texte
# ⚠️ Cela supprime aussi des <mots> qui font partie du contenu !
# clean_text = re.sub(r"<[^>]+>", "", clean_text)

print(clean_text[:500])