import json
import re

# Données extraites de tes captures SumUp (Nom, Quantité)
items_from_images = [
    ("Olives vertes anchois", 95), ("Kir", 92), ("Expresso", 68),
    ("Expresso double", 64), ("Expresso allongé", 64), ("Tisane", 54),
    ("Thé", 42), ("Bières - Pomme", 29), ("Biscuit sucré", 27),
    ("Eau gazeuse", 25), ("Jus & Sodas - Oasis", 22), ("Eau plate btl 50 cl", 19),
    ("Jus & Sodas - Jus de orange bio 20 cL brique", 19), ("Chips", 18),
    ("Jus & Sodas - Jus de pomme verre", 18), ("Vin - Blanc", 17),
    ("Jus & Sodas - Lipton Ice Tea", 16), ("Chocolat chaud", 15),
    ("Vin - Rouge", 15), ("Jus & Sodas - Coca Cola classique", 14),
    ("Bières - Commerce blonde 25 cL", 14),
    ("Saucissons et saucisses sèches - Saucissons mini sachet", 14),
    ("Jus & Sodas - Orangina", 12), ("Bières - Commerce ambrée 25 cL", 8),
    ("Fromage apéritif", 7), ("Cidre - Cidre bouteille", 6),
    ("Jus & Sodas - Coca Cola Zero", 4),
    ("Saucissons et saucisses sèches - Saucisse", 3),
    ("Jus & Sodas - Jus de pomme bio 20 cL brique", 3), ("Pop corn salé", 3),
    ("Schweppes agrumes", 2), ("Sachet cacahuetes", 1), ("Pate en croute", 1),
    ("Bières - Artisanale 33 cL", 0), ("Cidre - Cidre verre", 0)
]

with open('stock_items.json', 'r', encoding='utf-8') as f:
    original_items = json.load(f)

inventory_date = "2026-04-05"
updated_items = []
seen_names = set()

# 1. Mise à jour des articles existants

for item in original_items:
    sm = item.get("sumup_match", {})
    name_sm = sm.get("name", "")
    variant_sm = sm.get("variant", "")

    full_name2 = f"{name_sm} - {variant_sm}" if variant_sm else name_sm

    match_found = None

    for img_name, qty in items_from_images:
        if name_sm == img_name or full_name2 == img_name:
            match_found = (img_name, qty)

            break

    if match_found:
        img_name, qty = match_found

        if "stock_state" not in item:
            item["stock_state"] = {}
        item["stock_state"]["stock_on_hand"] = qty
        item["stock_state"]["last_inventory_date"] = inventory_date
        item["stock_state"]["inventory_count_method"] = "manual"

        # Nettoyage de l'historique auto car c'est un nouvel inventaire

        if "last_auto_update" in item["stock_state"]:
            del item["stock_state"]["last_auto_update"]

        if "stock_history" in item["stock_state"]:
            item["stock_state"]["stock_history"] = []

        seen_names.add(img_name)
    updated_items.append(item)

# 2. Ajout des nouveaux articles non reconnus

for img_name, qty in items_from_images:
    if img_name not in seen_names:
        clean_name = img_name.lower()
        clean_name = re.sub(r'[^a-z0-9]', '_', clean_name)
        sku = re.sub(r'_+', '_', clean_name).strip('_')

        new_item = {
            "sku": sku,
            "stock_sku": sku,
            "is_stock_reference": True,
            "label": img_name,
            "stock_label": img_name,
            "enabled": True,
            "unit": "piece",
            "stock_unit": "piece",
            "consumption_per_sale": 1,
            "category": "a_classer",
            "sumup_match": {
                "name": img_name,
                "variant": ""
            },
            "stock_state": {
                "stock_on_hand": qty,
                "stock_reserved": 0,
                "incoming_qty": 0,
                "incoming_eta": "",
                "last_inventory_date": inventory_date,
                "inventory_count_method": "manual",
                "stock_history": []
            }
        }
        updated_items.append(new_item)

# Sauvegarde dans un nouveau fichier pour vérification
with open('stock_items_nouveau.json', 'w', encoding='utf-8') as f:
    json.dump(updated_items, f, indent=2, ensure_ascii=False)

print(f"Terminé. {len(items_from_images) - len(seen_names)} nouveaux articles ajoutés.")
