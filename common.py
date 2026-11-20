import numpy as np, csv, json, os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data'); FIG = os.path.join(BASE, 'figures')
def write_csv(name, rows, header):
    with open(os.path.join(DATA, name), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
def save_json(name, obj):
    with open(os.path.join(DATA, name), 'w') as f: json.dump(obj, f, indent=1, default=float)
