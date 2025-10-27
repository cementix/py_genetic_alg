import pandas as pd

def parse_tsp_df(path: str) -> pd.DataFrame:
    ids, xs, ys = [], [], []
    in_coords = False
    declared_dim = None
    
    with open(path, "r", encoding="utf-8", errors='ignore') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            
            u = s.upper()
            
            if ":" in s and not in_coords:
                key, val = s.split(":", 1)
                if key.strip().upper() == 'DIMENSION':
                    try:
                        declared_dim = int(val.strip().split()[0])
                    except Exception:
                        declared_dim = None
                        
            if u.startswith("NODE_COORD_SECTION"):
                in_coords = True
                continue
            
            if u.startswith("EOF"):
                break
            
            if not in_coords:
                continue
            
            parts = s.split()
            if len(parts) < 3:
                continue
            try:
                cid = int(float(parts[0]))
                x = float(parts[1])
                y = float(parts[2])
            except ValueError:
                continue
            
            ids.append(cid); xs.append(x); ys.append(y)
            
    df = (
        pd.DataFrame({"id": ids, "x": xs, "y": ys})
          .astype({"id": "int64", "x": "float64", "y": "float64"})
          .sort_values("id")
          .reset_index(drop=True)
    )

    if declared_dim is not None and declared_dim != df.shape[0]:
        print(f"warning: dimension mismatch (declared={declared_dim}, parsed={df.shape[0]})")

    return df