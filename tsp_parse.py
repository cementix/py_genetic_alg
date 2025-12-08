import pandas as pd


def parse_tsp_df(path: str) -> pd.DataFrame:
    """Parse TSP file and return DataFrame with city coordinates."""
    ids, xs, ys = [], [], []
    in_coords = False
    declared_dim = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            upper = stripped.upper()

            if ":" in stripped and not in_coords:
                key, value = stripped.split(":", 1)
                if key.strip().upper() == "DIMENSION":
                    try:
                        declared_dim = int(value.strip().split()[0])
                    except (ValueError, IndexError):
                        declared_dim = None

            if upper.startswith("NODE_COORD_SECTION"):
                in_coords = True
                continue

            if upper.startswith("EOF"):
                break

            if not in_coords:
                continue

            parts = stripped.split()
            if len(parts) < 3:
                continue

            try:
                city_id = int(float(parts[0]))
                x = float(parts[1])
                y = float(parts[2])
                ids.append(city_id)
                xs.append(x)
                ys.append(y)
            except ValueError:
                continue

    df = (
        pd.DataFrame({"id": ids, "x": xs, "y": ys})
        .astype({"id": "int64", "x": "float64", "y": "float64"})
        .sort_values("id")
        .reset_index(drop=True)
    )

    if declared_dim is not None and declared_dim != len(df):
        print(
            f"warning: dimension mismatch (declared={declared_dim}, parsed={len(df)})"
        )

    return df
