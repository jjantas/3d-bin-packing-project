from __future__ import annotations
from typing import List, Tuple, Optional
import csv
from models import Container, Dims


def make_containers(boxes: List[Dims], warehouse: Dims) -> List[Container]:
    Wx, Wy, Wz = warehouse
    containers = []
    for l, w, h in boxes:
        c = Container(l, w, h)
        c.pick_warehouse(Wx, Wy, Wz)
        containers.append(c)
    return containers


def load_boxes_from_csv(path: str) -> Tuple[List[Dims], Optional[Dims]]:
    """
    Zwraca:
      - listę boxów [(l,w,h), ...]
      - opcjonalnie wymiary magazynu (Wx,Wy,Wz) jeśli plik je zawiera
    Akceptuje nagłówki: l,w,h lub length,width,height, oraz opcjonalnie Wx,Wy,Wz.
    """
    boxes: List[Dims] = []
    warehouse: Optional[Dims] = None

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [row for row in reader if any(cell.strip() for cell in row)]

    if not rows:
        raise ValueError(f"CSV is empty: {path}")

    header = [c.strip() for c in rows[0]]
    header_lower = [c.lower() for c in header]
    has_header = any(
        x in header_lower
        for x in ["l", "w", "h", "length", "width", "height", "wx", "wy", "wz"]
    )

    if has_header:
        col_index = {name: i for i, name in enumerate(header_lower)}
        data_rows = rows[1:]

        def get_int(row, *names) -> Optional[int]:
            for n in names:
                if n in col_index:
                    val = row[col_index[n]].strip()
                    if val != "":
                        return int(float(val))
            return None

        for row in data_rows:
            row = [c.strip() for c in row]
            l = get_int(row, "l", "length")
            w = get_int(row, "w", "width")
            h = get_int(row, "h", "height")
            if l is None or w is None or h is None:
                continue
            boxes.append((l, w, h))

            Wx = get_int(row, "wx")
            Wy = get_int(row, "wy")
            Wz = get_int(row, "wz")
            if (
                warehouse is None
                and Wx is not None
                and Wy is not None
                and Wz is not None
            ):
                warehouse = (Wx, Wy, Wz)

    else:
        for row in rows:
            if len(row) < 3:
                continue
            l, w, h = int(float(row[0])), int(float(row[1])), int(float(row[2]))
            boxes.append((l, w, h))

    if not boxes:
        raise ValueError(f"No boxes parsed from CSV: {path}")

    return boxes, warehouse
