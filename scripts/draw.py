from rich.console import Console
from rich.table import Table

console = Console()

MAX_LAYER = 23
SCALE = 2  # số ký tự cho mỗi layer
LABEL_WIDTH = 3

SEGMENTS = [(0, 5), (5, 10), (10, 15), (15, 23)]

values = {
    "Server": [None, 200, 350, 400],
    "Client 1": [120, None, None, None],
    "Client 2": [80, None, None, None],
    "Client 3": [150, 150, None, None],
    "Client 4": [50, 50, 50, None],
}

CLIENT_IMAGE_COUNTS = {
    "Client 1": 120,
    "Client 2": 80,
    "Client 3": 150,
    "Client 4": 50,
}


def get_range(row_values):
    """Trả về layer bắt đầu và kết thúc của một dòng."""
    active_segments = [
        segment
        for segment, value in zip(SEGMENTS, row_values)
        if value is not None
    ]

    if not active_segments:
        return 0, 0

    return active_segments[0][0], active_segments[-1][1]

# Các vị trí cần dóng
GUIDES = [end for _, end in SEGMENTS]


def draw_line(start, end, row_values):
    width = MAX_LAYER * SCALE
    chars = [" "] * (width + 1)

    s = start * SCALE
    e = end * SCALE

    chars[s] = "├"

    for (segment_start, segment_end), value in zip(SEGMENTS, row_values):
        if segment_end <= start or segment_start >= end:
            continue

        segment_s = max(segment_start, start) * SCALE
        segment_e = min(segment_end, end) * SCALE

        for index in range(segment_s + 1, segment_e + 1):
            chars[index] = "─"

        if value is not None:
            label = str(value)
            label_start = segment_s + (segment_e - segment_s - LABEL_WIDTH) // 2
            chars[label_start:label_start + len(label)] = label

    chars[e] = "▶"

    return "".join(chars)


def draw_guides():
    """Tạo một dòng chỉ chứa các đường dóng dọc"""
    width = MAX_LAYER * SCALE
    chars = [" "] * (width + 2)

    for x in GUIDES:
        chars[x * SCALE] = "┆"   # Có thể đổi thành │ ╎ ┊

    return "".join(chars)


guide = draw_guides()

table = Table(show_header=False, box=None, pad_edge=False)
table.add_column(width=15)
table.add_column()

# Full model
table.add_row("Full model", draw_line(0, MAX_LAYER, [None] * len(SEGMENTS)))
table.add_row("", guide)

# Server and clients
for name, row_values in values.items():
    start, end = get_range(row_values)
    display_name = f"{name}: {CLIENT_IMAGE_COUNTS[name]}" if name in CLIENT_IMAGE_COUNTS else name
    table.add_row(display_name, draw_line(start, end, row_values))
    table.add_row("", guide)

console.print(table)

# Layer numbers
number_line = [" "] * (MAX_LAYER * SCALE + 15)
offset = 17

for x in [0, 5, 10, 15, 23]:
    pos = offset + x * SCALE
    s = str(x)
    number_line[pos:pos + len(s)] = list(s)

console.print("".join(number_line))
