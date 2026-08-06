from rich.console import Console
from rich.table import Table


DEFAULT_MAX_LAYER = 23
SCALE = 2


def _get_range(row_values, segments):
    active_segments = [
        segment
        for segment, value in zip(segments, row_values)
        if value is not None
    ]

    if not active_segments:
        return 0, 0

    return active_segments[0][0], active_segments[-1][1]


def _draw_line(start, end, row_values, segments, max_layer, scale=SCALE):
    width = max_layer * scale
    characters = [" "] * (width + 1)

    start_position = start * scale
    end_position = end * scale
    characters[start_position] = "├"

    for (segment_start, segment_end), value in zip(segments, row_values):
        if segment_end <= start or segment_start >= end:
            continue

        segment_start_position = max(segment_start, start) * scale
        segment_end_position = min(segment_end, end) * scale

        for index in range(segment_start_position + 1, segment_end_position + 1):
            characters[index] = "─"

        if value is not None:
            label = str(value)
            label_start = segment_start_position + (
                segment_end_position - segment_start_position - len(label)
            ) // 2
            characters[label_start:label_start + len(label)] = label

    characters[end_position] = "▶"
    return "".join(characters)


def _draw_guides(segments, max_layer, scale=SCALE):
    width = max_layer * scale
    characters = [" "] * (width + 1)

    for _, segment_end in segments:
        characters[segment_end * scale] = "┆"

    return "".join(characters)


def draw_graph(client_data, max_layer=DEFAULT_MAX_LAYER, output=None, scale=SCALE):
    """Draw the split-learning graph for the registered edge clients.

    Each item in ``client_data`` must contain ``cut_layer`` and
    ``image_count``. ``name`` is optional and defaults to ``Client N``.
    """
    if not client_data:
        raise ValueError("At least one client is required to draw the graph.")
    if max_layer < 1:
        raise ValueError("max_layer must be positive.")
    if scale < 2:
        raise ValueError("scale must be at least 2.")

    clients = []
    for index, client in enumerate(client_data, start=1):
        cut_layer = int(client["cut_layer"])
        image_count = int(client["image_count"])
        if not 0 <= cut_layer <= max_layer:
            raise ValueError(
                f"Client cut layer must be between 0 and {max_layer}; "
                f"received {cut_layer}."
            )
        if image_count < 0:
            raise ValueError("image_count cannot be negative.")
        clients.append({
            "name": client.get("name", f"Client {index}"),
            "cut_layer": cut_layer,
            "image_count": image_count,
        })

    segment_ends = sorted(
        ({client["cut_layer"] for client in clients} | {max_layer}) - {0}
    )
    segments = list(zip([0] + segment_ends[:-1], segment_ends))

    server_values = []
    for segment_start, _ in segments:
        image_count = sum(
            client["image_count"]
            for client in clients
            if client["cut_layer"] <= segment_start
        )
        server_values.append(image_count if image_count else None)

    rows = [("Server", server_values)]
    for client in clients:
        client_values = [
            client["image_count"] if segment_end <= client["cut_layer"] else None
            for _, segment_end in segments
        ]
        rows.append((f"{client['name']}: {client['image_count']}", client_values))

    labels = ["Full model"] + [name for name, _ in rows]
    column_width = max(15, max(len(label) for label in labels))
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(width=column_width)
    table.add_column()

    table.add_row(
        "Full model",
        _draw_line(
            0,
            max_layer,
            [None] * len(segments),
            segments,
            max_layer,
            scale,
        ),
    )
    guide = _draw_guides(segments, max_layer, scale)
    table.add_row("", guide)

    for name, row_values in rows:
        start, end = _get_range(row_values, segments)
        table.add_row(
            name,
            _draw_line(start, end, row_values, segments, max_layer, scale),
        )
        table.add_row("", guide)

    if output is None:
        output = Console()
    output.print(table)

    number_offset = column_width + 2
    number_line = [" "] * (max_layer * scale + number_offset + 2)
    for layer in [0] + segment_ends:
        position = number_offset + layer * scale
        layer_label = str(layer)
        number_line[position:position + len(layer_label)] = layer_label
    output.print("".join(number_line))


if __name__ == "__main__":
    draw_graph([
        {"name": "Client 1", "cut_layer": 5, "image_count": 120},
        {"name": "Client 2", "cut_layer": 5, "image_count": 80},
        {"name": "Client 3", "cut_layer": 10, "image_count": 150},
        {"name": "Client 4", "cut_layer": 15, "image_count": 50},
    ])
