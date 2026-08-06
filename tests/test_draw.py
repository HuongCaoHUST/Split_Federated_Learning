from io import StringIO

from rich.console import Console

from scripts.draw import draw_graph


def render_graph(client_data, max_layer):
    output = StringIO()
    console = Console(
        file=output,
        width=140,
        force_terminal=False,
        color_system=None,
    )
    draw_graph(client_data, max_layer=max_layer, output=console)
    return output.getvalue()


def test_draw_graph_keeps_existing_detection_usage():
    rendered = render_graph(
        [
            {"name": "Client 1", "cut_layer": 5, "image_count": 120},
            {"name": "Client 2", "cut_layer": 10, "image_count": 80},
        ],
        max_layer=23,
    )
    assert "Full model" in rendered
    assert "Client 1: 120" in rendered
    assert "Client 2: 80" in rendered
    assert "Server" in rendered


def test_draw_graph_supports_alexnet_and_zero_cut():
    rendered = render_graph(
        [
            {"name": "Client 1", "cut_layer": 0, "image_count": 100},
            {"name": "Client 2", "cut_layer": 4, "image_count": 200},
        ],
        max_layer=8,
    )
    assert "Client 1: 100" in rendered
    assert "Client 2: 200" in rendered
    assert "100" in rendered
    assert "8" in rendered
