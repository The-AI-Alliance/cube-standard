from cube_browser_tool._utils import flatten_axtree, prune_html


# ---------------------------------------------------------------------------
# prune_html
# ---------------------------------------------------------------------------


def test_prune_html_empty_string() -> None:
    assert prune_html("") == ""


def test_prune_html_removes_script_tags() -> None:
    result = prune_html("<html><body><script>alert(1)</script><p>hello</p></body></html>")
    assert "<script>" not in result
    assert "hello" in result


def test_prune_html_removes_style_tags() -> None:
    result = prune_html("<html><body><style>body{color:red}</style><p>hi</p></body></html>")
    assert "<style>" not in result
    assert "hi" in result


def test_prune_html_removes_html_comments() -> None:
    result = prune_html("<html><body><!-- this is a comment --><p>visible</p></body></html>")
    assert "this is a comment" not in result
    assert "visible" in result


def test_prune_html_preserves_content_tags() -> None:
    result = prune_html("<html><body><button id='btn'>Click</button></body></html>")
    assert "Click" in result
    assert "btn" in result


def test_prune_html_returns_string() -> None:
    assert isinstance(prune_html("<html><body><p>hi</p></body></html>"), str)


# ---------------------------------------------------------------------------
# flatten_axtree
# ---------------------------------------------------------------------------


def test_flatten_axtree_none_returns_sentinel() -> None:
    result = flatten_axtree(None)
    assert result == "[accessibility tree unavailable]"
    assert result != ""


def test_flatten_axtree_empty_dict() -> None:
    result = flatten_axtree({})
    assert isinstance(result, str)


def test_flatten_axtree_simple_node() -> None:
    tree = {"role": "button", "name": "Submit", "children": []}
    result = flatten_axtree(tree)
    assert "button" in result
    assert "Submit" in result


def test_flatten_axtree_with_value() -> None:
    tree = {"role": "textbox", "name": "Username", "value": "alice", "children": []}
    result = flatten_axtree(tree)
    assert "alice" in result


def test_flatten_axtree_nested_children() -> None:
    tree = {
        "role": "dialog",
        "name": "Login",
        "children": [
            {"role": "textbox", "name": "Email", "children": []},
            {"role": "button", "name": "Submit", "children": []},
        ],
    }
    result = flatten_axtree(tree)
    lines = result.splitlines()
    assert any("dialog" in line for line in lines)
    assert any("textbox" in line for line in lines)
    assert any("button" in line for line in lines)
    # Children should be indented relative to parent
    dialog_line = next(i for i, l in enumerate(lines) if "dialog" in l)
    child_line = next(i for i, l in enumerate(lines) if "textbox" in l)
    assert lines[child_line].startswith(" "), "children should be indented"
    assert dialog_line < child_line


def test_flatten_axtree_missing_keys_does_not_raise() -> None:
    # Nodes with no role, name, or value should not raise
    tree = {"children": [{"children": []}]}
    result = flatten_axtree(tree)
    assert isinstance(result, str)


def test_flatten_axtree_returns_string() -> None:
    assert isinstance(flatten_axtree({"role": "main", "name": "", "children": []}), str)
