from __future__ import annotations

from ocr2md.utils.markdown_utils import (
    normalize_fenced_code_block_spacing,
    normalize_markdown_output,
)


def test_normalize_fenced_code_block_spacing_removes_sparse_blank_lines() -> None:
    markdown = """Before

```xml

<root>

  <child>value</child>

</root>

```

After
"""

    normalized = normalize_fenced_code_block_spacing(markdown)

    assert "```xml\n<root>\n  <child>value</child>\n</root>\n```" in normalized
    assert "Before\n\n" in normalized
    assert "\n\nAfter" in normalized


def test_normalize_fenced_code_block_spacing_preserves_intentional_blank_line() -> None:
    markdown = """```python
def first():
    return 1

def second():
    return 2
```"""

    normalized = normalize_fenced_code_block_spacing(markdown)

    assert normalized == markdown


def test_normalize_markdown_output_fixes_xml_code_fence_language() -> None:
    markdown = """```html
<bean>
  <local-jndi-name>AddressHomeLocal</local-jndi-name>
</bean>
```"""

    normalized = normalize_markdown_output(markdown)

    assert normalized.startswith("```xml\n")


def test_normalize_markdown_output_collapses_blank_lines_between_list_items() -> None:
    markdown = """1. first item

2. second item

- third item"""

    normalized = normalize_markdown_output(markdown)

    assert normalized == "1. first item\n2. second item\n- third item"


def test_normalize_markdown_output_trims_formula_block_spacing() -> None:
    markdown = """Before


$$

x_{1}^{2}+x_{n}^{2}

$$


After"""

    normalized = normalize_markdown_output(markdown)

    assert normalized == "Before\n\n$$\nx_{1}^{2}+x_{n}^{2}\n$$\n\nAfter"
