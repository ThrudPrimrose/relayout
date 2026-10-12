# Copyright 2026 ETH Zurich and the relayout authors.
"""Index expressions: the integer arithmetic a request's factors, grid extents and shuffles are written
in, and every layout's index maps (:mod:`relayout.layout`) are generated in. Python syntax over
non-negative integers, so ``//`` and ``%`` are also C's ``/`` and ``%``."""

import ast
import operator
from collections.abc import Callable, Mapping

__all__ = ["BINOP_FUNCTIONS", "INDEX_CALLS", "check_index_expr", "eval_index", "int_root"]

#: The operators an index expression may use: integer arithmetic, floor and mod, and the bitwise ones
#: XOR swizzles need.
BINOP_FUNCTIONS: dict[type[ast.operator], Callable[[int, int], int]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.BitXor: operator.xor,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
}


def int_root(value: int, degree: int) -> int:
    """The floor of the ``degree``-th root of ``value`` (``root(P, 2)`` is a square grid's edge)."""
    if value < 0 or degree < 1:
        raise ValueError(f"root({value}, {degree}) needs a non-negative value and a degree >= 1")
    guess = round(value ** (1 / degree))
    while guess**degree > value:
        guess -= 1
    while (guess + 1) ** degree <= value:
        guess += 1
    return guess


#: Function name -> (arity, implementation).
INDEX_CALLS: dict[str, tuple[int, Callable[..., int]]] = {
    "min": (2, min),
    "max": (2, max),
    "root": (2, int_root),
}


def check_index_expr(text: str, where: str) -> str:
    """``text`` when it is an integer expression of the allowed form, else a request fault."""
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        raise ValueError(f"{where}: {text!r} is not an expression") from None
    for walked in ast.walk(tree.body):
        node = walked.op if isinstance(walked, (ast.BinOp, ast.UnaryOp)) else walked
        allowed = (
            isinstance(node, (ast.Name, ast.Load, ast.USub, *BINOP_FUNCTIONS))
            or (isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool))
            or (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in INDEX_CALLS
                and len(node.args) == INDEX_CALLS[node.func.id][0]
                and not node.keywords
            )
        )
        if not allowed:
            raise ValueError(
                f"{where}: {text!r} uses {type(node).__name__}; an index expression takes ints, names, "
                f"+ - * // % ^ & | << >> and min(a, b), max(a, b), root(a, k)"
            )
    return text


def eval_index(text: str, names: Mapping[str, int], where: str) -> int:
    """The value of the index expression ``text`` (:func:`check_index_expr`'s form) with ``names`` bound;
    ``//`` and ``%`` floor like Python's."""

    def value(node: ast.expr) -> int:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in names:
                raise ValueError(f"{where}: {text!r} names {node.id!r}, which is not bound here ({sorted(names)})")
            return names[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -value(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in BINOP_FUNCTIONS:
            return BINOP_FUNCTIONS[type(node.op)](value(node.left), value(node.right))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in INDEX_CALLS:
            return INDEX_CALLS[node.func.id][1](*(value(arg) for arg in node.args))
        raise ValueError(f"{where}: {text!r} is not an index expression")

    return value(ast.parse(check_index_expr(text, where), mode="eval").body)
