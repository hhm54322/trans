#!/usr/bin/env python3
"""Prototype path-level removal for outlined CAD text in PDF content streams."""

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import fitz
import pikepdf


Matrix = Tuple[float, float, float, float, float, float]


def multiply(first: Matrix, second: Matrix) -> Matrix:
    """Compose PDF affine matrices so points use ``first`` then ``second``."""
    a1, b1, c1, d1, e1, f1 = first
    a2, b2, c2, d2, e2, f2 = second
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def transform(matrix: Matrix, x: float, y: float) -> Tuple[float, float]:
    a, b, c, d, e, f = matrix
    return a * x + c * y + e, b * x + d * y + f


def instruction_name(instruction) -> str:
    return str(instruction.operator)


def path_points(operator: str, operands, matrix: Matrix) -> Iterable[Tuple[float, float]]:
    values = [float(value) for value in operands]
    if operator in {"m", "l"}:
        yield transform(matrix, values[0], values[1])
    elif operator in {"c"}:
        for index in range(0, 6, 2):
            yield transform(matrix, values[index], values[index + 1])
    elif operator in {"v"}:
        for index in range(0, 4, 2):
            yield transform(matrix, values[index], values[index + 1])
    elif operator in {"y"}:
        for index in range(0, 4, 2):
            yield transform(matrix, values[index], values[index + 1])
    elif operator == "re":
        x, y, width, height = values
        for point_x, point_y in (
            (x, y),
            (x + width, y),
            (x + width, y + height),
            (x, y + height),
        ):
            yield transform(matrix, point_x, point_y)


def remove_contained_paths(
    instructions: Sequence,
    target_rect: fitz.Rect,
    page_height: float,
) -> Tuple[List, int]:
    """Remove fully contained painted path groups while retaining all other ops."""
    graphics_stack = [(1.0, 0.0, 0.0, 1.0, 0.0, 0.0)]
    current_matrix = graphics_stack[-1]
    path_indices: List[int] = []
    path_coordinates: List[Tuple[float, float]] = []
    path_has_clip = False
    remove_indices = set()
    paint_operators = {"S", "s", "f", "F", "f*", "B", "B*", "b", "b*", "n"}
    path_operators = {"m", "l", "c", "v", "y", "h", "re"}

    for index, instruction in enumerate(instructions):
        operator = instruction_name(instruction)
        if operator == "q":
            graphics_stack.append(current_matrix)
            continue
        if operator == "Q":
            current_matrix = graphics_stack.pop() if len(graphics_stack) > 1 else graphics_stack[0]
            continue
        if operator == "cm":
            values = tuple(float(value) for value in instruction.operands)
            current_matrix = multiply(current_matrix, values)  # type: ignore[arg-type]
            continue
        if operator in path_operators:
            path_indices.append(index)
            path_coordinates.extend(path_points(operator, instruction.operands, current_matrix))
            continue
        if operator in {"W", "W*"}:
            path_has_clip = True
            continue
        if operator not in paint_operators:
            continue
        if path_coordinates and not path_has_clip:
            xs = [point[0] for point in path_coordinates]
            ys = [point[1] for point in path_coordinates]
            path_rect = fitz.Rect(
                min(xs),
                page_height - max(ys),
                max(xs),
                page_height - min(ys),
            )
            if not path_rect.is_empty and target_rect.contains(path_rect):
                remove_indices.update(path_indices)
                remove_indices.add(index)
        path_indices = []
        path_coordinates = []
        path_has_clip = False
    return (
        [instruction for index, instruction in enumerate(instructions) if index not in remove_indices],
        len(remove_indices),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--page", type=int, required=True)
    parser.add_argument("--content-xref", type=int, required=True)
    parser.add_argument("--rect", type=float, nargs=4, required=True)
    args = parser.parse_args()

    with fitz.open(args.source) as source:
        page_height = float(source[args.page - 1].rect.height)
    with pikepdf.Pdf.open(args.source) as document:
        stream = document.get_object(args.content_xref, 0)
        instructions = pikepdf.parse_content_stream(stream)
        retained, removed = remove_contained_paths(
            instructions,
            fitz.Rect(args.rect),
            page_height,
        )
        stream.write(pikepdf.unparse_content_stream(retained))
        document.save(args.output)
    print(f"removed_instructions={removed} output={args.output}")


if __name__ == "__main__":
    main()
