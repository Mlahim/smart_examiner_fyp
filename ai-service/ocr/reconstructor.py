"""
Smart Examiner — Text Reconstructor

Responsibilities:
  - Sort detected bounding boxes into reading order
  - Group word-level boxes into logical text lines
  - Merge recognized text from all lines into final output
"""

import logging
from dataclasses import dataclass

import config

logger = logging.getLogger(__name__)


@dataclass
class TextLine:
    """A single line of recognized text with its position."""
    line_index: int
    text: str
    confidence: float
    y_position: float       # Average Y-coordinate for ordering
    is_low_confidence: bool = False


def group_into_lines(boxes: list) -> list[list]:
    """Group bounding boxes into logical text lines using slope-based spatial clustering.
    
    This replaces the old vertical-overlap logic, which fails on slanted handwriting.
    """
    if not boxes:
        return []

    # 1. Sort left-to-right
    sorted_boxes = sorted(boxes, key=lambda b: b.x_min)
    lines: list[list] = []

    for box in sorted_boxes:
        best_line = None
        best_dist_sq = float('inf')

        for line in lines:
            tail = line[-1]
            
            # Directionality: The new box must be to the right of the line's tail
            if box.center_x <= tail.center_x:
                continue
                
            # Horizontal gap: Don't jump across large gaps (e.g., column breaks)
            gap_x = box.x_min - tail.x_max
            if gap_x > 4 * tail.height:
                continue
                
            # Vertical proximity: Don't jump to the line below or above
            dy = box.center_y - tail.center_y
            if abs(dy) > 2 * tail.height:
                continue
                
            # Slope constraint: Ensure the angle follows a natural handwriting slant
            dx = box.center_x - tail.center_x
            slope = dy / max(1.0, dx)  # Prevent division by zero
            if abs(slope) > 0.4:  # ~22 degree max slant
                continue
                
            # Select the best match based on closest Euclidean distance
            dist_sq = dx**2 + dy**2
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_line = line
                
        if best_line is not None:
            best_line.append(box)
        else:
            lines.append([box])

    # 2. Sort lines top-to-bottom by median center_y
    lines.sort(key=lambda line: sorted([b.center_y for b in line])[len(line)//2])

    # 3. Sort boxes left-to-right within each line (already roughly sorted, but good to guarantee)
    for line in lines:
        line.sort(key=lambda b: b.center_x)

    return lines


def sort_boxes_reading_order(boxes: list) -> list:
    """Sort bounding boxes in reading order: top-to-bottom, left-to-right."""
    lines = group_into_lines(boxes)
    result = []
    for line in lines:
        result.extend(line)
        
    logger.info("Organized %d boxes into %d line(s).", len(result), len(lines))
    return result


def merge_line_texts(text_lines: list[TextLine]) -> str:
    """Merge recognized text lines into final output string.

    Filters out lines below the minimum confidence threshold.
    """
    filtered = [
        line for line in text_lines
        if line.confidence >= config.MIN_CONFIDENCE and line.text.strip()
    ]

    skipped = len(text_lines) - len(filtered)
    if skipped > 0:
        logger.info("Filtered out %d low-confidence/empty line(s).", skipped)

    # Sort by line_index to maintain reading order
    filtered.sort(key=lambda line: line.line_index)

    return "\n".join(line.text for line in filtered)
