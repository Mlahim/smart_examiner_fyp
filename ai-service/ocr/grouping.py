import re
import logging
from typing import Any

logger = logging.getLogger(__name__)


def is_new_student_evidence(identity: dict[str, Any]) -> bool:
    """Check if there is strong evidence to start a new student."""
    if not identity:
        return False
    name = identity.get("name")
    roll_no = identity.get("roll_no")
    
    # Simple heuristic: If roll_no is confidently extracted (not null and not empty or 'None')
    if roll_no and str(roll_no).strip().lower() not in ("null", "none", "", "[unclear]"):
        return True
        
    # Same for name
    if name and str(name).strip().lower() not in ("null", "none", "", "[unclear]"):
        return True
        
    return False


def group_students(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group pages into students and extract questions.
    
    Args:
        pages: List of dictionaries containing page_number, raw_text, student_identity, administrative_text
        
    Returns:
        List of student dictionaries with their pages and grouped questions.
    """
    students = []
    current_student = None
    
    for i, page in enumerate(pages):
        identity = page.get("student_identity", {})
        is_new = is_new_student_evidence(identity)
        
        # Start a new student if evidence exists OR if it's the very first page
        if is_new or current_student is None:
            current_student = {
                "student_index": len(students) + 1,
                "identity": identity,
                "pages": [],
                "administrative_text": "",
                "page_objects": [],  # Store full page objects for question grouping
                "questions": []
            }
            students.append(current_student)
            
            # If we created a new student without strong evidence (e.g. first page), log it
            if not is_new and i == 0:
                current_student["grouping_status"] = "assumed_first_student"
                
        else:
            # Continuation assumed
            if "grouping_status" not in current_student:
                current_student["grouping_status"] = "continuation_assumed"
            
        current_student["pages"].append(page.get("page_number", i + 1))
        
        # Accumulate administrative text from the page (e.g. headers, name, roll no)
        page_admin = page.get("administrative_text") or ""
        page_admin = page_admin.strip()
        if page_admin:
            if current_student["administrative_text"]:
                current_student["administrative_text"] += "\n" + page_admin
            else:
                current_student["administrative_text"] = page_admin
                
        current_student["page_objects"].append(page)
        
    # Now group questions for each student
    for student in students:
        questions, preamble = extract_questions_from_pages(student.pop("page_objects"))
        student["questions"] = questions
        
        # Append any raw preamble text to the administrative text
        if preamble:
            if student["administrative_text"]:
                student["administrative_text"] += "\n" + preamble
            else:
                student["administrative_text"] = preamble
                
    return students


def extract_questions_from_pages(page_objects: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    """Extract questions recursively from a list of pages belonging to a single student."""
    
    q_pattern = re.compile(
        r"^\s*[^a-zA-Z0-9]*\s*(q?uestion|q?ues|q|q?no\.?)\s*(?:\.?\s*)?(no\.?|number)?\s*[:\-\.]?\s*(\d+)",
        re.IGNORECASE
    )

    part_pattern = re.compile(
        r"^\s*(part|section)\s+([a-zA-Z]|[0-9]+|[ivxIVX]+)\b",
        re.IGNORECASE
    )

    general_marker_pattern = re.compile(
        r"^\s*("
        # Parenthesis ending markers (can be followed by anything, even \w directly)
        r"(?:\([a-zA-Z]\)|[a-zA-Z]\)|\([ivxIVX]{2,4}\)|[ivxIVX]{2,4}\)|\([0-9]{1,2}\)|[0-9]{1,2}\))\s*[:\-\.]?\s*(?=\w|\S|\n|$)"
        r"|"
        # Period ending markers (must be followed by whitespace, newline, or EOF to avoid R.E, e.g., i.e.)
        r"(?:[a-zA-Z]\.|[ivxIVX]{2,4}\.|[0-9]{1,2}\.)\s*[:\-]?\s*(?=\s|\n|$)"
        r")",
        re.IGNORECASE
    )

    def get_marker_info(match_str):
        s = re.sub(r'[\(\)\.\-:\s]', '', match_str).lower()
        if s.isdigit():
            return ('numeric', int(s))
        roman_map = {'i': 1, 'ii': 2, 'iii': 3, 'iv': 4, 'v': 5, 'vi': 6, 'vii': 7, 'viii': 8, 'ix': 9, 'x': 10}
        if s in roman_map:
            return ('roman', roman_map[s])
        if len(s) == 1 and 'a' <= s <= 'z':
            return ('alpha', ord(s) - ord('a') + 1)
        return ('unknown', 0)

    class Block:
        def __init__(self, m_type, m_val, m_str, norm, remaining, page):
            self.type = m_type
            self.val = m_val
            self.marker_str = m_str
            self.norm = norm
            self.first_line = remaining
            self.first_line_page = page
            self.extra_lines = []
            self.pages = {page} if page else set()
            self.children = []
            
        def add_line(self, page, line):
            self.extra_lines.append((page, line))
            if page:
                self.pages.add(page)
                
        def has_marks(self):
            text = self.first_line + "\n" + "\n".join(l for p, l in self.extra_lines)
            return bool(re.search(r'\d+\s*(marks?|pts?)', text, re.IGNORECASE))
            
        def get_full_answer_lines(self, include_marker=False):
            lines = []
            first = self.first_line
            if include_marker:
                first = self.marker_str + first
            if first.strip():
                lines.append(first.strip())
            for p, l in self.extra_lines:
                lines.append(l)
            for child in self.children:
                lines.extend(child.get_full_answer_lines(include_marker=True))
            return lines
            
        def collect_pages(self):
            pgs = set(self.pages)
            for c in self.children:
                pgs.update(c.collect_pages())
            return pgs
            
        def to_dict(self):
            pages_list = sorted(list(self.collect_pages()))
            ans = "\n".join(self.get_full_answer_lines(include_marker=False)).strip()
            
            out = {
                "pages": pages_list,
                "answer": ans,
                "subquestions": [c.to_dict() for c in self.children]
            }
            if self.type == 'q_prefix' or self.type == 'root_numeric':
                out["question_id"] = self.marker_str.strip().rstrip(":- \t")
                out["normalized_question_id"] = self.norm
            else:
                out["subquestion_id"] = self.marker_str.strip().rstrip(":- \t")
                out["normalized_subquestion_id"] = self.norm
            return out

    # Pass 1: Parse flat blocks
    root = Block('root', 0, '', '', '', None)
    current_block = root
    blocks = []
    
    for page_obj in page_objects:
        page_num = page_obj.get("page_number")
        raw_text = page_obj.get("raw_text") or ""
        
        for line in raw_text.split("\n"):
            if not line.strip():
                continue
                
            q_m = q_pattern.search(line)
            if q_m:
                m_str = q_m.group(0)
                norm = str(int(q_m.group(3)))
                rem = line[q_m.end():].lstrip(":- \t")
                b = Block('q_prefix', int(norm), m_str, norm, rem, page_num)
                blocks.append(b)
                current_block = b
                continue
                
            p_m = part_pattern.search(line)
            if p_m:
                m_str = p_m.group(0)
                norm = p_m.group(2).lower()
                m_type, m_val = get_marker_info(norm)
                rem = line[p_m.end():].lstrip(":- \t")
                b = Block('part_prefix', m_val, m_str, norm, rem, page_num)
                blocks.append(b)
                current_block = b
                continue
                
            g_m = general_marker_pattern.search(line)
            if g_m:
                m_str = g_m.group(0)
                norm = re.sub(r'[\(\)\.\-:\s]', '', g_m.group(1)).lower()
                m_type, m_val = get_marker_info(norm)
                if m_type != 'unknown':
                    rem = line[g_m.end():].lstrip(":- \t")
                    b = Block(m_type, m_val, m_str, norm, rem, page_num)
                    blocks.append(b)
                    current_block = b
                    continue
                    
            current_block.add_line(page_num, line)
            
    # Pass 2: Build Tree
    has_q_prefix = any(b.type == 'q_prefix' for b in blocks)
    stack = [root]
    
    for b in blocks:
        if b.type == 'q_prefix':
            stack = [root]
            root.children.append(b)
            stack.append(b)
            continue
            
        parent = None
        for i in range(len(stack)-1, -1, -1):
            node = stack[i]
            if node.type == b.type and b.val > node.val:
                parent = stack[i-1]
                stack = stack[:i]
                break
                
        if parent is None:
            parent = stack[-1]
            
        parent.children.append(b)
        stack.append(b)
        
    # Pass 3: Flatten Rejected
    def flatten_rejected(node, depth=0):
        new_children = []
        for child in node.children:
            flatten_rejected(child, depth + 1)
            
            is_rejected = False
            if child.type == 'numeric':
                is_top_level = (depth == 0 and not has_q_prefix)
                has_child_nodes = len(child.children) > 0
                has_explicit_marks = child.has_marks()
                
                if not (is_top_level or has_child_nodes or has_explicit_marks):
                    is_rejected = True
            
            if is_rejected:
                # Merge into node (preserves content exactly)
                if child.first_line_page:
                    node.extra_lines.append((child.first_line_page, child.marker_str + child.first_line))
                node.extra_lines.extend(child.extra_lines)
                node.pages.update(child.pages)
                # If a rejected node somehow had children (should be caught by has_child_nodes above),
                # they would be lost, but the logic guarantees a rejected node has no children.
            else:
                if child.type == 'numeric' and depth == 0 and not has_q_prefix:
                    child.type = 'root_numeric'
                new_children.append(child)
                
        node.children = new_children

    flatten_rejected(root)
    
    # Extract preamble
    preamble_lines = []
    if root.first_line.strip():
        preamble_lines.append(root.first_line.strip())
    for p, l in root.extra_lines:
        preamble_lines.append(l)
    preamble_text = "\n".join(preamble_lines).strip()
    
    questions = [c.to_dict() for c in root.children]
    return questions, preamble_text
