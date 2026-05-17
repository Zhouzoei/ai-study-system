from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import re
import uuid


@dataclass
class ChunkNode:
    node_id: str = ""
    level: int = 3
    content: str = ""
    title: str = ""
    children_ids: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""
    start_char: int = 0
    end_char: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "level": self.level,
            "content": self.content,
            "title": self.title,
            "children_ids": self.children_ids,
            "parent_id": self.parent_id,
            "metadata": self.metadata,
            "doc_id": self.doc_id,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkNode":
        return cls(
            node_id=data.get("node_id", ""),
            level=data.get("level", 3),
            content=data.get("content", ""),
            title=data.get("title", ""),
            children_ids=data.get("children_ids", []),
            parent_id=data.get("parent_id"),
            metadata=data.get("metadata", {}),
            doc_id=data.get("doc_id", ""),
            start_char=data.get("start_char", 0),
            end_char=data.get("end_char", 0),
        )


@dataclass
class Section:
    title: str = ""
    level: int = 1
    content: str = ""
    children: List["Section"] = field(default_factory=list)
    start_pos: int = 0
    end_pos: int = 0

    def flatten_with_parent(self, parent_title: str = "") -> List[Dict]:
        results = []
        full_title = f"{parent_title} > {self.title}" if parent_title else self.title
        results.append({
            "title": self.title,
            "full_title": full_title,
            "level": self.level,
            "content": self.content.strip(),
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
        })
        for child in self.children:
            results.extend(child.flatten_with_parent(full_title))
        return results


class HierarchicalChunker:
    def __init__(
        self,
        l1_max_size: int = 2000,
        l2_max_size: int = 500,
        l3_max_size: int = 200,
        l3_min_size: int = 60,
        overlap: int = 30,
    ):
        self.l1_max_size = l1_max_size
        self.l2_max_size = l2_max_size
        self.l3_max_size = l3_max_size
        self.l3_min_size = l3_min_size
        self.overlap = overlap

    def chunk(self, text: str, doc_id: str = "") -> List[ChunkNode]:
        sections = self._parse_markdown_sections(text)
        if not sections:
            sections = [Section(title="Document", level=1, content=text, start_pos=0, end_pos=len(text))]

        has_l2_or_l3 = any(s.level >= 2 for s in sections)
        if not has_l2_or_l3 and sections:
            sections = self._expand_flat_sections(sections, text)

        nodes = self._sections_to_nodes(sections, doc_id)
        return nodes

    def _parse_markdown_sections(self, text: str) -> List[Section]:
        lines = text.split("\n")
        sections = []
        current_section = None
        current_content_lines = []
        current_start = 0
        char_pos = 0

        heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$")

        for line in lines:
            match = heading_pattern.match(line)
            if match:
                if current_section is not None or current_content_lines:
                    content = "\n".join(current_content_lines).strip()
                    if current_section is None:
                        current_section = Section(
                            title="Preamble",
                            level=1,
                            content=content,
                            start_pos=current_start,
                            end_pos=char_pos,
                        )
                    else:
                        current_section.content = content
                        current_section.end_pos = char_pos
                    sections.append(current_section)

                heading_level = len(match.group(1))
                heading_title = match.group(2).strip()
                current_section = Section(
                    title=heading_title,
                    level=heading_level,
                    start_pos=char_pos,
                )
                current_content_lines = []
                current_start = char_pos
            else:
                current_content_lines.append(line)

            char_pos += len(line) + 1

        if current_section is not None or current_content_lines:
            content = "\n".join(current_content_lines).strip()
            if current_section is None:
                current_section = Section(
                    title="Preamble",
                    level=1,
                    content=content,
                    start_pos=current_start,
                    end_pos=char_pos,
                )
            else:
                current_section.content = content
                current_section.end_pos = char_pos
            sections.append(current_section)

        return sections

    def _sections_to_nodes(self, sections: List[Section], doc_id: str) -> List[ChunkNode]:
        nodes = []
        current_l1_node = None
        current_l2_node = None
        synthetic_l2_count = 0

        for section in sections:
            if section.level == 1:
                current_l1_node = self._create_l1_node(section, doc_id, None)
                nodes.append(current_l1_node)
                current_l2_node = None

            elif section.level == 2:
                l2_node = self._create_l2_node(section, doc_id, current_l1_node.node_id if current_l1_node else None)
                nodes.append(l2_node)
                if current_l1_node:
                    current_l1_node.children_ids.append(l2_node.node_id)
                current_l2_node = l2_node

            elif section.level == 3:
                synthetic_l2_count += 1
                l1_id = current_l1_node.node_id if current_l1_node else None
                l2_node = ChunkNode(
                    node_id=f"L2_{uuid.uuid4().hex[:12]}",
                    level=2,
                    content=section.content[:100],
                    title=section.title,
                    parent_id=l1_id,
                    doc_id=doc_id,
                )
                nodes.append(l2_node)
                if current_l1_node:
                    current_l1_node.children_ids.append(l2_node.node_id)
                current_l2_node = l2_node
                l3_nodes = self._chunk_subsection(section, doc_id, current_l2_node.node_id)
                for ln in l3_nodes:
                    current_l2_node.children_ids.append(ln.node_id)
                nodes.extend(l3_nodes)

            else:
                pass

        return nodes

    def _create_l1_node(self, section: Section, doc_id: str, parent_id: Optional[str]) -> ChunkNode:
        content = section.content[:self.l1_max_size] if section.content else section.title
        if len(section.content) > self.l1_max_size:
            children_content = "\n".join(
                c.content[:200] for c in section.children[:5]
            )
            content = f"{section.title}\n\n{children_content}"[:self.l1_max_size]

        return ChunkNode(
            node_id=f"L1_{uuid.uuid4().hex[:12]}",
            level=1,
            content=content,
            title=section.title,
            parent_id=parent_id,
            metadata={"full_title": section.title, "has_children": bool(section.children)},
            doc_id=doc_id,
            start_char=section.start_pos,
            end_char=section.end_pos,
        )

    def _create_l2_node(self, section: Section, doc_id: str, parent_id: Optional[str]) -> ChunkNode:
        content = section.content[:self.l2_max_size]
        return ChunkNode(
            node_id=f"L2_{uuid.uuid4().hex[:12]}",
            level=2,
            content=content,
            title=section.title,
            parent_id=parent_id,
            metadata={"full_title": section.title},
            doc_id=doc_id,
            start_char=section.start_pos,
            end_char=section.end_pos,
        )

    def _chunk_subsection(
        self, child: Section, doc_id: str, parent_id: str
    ) -> List[ChunkNode]:
        content = child.content.strip()
        if not content:
            return []

        if len(content) <= self.l3_max_size:
            return [ChunkNode(
                node_id=f"L3_{uuid.uuid4().hex[:12]}",
                level=3,
                content=content,
                title=child.title,
                parent_id=parent_id,
                metadata={"full_title": child.title},
                doc_id=doc_id,
                start_char=child.start_pos,
                end_char=child.end_pos,
            )]

        chunks = self._split_text_at_sentence_boundaries(content, child)
        result = []
        for chunk_text, start, end in chunks:
            result.append(ChunkNode(
                node_id=f"L3_{uuid.uuid4().hex[:12]}",
                level=3,
                content=chunk_text,
                title=child.title,
                parent_id=parent_id,
                metadata={"full_title": child.title},
                doc_id=doc_id,
                start_char=start,
                end_char=end,
            ))
        return result

    def _split_text_at_sentence_boundaries(
        self, text: str, section: Section
    ) -> List[tuple]:
        sentences = re.split(r"(?<=[。！？.!?\n])\s*", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = ""
        chunk_start = section.start_pos

        for sent in sentences:
            if len(current_chunk) + len(sent) > self.l3_max_size and current_chunk:
                chunks.append((current_chunk.strip(), chunk_start, chunk_start + len(current_chunk)))
                overlap_text = current_chunk[-self.overlap:] if self.overlap > 0 else ""
                current_chunk = overlap_text + sent
                chunk_start = chunk_start + len(current_chunk) - len(overlap_text) - len(sent)
            else:
                current_chunk = current_chunk + sent if current_chunk else sent

        if current_chunk.strip():
            chunks.append((current_chunk.strip(), chunk_start, chunk_start + len(current_chunk)))

        return chunks

    def _split_to_l3(
        self, content: str, doc_id: str, parent_id: str, section: Section
    ) -> List[ChunkNode]:
        if not content or not content.strip():
            return []

        paragraphs = self._split_into_paragraphs(content)
        l3_nodes = []
        current_chunk = ""
        chunk_start = section.start_pos

        for para in paragraphs:
            if len(current_chunk) + len(para) + 1 > self.l3_max_size and current_chunk:
                node = ChunkNode(
                    node_id=f"L3_{uuid.uuid4().hex[:12]}",
                    level=3,
                    content=current_chunk.strip(),
                    title=self._extract_title_from_text(current_chunk) or section.title,
                    parent_id=parent_id,
                    metadata={"full_title": section.title},
                    doc_id=doc_id,
                    start_char=chunk_start,
                    end_char=chunk_start + len(current_chunk),
                )
                l3_nodes.append(node)

                overlap_text = current_chunk[-self.overlap:] if self.overlap > 0 else ""
                current_chunk = overlap_text + para
                chunk_start = section.start_pos + len(content) - len(current_chunk)
            else:
                current_chunk = current_chunk + "\n" + para if current_chunk else para

        if current_chunk.strip():
            node = ChunkNode(
                node_id=f"L3_{uuid.uuid4().hex[:12]}",
                level=3,
                content=current_chunk.strip(),
                title=self._extract_title_from_text(current_chunk) or section.title,
                parent_id=parent_id,
                metadata={"full_title": section.title},
                doc_id=doc_id,
                start_char=chunk_start,
                end_char=chunk_start + len(current_chunk),
            )
            l3_nodes.append(node)

        l3_nodes = self._merge_short_l3_nodes(l3_nodes, parent_id, doc_id, section)

        return l3_nodes

    def _merge_short_l3_nodes(
        self, nodes: List[ChunkNode], parent_id: str, doc_id: str, section: Section
    ) -> List[ChunkNode]:
        if not nodes:
            return nodes

        merged = []
        i = 0
        while i < len(nodes):
            if i < len(nodes) - 1 and len(nodes[i].content) < self.l3_min_size:
                combined = nodes[i].content + "\n" + nodes[i + 1].content
                if len(combined) <= self.l3_max_size * 1.5:
                    merged_node = ChunkNode(
                        node_id=f"L3_{uuid.uuid4().hex[:12]}",
                        level=3,
                        content=combined,
                        title=nodes[i].title,
                        parent_id=parent_id,
                        metadata={"full_title": section.title, "merged": True},
                        doc_id=doc_id,
                        start_char=nodes[i].start_char,
                        end_char=nodes[i + 1].end_char,
                    )
                    merged.append(merged_node)
                    i += 2
                    continue
            merged.append(nodes[i])
            i += 1

        return merged

    def _split_into_paragraphs(self, text: str) -> List[str]:
        paragraphs = re.split(r"\n\s*\n", text)
        result = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(para) > self.l3_max_size * 1.5:
                sentences = re.split(r"(?<=[。！？.!?])\s*", para)
                current = ""
                for sent in sentences:
                    if len(current) + len(sent) > self.l3_max_size and current:
                        result.append(current.strip())
                        overlap_text = current[-self.overlap:] if self.overlap > 0 else ""
                        current = overlap_text + sent
                    else:
                        current = current + sent if current else sent
                if current.strip():
                    result.append(current.strip())
            else:
                result.append(para)
        return result

    def _split_content_to_sections(self, content: str, max_size: int) -> List[Section]:
        sections = []
        paragraphs = self._split_into_paragraphs(content)
        current_text = ""
        start = 0

        for para in paragraphs:
            if len(current_text) + len(para) + 1 > max_size and current_text:
                sections.append(
                    Section(
                        title="",
                        level=2,
                        content=current_text.strip(),
                        start_pos=start,
                        end_pos=start + len(current_text),
                    )
                )
                start += len(current_text)
                current_text = para
            else:
                current_text = current_text + "\n" + para if current_text else para

        if current_text.strip():
            sections.append(
                Section(
                    title="",
                    level=2,
                    content=current_text.strip(),
                    start_pos=start,
                    end_pos=start + len(current_text),
                )
            )
        return sections

    def _expand_flat_sections(self, sections: List[Section], text: str) -> List[Section]:
        expanded = []
        for sec in sections:
            content = sec.content or text
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]

            if not paragraphs:
                expanded.append(sec)
                continue

            if len(paragraphs) == 1 and len(content) <= self.l2_max_size:
                expanded.append(sec)
                continue

            l2_groups = []
            current_group = []
            current_len = 0
            for para in paragraphs:
                if current_len + len(para) > self.l2_max_size and current_group:
                    l2_groups.append("\n".join(current_group))
                    current_group = [para]
                    current_len = len(para)
                else:
                    current_group.append(para)
                    current_len += len(para)
            if current_group:
                l2_groups.append("\n".join(current_group))

            for i, group_text in enumerate(l2_groups):
                title = f"{sec.title} - 第{i+1}部分" if len(l2_groups) > 1 else sec.title
                l2_sec = Section(
                    title=title,
                    level=2,
                    content=group_text,
                    start_pos=sec.start_pos,
                    end_pos=sec.end_pos,
                )
                expanded.append(l2_sec)

                sentences = re.split(r"(?<=[。！？.!?\n])\s*", group_text)
                sentences = [s.strip() for s in sentences if s.strip()]

                l3_chunks = []
                current_chunk = ""
                for sent in sentences:
                    if len(current_chunk) + len(sent) > self.l3_max_size and current_chunk:
                        l3_chunks.append(current_chunk.strip())
                        current_chunk = sent
                    else:
                        current_chunk = current_chunk + sent if current_chunk else sent
                if current_chunk.strip():
                    l3_chunks.append(current_chunk.strip())

                for chunk_text in l3_chunks:
                    expanded.append(Section(
                        title=title,
                        level=3,
                        content=chunk_text,
                        start_pos=sec.start_pos,
                        end_pos=sec.end_pos,
                    ))

        return expanded

    def _extract_title_from_text(self, text: str) -> str:
        first_line = text.strip().split("\n")[0]
        if len(first_line) > 50:
            return first_line[:50] + "..."
        return first_line
