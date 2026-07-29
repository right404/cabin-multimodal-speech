from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SpeechInterval:
    speaker_id: str
    start_seconds: float
    end_seconds: float
    text: str

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


@dataclass(frozen=True)
class AMIParticipant:
    agent_id: str
    headset_channel: int
    camera_id: str
    global_name: str
    role: str


def read_textgrid_intervals(path: str | Path) -> list[SpeechInterval]:
    """Read interval tiers from the long-text Praat TextGrid format."""

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    speaker_id: str | None = None
    interval: dict[str, float] | None = None
    results: list[SpeechInterval] = []

    for raw_line in lines:
        line = raw_line.strip()
        name_match = re.fullmatch(r'name\s*=\s*"(.*)"', line)
        if name_match:
            speaker_id = name_match.group(1).replace('""', '"')
            continue
        if re.fullmatch(r"intervals\s*\[\d+\]:", line):
            interval = {}
            continue
        if interval is None:
            continue
        value_match = re.fullmatch(r"(xmin|xmax)\s*=\s*([-+0-9.eE]+)", line)
        if value_match:
            interval[value_match.group(1)] = float(value_match.group(2))
            continue
        text_match = re.fullmatch(r'text\s*=\s*"(.*)"', line)
        if text_match and speaker_id is not None:
            text = text_match.group(1).replace('""', '"').strip()
            if text and "xmin" in interval and "xmax" in interval:
                results.append(
                    SpeechInterval(
                        speaker_id=speaker_id,
                        start_seconds=interval["xmin"],
                        end_seconds=interval["xmax"],
                        text=text,
                    )
                )
            interval = None
    return results


def _has_lexical_content(text: str) -> bool:
    without_tags = re.sub(r"<[^>]*>", "", text)
    return bool(re.search(r"[\u4e00-\u9fffA-Za-z0-9]", without_tags))


def select_isolated_speech(
    intervals: list[SpeechInterval],
    min_duration_seconds: float = 2.0,
    max_duration_seconds: float = 8.0,
) -> list[SpeechInterval]:
    """Keep lexical intervals that do not overlap another speaker."""

    selected: list[SpeechInterval] = []
    for candidate in intervals:
        if not min_duration_seconds <= candidate.duration_seconds <= max_duration_seconds:
            continue
        if not _has_lexical_content(candidate.text):
            continue
        overlaps_other_speaker = any(
            other.speaker_id != candidate.speaker_id
            and _has_lexical_content(other.text)
            and candidate.start_seconds < other.end_seconds
            and other.start_seconds < candidate.end_seconds
            for other in intervals
        )
        if not overlaps_other_speaker:
            selected.append(candidate)
    return sorted(selected, key=lambda item: item.start_seconds)


def read_ami_participants(
    meetings_xml_path: str | Path,
    meeting_id: str,
) -> dict[str, AMIParticipant]:
    """Read AMI agent-to-headset/camera mappings for one meeting."""

    root = ET.parse(meetings_xml_path).getroot()
    meeting = next(
        (
            element
            for element in root.iter("meeting")
            if element.attrib.get("observation") == meeting_id
        ),
        None,
    )
    if meeting is None:
        raise ValueError(f"meeting {meeting_id!r} was not found")

    participants: dict[str, AMIParticipant] = {}
    for speaker in meeting.findall("speaker"):
        agent_id = speaker.attrib["nxt_agent"]
        participants[agent_id] = AMIParticipant(
            agent_id=agent_id,
            headset_channel=int(speaker.attrib["channel"]),
            camera_id=speaker.attrib["camera"],
            global_name=speaker.attrib["global_name"],
            role=speaker.attrib["role"],
        )
    return participants


def _join_ami_tokens(tokens: list[tuple[str, bool]]) -> str:
    text = ""
    for token, is_punctuation in tokens:
        token = token.strip()
        if not token:
            continue
        if is_punctuation:
            text += token
        elif text:
            text += f" {token}"
        else:
            text = token
    return text


def read_ami_speech_intervals(
    annotation_root: str | Path,
    meeting_id: str,
) -> list[SpeechInterval]:
    """Read utterance intervals and reference transcripts from AMI XML."""

    root_path = Path(annotation_root)
    results: list[SpeechInterval] = []
    for words_path in sorted((root_path / "words").glob(f"{meeting_id}.*.words.xml")):
        agent_id = words_path.name.split(".")[1]
        words: list[tuple[float, float, str, bool]] = []
        for element in ET.parse(words_path).getroot():
            if element.tag.rsplit("}", 1)[-1] != "w":
                continue
            if "starttime" not in element.attrib or "endtime" not in element.attrib:
                continue
            words.append(
                (
                    float(element.attrib["starttime"]),
                    float(element.attrib["endtime"]),
                    element.text or "",
                    element.attrib.get("punc") == "true",
                )
            )

        segments_path = root_path / "segments" / f"{meeting_id}.{agent_id}.segments.xml"
        if not segments_path.exists():
            continue
        for segment in ET.parse(segments_path).getroot():
            if segment.tag.rsplit("}", 1)[-1] != "segment":
                continue
            start = float(segment.attrib["transcriber_start"])
            end = float(segment.attrib["transcriber_end"])
            tokens = [
                (text, is_punctuation)
                for word_start, word_end, text, is_punctuation in words
                if word_end >= start - 0.05 and word_start <= end + 0.05
            ]
            transcript = _join_ami_tokens(tokens)
            if transcript:
                results.append(
                    SpeechInterval(
                        speaker_id=agent_id,
                        start_seconds=start,
                        end_seconds=end,
                        text=transcript,
                    )
                )
    return sorted(results, key=lambda item: item.start_seconds)
