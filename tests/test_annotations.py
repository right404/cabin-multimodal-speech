from pathlib import Path

from cabin_speech.annotations import (
    read_textgrid_intervals,
    select_isolated_speech,
)


def test_read_and_select_textgrid_intervals(tmp_path: Path) -> None:
    textgrid = tmp_path / "sample.TextGrid"
    textgrid.write_text(
        """
File type = "ooTextFile"
Object class = "TextGrid"
item []:
    item [1]:
        class = "IntervalTier"
        name = "speaker-a"
        intervals: size = 2
        intervals [1]:
            xmin = 1.0
            xmax = 4.0
            text = "今天开会"
        intervals [2]:
            xmin = 5.0
            xmax = 8.0
            text = "第二句话"
    item [2]:
        class = "IntervalTier"
        name = "speaker-b"
        intervals: size = 1
        intervals [1]:
            xmin = 6.0
            xmax = 7.0
            text = "重叠"
""".strip(),
        encoding="utf-8",
    )
    intervals = read_textgrid_intervals(textgrid)
    assert len(intervals) == 3
    selected = select_isolated_speech(intervals)
    assert len(selected) == 1
    assert selected[0].text == "今天开会"
