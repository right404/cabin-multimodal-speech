from pathlib import Path

from cabin_speech.annotations import read_ami_participants, read_ami_speech_intervals


def test_read_ami_annotations(tmp_path: Path) -> None:
    corpus = tmp_path / "corpusResources"
    words = tmp_path / "words"
    segments = tmp_path / "segments"
    corpus.mkdir()
    words.mkdir()
    segments.mkdir()
    (corpus / "meetings.xml").write_text(
        """<root><meeting observation="M1"><speaker nxt_agent="A" channel="0"
        camera="Closeup1" global_name="P1" role="PM"/></meeting></root>""",
        encoding="utf-8",
    )
    (words / "M1.A.words.xml").write_text(
        """<root><w starttime="1.0" endtime="1.2">Hello</w>
        <w starttime="1.2" endtime="1.2" punc="true">,</w>
        <w starttime="1.3" endtime="1.6">world</w></root>""",
        encoding="utf-8",
    )
    (segments / "M1.A.segments.xml").write_text(
        """<root><segment transcriber_start="0.95" transcriber_end="1.65"/></root>""",
        encoding="utf-8",
    )

    participants = read_ami_participants(corpus / "meetings.xml", "M1")
    intervals = read_ami_speech_intervals(tmp_path, "M1")
    assert participants["A"].camera_id == "Closeup1"
    assert participants["A"].headset_channel == 0
    assert len(intervals) == 1
    assert intervals[0].text == "Hello, world"
