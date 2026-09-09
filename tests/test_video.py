from pathlib import Path

from src.io.video_reader import resolve_input_video


def test_resolve_input_video_existing_file(tmp_path):
    video_file = tmp_path / "input.mp4"
    video_file.write_bytes(b"fake")

    resolved = resolve_input_video(str(video_file))
    assert resolved == str(video_file)


def test_resolve_input_video_directory_single_video(tmp_path):
    video_dir = tmp_path / "input"
    video_dir.mkdir()
    video = video_dir / "clip.mp4"
    video.write_bytes(b"fake")

    resolved = resolve_input_video(str(video_dir))
    assert resolved == str(video)


def test_resolve_input_video_directory_multiple_video_raises(tmp_path):
    video_dir = tmp_path / "input"
    video_dir.mkdir()
    (video_dir / "a.mp4").write_bytes(b"fake")
    (video_dir / "b.mp4").write_bytes(b"fake")

    try:
        resolve_input_video(str(video_dir))
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass
