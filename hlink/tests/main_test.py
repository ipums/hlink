from hlink.scripts.main import _read_history_file


def test_read_history_file_does_not_exist(tmp_path) -> None:
    """
    _read_history_file() does not raise an exception if the history file does
    not exist.
    """
    history_file = tmp_path / ".history_notthere"
    _read_history_file(history_file)
