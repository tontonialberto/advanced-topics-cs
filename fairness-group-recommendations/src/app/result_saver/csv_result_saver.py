from pathlib import Path
from typing import Any, Callable, List
from app.domain.result_saver import ResultSaver


class CsvResultSaver(ResultSaver):
    def __init__(self, writer: Callable[[Path, str], Any]):
        self.__writer = writer

    def save(self, output_path: Path, headers: List[str], rows: List[List[str]]) -> None:
        content = ",".join(headers) + "\n"
        content += "\n".join([",".join(row) for row in rows]) + "\n"
        self.__writer(output_path, content)