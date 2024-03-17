from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock

from app.result_saver.csv_result_saver import CsvResultSaver


class TestCsvResultSaver(TestCase):
    def test_save(self) -> None:
        writer = Mock()
        saver = CsvResultSaver(writer)
        
        saver.save(Path("output.csv"), ["userId", "itemId", "rating"], [["1", "1", "5"], ["1", "2", "4"], ["2", "1", "3"]])
        
        writer.assert_called_once_with(Path("output.csv"), "userId,itemId,rating\n1,1,5\n1,2,4\n2,1,3\n")