from pathlib import Path
from app.domain.data_loader import DataLoader
from app.domain.dataset import Dataset


class FileDataLoader(DataLoader):
    def __init__(self, path: Path) -> None:
        self.__path = path
        
    def load(self) -> Dataset:
        data = []
        with open(self.__path, 'r') as file:
            next(file)  # Ignore the header line
            for line in file:
                values = line.strip().split(',')
                if len(values) >= 3:
                    try:
                        user_id = int(values[0])
                        item_id = int(values[1])
                        rating = float(values[2])
                        data.append((user_id, item_id, rating))
                    except ValueError:
                        print(f"WARN: skipping invalid line '{line}'")
                        pass
        dataset = Dataset(data)
        return dataset