---
draft: true
title: "Behavioural Design Patterns"
date: 2023-10-24
tags: ["OOP", "Design Patters", "UML"]
# images:
#   - /images/banners/content/programming/0300-oop/0309-behavioural-design-patterns.png
---














<!-- ## Design Patterns Leveraging Polymorphism

### Observer Pattern

```python
from abc import ABC, abstractmethod
from typing import List, Any

class Observer(ABC):
    @abstractmethod
    def update(self, subject: 'Subject') -> None:
        pass

class Subject:
    def __init__(self):
        self._observers: List[Observer] = []
        self._state: Any = None

    def attach(self, observer: Observer) -> None:
        self._observers.append(observer)

    def detach(self, observer: Observer) -> None:
        self._observers.remove(observer)

    def notify(self) -> None:
        for observer in self._observers:
            observer.update(self)

    @property
    def state(self) -> Any:
        return self._state

    @state.setter
    def state(self, value: Any) -> None:
        self._state = value
        self.notify()

class ConcreteObserverA(Observer):
    def update(self, subject: Subject) -> None:
        print(f"ConcreteObserverA: Reacted to state change: {subject.state}")

class ConcreteObserverB(Observer):
    def update(self, subject: Subject) -> None:
        print(f"ConcreteObserverB: Reacted to state change: {subject.state}")
```

### Template Method Pattern

```python
class DataMiner(ABC):
    def mine_data(self, path: str) -> None:
        """Template method defining the data mining algorithm."""
        data = self._extract_data(path)
        text = self._parse_data(data)
        self._analyze_data(text)
        self._send_report()

    @abstractmethod
    def _extract_data(self, path: str) -> bytes:
        pass

    @abstractmethod
    def _parse_data(self, data: bytes) -> str:
        pass

    def _analyze_data(self, text: str) -> None:
        print(f"Analysing data: {text[:100]}...")

    def _send_report(self) -> None:
        print("Sending report...")

class PDFDataMiner(DataMiner):
    def _extract_data(self, path: str) -> bytes:
        print(f"Extracting PDF data from {path}")
        return b"PDF data"

    def _parse_data(self, data: bytes) -> str:
        return f"Parsed PDF content: {data.decode()}"

class CSVDataMiner(DataMiner):
    def _extract_data(self, path: str) -> bytes:
        print(f"Extracting CSV data from {path}")
        return b"CSV data"

    def _parse_data(self, data: bytes) -> str:
        return f"Parsed CSV content: {data.decode()}"
``` -->
