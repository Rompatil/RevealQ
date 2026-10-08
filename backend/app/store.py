"""Small in-memory dataset store with TTL and size cap.

Good enough for a single-instance demo. For multi-instance production you'd
swap this for Redis or object storage behind the same interface.
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict

import pandas as pd

from .services.ingest import Schema


class DatasetStore:
    def __init__(self, ttl_seconds: int = 3600, max_items: int = 50):
        self.ttl = ttl_seconds
        self.max_items = max_items
        self._items: OrderedDict[str, tuple[float, str, pd.DataFrame, Schema]] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, name: str, df: pd.DataFrame, schema: Schema) -> str:
        dataset_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._evict()
            self._items[dataset_id] = (time.time(), name, df, schema)
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)
        return dataset_id

    def get(self, dataset_id: str) -> tuple[str, pd.DataFrame, Schema] | None:
        with self._lock:
            self._evict()
            item = self._items.get(dataset_id)
            if item is None:
                return None
            self._items.move_to_end(dataset_id)
            _, name, df, schema = item
            return name, df, schema

    def _evict(self) -> None:
        cutoff = time.time() - self.ttl
        for key in [k for k, (ts, *_rest) in self._items.items() if ts < cutoff]:
            del self._items[key]


store = DatasetStore()
