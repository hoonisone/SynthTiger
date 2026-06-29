"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import io
import sys

import numpy as np

from synthtiger import utils
from synthtiger.components.component import Component


class BaseCorpus(Component):
    def __init__(
        self,
        paths=(),
        weights=(),
        min_length=None,
        max_length=None,
        charset=None,
        textcase=None,
        sampling="random",
        shard_index=0,
        num_shards=1,
    ):
        super().__init__()
        self.paths = paths
        self.weights = weights
        self.min_length = min_length
        self.max_length = max_length
        self.charset = charset
        self.textcase = textcase
        self.sampling = sampling
        self.shard_index = int(shard_index)
        self.num_shards = int(num_shards)
        if self.num_shards < 1:
            raise RuntimeError("num_shards must be >= 1")
        if self.shard_index < 0 or self.shard_index >= self.num_shards:
            raise RuntimeError(
                "shard_index must satisfy 0 <= shard_index < num_shards"
            )
        self._contents = []
        self._offsets = []
        self._counts = []
        self._probs = np.array(self.weights) / sum(self.weights)
        self._sample_orders = []
        self._sample_cursors = []
        self._charset = set()
        self._update_charset()
        self._update_contents()

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        if len(self.paths) == 0:
            raise RuntimeError("Corpus path is not specified")
        if len(self.paths) != len(self.weights):
            raise RuntimeError(
                "The number of weights does not match the number of corpus paths"
            )

        text = self._sample_text()
        text = self._random_textcase(text)
        text = meta.get("text", text)

        meta = {
            "text": text,
        }

        return meta

    def data(self, meta):
        text = meta["text"]
        return text

    def _update_charset(self):
        self._charset = set()
        if self.charset is not None:
            self._charset = utils.read_charset(self.charset)

    def _update_contents(self):
        self._contents = []
        self._offsets = []
        self._counts = []

        for path in self.paths:
            offset = 0
            count = 0
            contents = io.StringIO()
            offsets = io.BytesIO()
            offsets.write(offset.to_bytes(4, sys.byteorder, signed=False))

            with open(path, "r", encoding="utf-8") as fp:
                for line_idx, text in enumerate(fp):
                    if (
                        self.num_shards > 1
                        and line_idx % self.num_shards != self.shard_index
                    ):
                        continue

                    text = text.strip("\r\n")

                    if not self._check_length(text):
                        continue
                    if not self._check_charset(text):
                        continue

                    contents.write(text)
                    offset += len(text)
                    offsets.write(offset.to_bytes(4, sys.byteorder, signed=False))
                    count += 1

            self._contents.append(contents.getvalue())
            self._offsets.append(np.frombuffer(offsets.getvalue(), dtype=np.uint32))
            self._counts.append(count)

            contents.close()
            offsets.close()

        self._update_sampling_orders()

    def _check_length(self, text):
        if self.min_length is not None and len(text) < self.min_length:
            return False
        if self.max_length is not None and len(text) > self.max_length:
            return False
        return True

    def _check_charset(self, text):
        if self.charset is not None:
            if len(set(text) - self._charset) > 0:
                return False
        return True

    def _get_text(self, key, idx):
        start = self._offsets[key][idx]
        end = self._offsets[key][idx + 1]
        text = self._contents[key][start:end]
        return text

    def _update_sampling_orders(self):
        self._sample_orders = []
        self._sample_cursors = []

        for count in self._counts:
            if count > 0:
                order = np.random.permutation(count)
            else:
                order = np.empty(0, dtype=np.int64)
            self._sample_orders.append(order)
            self._sample_cursors.append(0)

    def _sample_key(self):
        key = np.random.choice(len(self.paths), p=self._probs)
        return key

    def _sample_idx(self, key):
        if self.sampling == "random":
            return np.random.randint(self._counts[key])

        if self.sampling != "balanced":
            raise RuntimeError(
                f"Unknown sampling mode: {self.sampling}. Use 'random' or 'balanced'."
            )

        cursor = self._sample_cursors[key]
        if cursor >= self._counts[key]:
            self._sample_orders[key] = np.random.permutation(self._counts[key])
            cursor = 0

        idx = int(self._sample_orders[key][cursor])
        self._sample_cursors[key] = cursor + 1
        return idx

    def _sample_text(self):
        key = self._sample_key()
        if self._counts[key] == 0:
            raise RuntimeError(f"There is no text: {self.paths[key]}")

        idx = self._sample_idx(key)
        text = self._get_text(key, idx)
        return text

    def _random_textcase(self, text):
        if self.textcase is None:
            return text

        textcase = self.textcase[np.random.randint(len(self.textcase))]

        if textcase == "lower":
            text = text.lower()
        if textcase == "upper":
            text = text.upper()
        if textcase == "capitalize":
            text = text.capitalize()

        return text
