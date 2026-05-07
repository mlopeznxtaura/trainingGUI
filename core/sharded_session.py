"""
core/sharded_session.py — Multi-node session sharding (gap_01)
Extends InferenceSession to distribute inference across N adapters.
Supports round_robin and input_split strategies via config.
"""
import threading
import itertools
from datetime import datetime
from core.session import InferenceSession


class ShardedSession(InferenceSession):

    def __init__(self, adapters: list, config: dict):
        super().__init__(adapters[0], config)
        self.adapters = adapters
        self._shard_strategy = config.get("scaling", {}).get("sharding_strategy", "round_robin")
        self._rr_counter = itertools.cycle(range(len(adapters)))
        self._rr_lock = threading.Lock()

    def run(self, inputs: dict) -> dict:
        strategy = self._shard_strategy

        if strategy == "round_robin":
            with self._rr_lock:
                idx = next(self._rr_counter)
            result = self.adapters[idx].run_inference(inputs)

        elif strategy == "input_split":
            input_list = inputs.get("inputs", [inputs])
            n = len(self.adapters)
            chunks = [input_list[i::n] for i in range(n)]
            out = [None] * n

            def _run(adapter, chunk, i):
                out[i] = adapter.run_inference({"inputs": chunk})

            threads = [threading.Thread(target=_run, args=(a, c, i)) for i, (a, c) in enumerate(zip(self.adapters, chunks))]
            for t in threads: t.start()
            for t in threads: t.join()

            merged = []
            latency = 0.0
            for r in out:
                if r:
                    o = r.get("output", [])
                    merged.extend(o if isinstance(o, list) else [o])
                    latency = max(latency, r.get("latency_ms", 0.0))
            result = {"output": merged, "latency_ms": latency}

        else:
            result = self.adapters[0].run_inference(inputs)

        record = {"timestamp": datetime.utcnow().isoformat(), "inputs": inputs,
                  "output": result.get("output"), "latency_ms": result.get("latency_ms", 0.0), "strategy": strategy}
        with self._lock:
            self.history.append(record)
        for fn in self._callbacks["on_output"]:
            try: fn(result)
            except Exception: pass
        return result

    def stream(self, inputs: dict):
        import queue
        q = queue.Queue()
        sentinel = object()
        n = len(self.adapters)
        tokens = []

        def _stream_adapter(adapter, idx):
            try:
                for token in adapter.stream_tokens(inputs):
                    q.put(token)
            except Exception as e:
                q.put(f"[shard {idx} error: {e}]")
            finally:
                q.put(sentinel)

        threads = [threading.Thread(target=_stream_adapter, args=(a, i)) for i, a in enumerate(self.adapters)]
        for t in threads: t.start()

        active = n
        while active > 0:
            item = q.get()
            if item is sentinel:
                active -= 1
            else:
                tokens.append(item)
                for fn in self._callbacks["on_token"]:
                    try: fn(item)
                    except Exception: pass
                yield item

        for t in threads: t.join()

        with self._lock:
            self.history.append({"timestamp": datetime.utcnow().isoformat(), "inputs": inputs,
                                  "output": "".join(tokens), "latency_ms": 0.0, "type": "stream_summary"})
