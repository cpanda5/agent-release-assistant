import time


class StreamingHandler:
    """
    Wraps an OpenAI-compatible async stream.

    Iterate to get text chunks. After the async-for loop exits, the following
    attributes are populated:
      full_content  — complete assembled response
      usage         — {"input_tokens": N, "output_tokens": N} (if API provides it)
      ttfb_ms       — time to first byte in ms
      finish_reason — stop / length / content_filter / …
    """

    def __init__(self, raw_stream):
        self._stream = raw_stream
        self.full_content = ""
        self.usage: dict = {}
        self.ttfb_ms: int | None = None
        self.finish_reason: str | None = None
        self._start = time.monotonic()
        self._first_chunk = True

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        async for chunk in self._stream:
            if chunk.choices:
                choice = chunk.choices[0]
                delta = (choice.delta.content or "") if choice.delta else ""
                if delta:
                    if self._first_chunk:
                        self.ttfb_ms = int((time.monotonic() - self._start) * 1000)
                        self._first_chunk = False
                    self.full_content += delta
                    yield delta
                if choice.finish_reason:
                    self.finish_reason = choice.finish_reason

            # usage arrives in the last chunk when stream_options.include_usage=True
            if hasattr(chunk, "usage") and chunk.usage:
                self.usage = {
                    "input_tokens": getattr(chunk.usage, "prompt_tokens", 0) or 0,
                    "output_tokens": getattr(chunk.usage, "completion_tokens", 0) or 0,
                }
