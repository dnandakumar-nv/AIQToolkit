#!/usr/bin/env python3
"""
Sustained high-OSL load generator for inference router testing.
Maintains N concurrent requests at all times. Ctrl-C to stop and print metrics.

Usage: python3 blast_sustained.py [--concurrent 30] [--model meta-llama/Llama-3.1-8B-Instruct] [--max-tokens 4096] [--priority 0]
"""

import argparse
import asyncio
import aiohttp
import json
import time
import signal
import sys
import statistics
from dataclasses import dataclass, field

TOPICS = [
    "the complete history of the Roman Empire from founding to fall",
    "every major discovery in particle physics and its implications",
    "the evolution of programming languages from assembly to modern day",
    "a comprehensive guide to all known species of deep sea creatures",
    "the history of cryptography from ancient ciphers to post-quantum",
    "every major battle of World War II in chronological order with analysis",
    "the complete theory of general relativity explained from first principles",
    "a detailed history of the Silk Road trade routes and their cultural impact",
    "every United States Supreme Court landmark decision and its consequences",
    "the full history of space exploration from Sputnik to Mars missions",
    "a comprehensive overview of all major philosophical schools of thought",
    "the complete history of the internet from ARPANET to modern cloud computing",
    "every major volcanic eruption in recorded history and its global effects",
    "a detailed guide to human cellular biology and all major organ systems",
    "the history of mathematics from Babylonian numerals to category theory",
    "a comprehensive analysis of every Shakespeare play and its themes",
    "the complete history of artificial intelligence research and breakthroughs",
    "every major economic crisis in modern history and the policy responses",
    "a detailed exploration of plate tectonics and geological formations worldwide",
    "the full history of music theory from Gregorian chant to electronic music",
    "a comprehensive guide to organic chemistry reaction mechanisms",
    "the history of democracy from ancient Athens to modern representative systems",
    "every major pandemic in human history and how societies responded",
    "a detailed analysis of all major climate systems and weather patterns on Earth",
    "the complete history of naval warfare from triremes to aircraft carriers",
    "a comprehensive overview of quantum computing architectures and algorithms",
    "the full history of civil rights movements across every continent",
    "every major archaeological discovery and what it revealed about ancient civilizations",
    "a detailed guide to all branches of modern mathematics and their connections",
    "the complete history of aviation from the Wright brothers to hypersonic flight",
]


@dataclass
class Metrics:
    total_requests: int = 0
    completed: int = 0
    failed: int = 0
    # Per-phase tracking
    phase_ttfb: dict = field(default_factory=lambda: {"generate": [], "summarize": []})
    phase_total: dict = field(default_factory=lambda: {"generate": [], "summarize": []})
    phase_counts: dict = field(default_factory=lambda: {"generate": 0, "summarize": 0})
    start_time: float = 0.0
    in_flight: int = 0

    def record(self, ttfb: float, total: float, success: bool, phase: str = "generate"):
        self.total_requests += 1
        if success:
            self.completed += 1
            self.phase_ttfb[phase].append(ttfb)
            self.phase_total[phase].append(total)
            self.phase_counts[phase] += 1
        else:
            self.failed += 1

    def print_report(self):
        elapsed = time.time() - self.start_time
        print("\n")
        print("=" * 60)
        print("  LOAD GENERATOR METRICS")
        print("=" * 60)
        print(f"  Duration:           {elapsed:.1f}s")
        print(f"  Total requests:     {self.total_requests}")
        print(f"  Completed:          {self.completed}")
        print(f"  Failed:             {self.failed}")
        print(f"  Throughput:         {self.completed / elapsed:.2f} req/s")
        print(f"  Cycles (gen+sum):   {self.phase_counts['summarize']}")
        print()

        for phase in ["generate", "summarize"]:
            label = "Phase 1: Generate (high OSL)" if phase == "generate" else "Phase 2: Summarize (high ISL)"
            print(f"  --- {label} ---")
            print(f"  Count: {self.phase_counts[phase]}")
            if self.phase_ttfb[phase]:
                self._print_latency_stats("TTFB", self.phase_ttfb[phase])
                self._print_latency_stats("Total Latency", self.phase_total[phase])
            print()

        print("=" * 60)

    def _print_latency_stats(self, name: str, values: list):
        values_sorted = sorted(values)
        n = len(values_sorted)
        print(f"  {name} ({n} samples):")
        print(f"    Min:    {values_sorted[0]:.3f}s")
        print(f"    P25:    {values_sorted[int(n * 0.25)]:.3f}s")
        print(f"    P50:    {statistics.median(values_sorted):.3f}s")
        print(f"    P75:    {values_sorted[int(n * 0.75)]:.3f}s")
        print(f"    P95:    {values_sorted[int(n * 0.95)]:.3f}s")
        print(f"    P99:    {values_sorted[min(int(n * 0.99), n - 1)]:.3f}s")
        print(f"    Max:    {values_sorted[-1]:.3f}s")
        print(f"    Mean:   {statistics.mean(values_sorted):.3f}s")
        print()


class LoadGenerator:
    def __init__(self, args):
        self.url = args.url
        self.model = args.model
        self.max_tokens = args.max_tokens
        self.concurrent = args.concurrent
        self.priority = args.priority
        self.priority_header = args.priority_header
        self.metrics = Metrics()
        self.shutting_down = False
        self.topic_idx = 0

    def next_topic(self) -> str:
        topic = TOPICS[self.topic_idx % len(TOPICS)]
        self.topic_idx += 1
        return topic

    def build_payload(self, topic: str) -> dict:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": 0.7,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Write an extremely long, detailed, and comprehensive essay "
                        f"of at least 10000 words about {topic}. Cover every subtopic "
                        f"exhaustively. Do not summarize or abbreviate. Include specific "
                        f"dates, names, places, and technical details throughout. "
                        f"Every paragraph should introduce new information."
                    ),
                }
            ],
            "nvext": {
                "agent_hints": {
                    "latency_sensitivity": 0
                }
            }
        }

    def build_summarize_payload(self, long_text: str) -> dict:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": 0.7,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Below is a very long essay. Please write an extremely detailed, "
                        f"comprehensive, chapter-by-chapter summary of every single point "
                        f"made in this essay. Do not skip any details. Expand on each point "
                        f"with your own analysis and commentary. Your summary should be "
                        f"nearly as long as the original.\n\n{long_text}"
                    ),
                }
            ],
            "nvext": {
                "agent_hints": {
                    "latency_sensitivity": 0
                }
            }
        }

    async def _do_request(
        self, session: aiohttp.ClientSession, payload: dict, headers: dict, phase: str, req_id: int
    ) -> tuple[bool, str]:
        """Send a single request, return (success, response_text)."""
        t_start = time.time()
        ttfb = None
        success = False
        response_text = ""

        try:
            async with session.post(
                self.url, json=payload, headers=headers
            ) as resp:
                chunks = []
                async for chunk in resp.content.iter_any():
                    if ttfb is None:
                        ttfb = time.time() - t_start
                    chunks.append(chunk)

                success = resp.status == 200
                if success:
                    raw = b"".join(chunks).decode("utf-8", errors="replace")
                    # Try to extract content from chat completions response
                    try:
                        data = json.loads(raw)
                        response_text = data["choices"][0]["message"]["content"]
                    except (json.JSONDecodeError, KeyError, IndexError):
                        response_text = raw
        except Exception as e:
            if not self.shutting_down:
                print(f"\n  [req {req_id}/{phase}] Error: {e}", file=sys.stderr)

        t_total = time.time() - t_start
        if ttfb is None:
            ttfb = t_total

        self.metrics.record(ttfb, t_total, success, phase)
        return success, response_text

    async def send_request(self, session: aiohttp.ClientSession, req_id: int):
        topic = self.next_topic()
        headers = {
            "Content-Type": "application/json"
        }

        # Phase 1: High OSL - generate a long essay
        payload_gen = self.build_payload(topic)
        success, long_text = await self._do_request(
            session, payload_gen, headers, "generate", req_id
        )

        if not success or self.shutting_down or len(long_text) < 100:
            return success

        # Phase 2: High ISL - summarize the long essay (long input, long output)
        payload_sum = self.build_summarize_payload(long_text)
        success2, _ = await self._do_request(
            session, payload_sum, headers, "summarize", req_id
        )

        return success and success2

    async def worker(self, session: aiohttp.ClientSession, worker_id: int):
        """Each worker maintains exactly one in-flight request, replacing on completion."""
        req_count = 0
        while not self.shutting_down:
            req_count += 1
            self.metrics.in_flight += 1
            await self.send_request(session, worker_id * 10000 + req_count)
            self.metrics.in_flight -= 1

    async def run(self):
        self.metrics.start_time = time.time()

        # Handle Ctrl-C
        loop = asyncio.get_event_loop()
        stop_event = asyncio.Event()

        def on_signal():
            self.shutting_down = True
            stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, on_signal)

        connector = aiohttp.TCPConnector(limit=self.concurrent + 10)
        timeout = aiohttp.ClientTimeout(total=600)

        print(f"=== Sustained Load Generator ===")
        print(f"  Target:     {self.url}")
        print(f"  Model:      {self.model}")
        print(f"  Concurrent: {self.concurrent}")
        print(f"  Max tokens: {self.max_tokens}")
        print(f"  Priority:   {self.priority} (header: {self.priority_header})")
        print(f"================================")
        print(f"  Press Ctrl-C to stop and print metrics\n")

        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout
        ) as session:
            # Launch workers
            workers = [
                asyncio.create_task(self.worker(session, i))
                for i in range(self.concurrent)
            ]

            # Status printer
            async def print_status():
                while not self.shutting_down:
                    elapsed = time.time() - self.metrics.start_time
                    rps = (
                        self.metrics.completed / elapsed if elapsed > 0 else 0
                    )
                    print(
                        f"\r  [{elapsed:6.0f}s] "
                        f"in_flight={self.metrics.in_flight:3d}  "
                        f"completed={self.metrics.completed:5d}  "
                        f"failed={self.metrics.failed:3d}  "
                        f"rps={rps:.2f}",
                        end="",
                        flush=True,
                    )
                    await asyncio.sleep(1)

            status_task = asyncio.create_task(print_status())

            # Wait for shutdown signal
            await stop_event.wait()

            # Cancel everything
            status_task.cancel()
            for w in workers:
                w.cancel()

            await asyncio.gather(*workers, return_exceptions=True)

        self.metrics.print_report()


def main():
    parser = argparse.ArgumentParser(description="Sustained high-OSL high-OSL load generator")
    parser.add_argument(
        "--concurrent", "-n", type=int, default=12, help="Number of concurrent requests to maintain (default: 30)"
    )
    parser.add_argument(
        "--model", "-m", type=str, default="Qwen/Qwen3-14B-FP8", help="Model name"
    )
    parser.add_argument(
        "--url", type=str, default="http://localhost:8001/v1/chat/completions", help="API endpoint URL"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048, help="Max output tokens per request (default: 4096)"
    )
    parser.add_argument(
        "--priority", "-p", type=int, default=0, help="Request priority value (default: 0 = low)"
    )
    parser.add_argument(
        "--priority-header", type=str, default="x-request-priority", help="Header name for priority (default: x-request-priority)"
    )
    args = parser.parse_args()
    asyncio.run(LoadGenerator(args).run())


if __name__ == "__main__":
    main()