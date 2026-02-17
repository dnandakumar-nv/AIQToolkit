#!/usr/bin/env python3
"""
Sustained high-ISL / low-OSL load generator for inference router testing.
Sends >10,000 tokens of unique input per request and decodes only 256 tokens.
Maintains N concurrent requests at all times. Ctrl-C to stop and print metrics.

Usage: python3 blast_high_isl.py [--concurrent 12] [--model meta-llama/Llama-3.1-8B-Instruct] [--max-tokens 256]
"""

import argparse
import asyncio
import aiohttp
import hashlib
import json
import random
import signal
import statistics
import string
import sys
import time
import uuid
from dataclasses import dataclass, field

# Long, unique passages used as building blocks for high-ISL prompts.
# Each block is ~300-400 words; we concatenate several plus random padding
# to exceed 10k tokens per request while keeping inputs mostly unique.
PASSAGE_BLOCKS = [
    (
        "The development of semiconductor fabrication has undergone remarkable evolution since the invention "
        "of the transistor at Bell Labs in 1947. Early germanium-based transistors gave way to silicon planar "
        "technology in the 1960s, enabling Gordon Moore's famous observation about doubling transistor density. "
        "The progression from 10-micron processes in the early 1970s to today's sub-3nm nodes represents over "
        "three orders of magnitude improvement. Each generation demanded new lithographic techniques: contact "
        "printing, projection lithography, deep ultraviolet (DUV) with 248nm and 193nm excimer lasers, immersion "
        "lithography achieving effective NAs above 1.0, and finally extreme ultraviolet (EUV) lithography at "
        "13.5nm wavelength. The transition to EUV required decades of research into tin-droplet laser-produced "
        "plasma sources, multilayer Mo/Si reflective optics, and hydrogen-based pellicle membranes. Meanwhile, "
        "patterning innovations like self-aligned double and quadruple patterning (SADP/SAQP) extended 193nm "
        "immersion lithography far beyond its expected limits. FinFET transistor architectures introduced at "
        "22nm by Intel in 2011 replaced planar MOSFETs, wrapping the gate around three sides of the channel "
        "to improve electrostatic control. Gate-all-around (GAA) nanosheet transistors represent the next "
        "evolution, with Samsung and TSMC adopting them at 3nm and 2nm nodes respectively. Backend interconnect "
        "scaling has proven equally challenging, with copper dual-damascene processes facing increasing resistivity "
        "at narrow linewidths, driving research into ruthenium, cobalt, and molybdenum alternatives. The economic "
        "implications are staggering: a modern leading-edge fab costs upward of $20 billion, and a single EUV "
        "scanner from ASML exceeds $350 million. This capital intensity has consolidated leading-edge manufacturing "
        "to just three companies globally. The geopolitical ramifications of this concentration have become a "
        "central concern for national security strategists worldwide."
    ),
    (
        "Distributed consensus algorithms form the backbone of modern fault-tolerant systems. Lamport's Paxos, "
        "published in 1998 but conceived much earlier, established the theoretical foundation by proving that "
        "consensus is achievable in asynchronous systems with crash failures, provided a majority of nodes remain "
        "operational. The algorithm operates in two phases: a prepare phase where a proposer claims a ballot number "
        "and learns of any previously accepted values, and an accept phase where the proposer asks acceptors to "
        "commit to a value. Multi-Paxos optimizes the common case by electing a stable leader who can skip the "
        "prepare phase for consecutive log entries, amortizing leadership cost. However, Paxos's notorious "
        "difficulty in implementation led to the development of Raft by Ongaro and Ousterhout in 2014. Raft "
        "decomposes consensus into leader election, log replication, and safety, making each component more "
        "understandable. The leader appends entries to followers' logs in order, and entries are committed once "
        "replicated on a majority. Leader election uses randomized timeouts to break symmetry. Byzantine fault "
        "tolerance (BFT) extends consensus to adversarial settings. Castro and Liskov's PBFT (1999) tolerates "
        "up to f Byzantine faults with 3f+1 replicas, using a three-phase protocol (pre-prepare, prepare, commit) "
        "with O(n^2) message complexity. Modern BFT variants like HotStuff (2019) achieve linear message complexity "
        "through threshold signatures and a three-phase chained protocol, forming the basis of blockchain consensus "
        "in systems like Meta's Diem (formerly Libra). The CAP theorem, proven by Gilbert and Lynch in 2002, "
        "establishes that no distributed system can simultaneously provide consistency, availability, and partition "
        "tolerance. This has profound implications for database design, leading to the emergence of 'CP' systems "
        "like ZooKeeper and etcd, 'AP' systems like Cassandra and DynamoDB, and hybrid approaches like CockroachDB "
        "that offer serializable isolation while gracefully degrading under partitions."
    ),
    (
        "Protein folding represents one of the grand challenges of computational biology. The central dogma of "
        "molecular biology establishes that DNA is transcribed into mRNA, which is translated into amino acid "
        "sequences. These linear chains of amino acids then fold into specific three-dimensional structures "
        "determined by thermodynamic principles. Anfinsen's dogma, established through his Nobel Prize-winning "
        "work on ribonuclease in the 1960s, posits that a protein's native structure is determined solely by its "
        "amino acid sequence under physiological conditions. The protein folding problem — predicting 3D structure "
        "from sequence — remained largely unsolved for decades despite significant effort. Levinthal's paradox "
        "highlights the challenge: a 100-residue protein has roughly 3^198 possible conformations, yet folds in "
        "milliseconds to seconds, implying that folding follows specific pathways rather than random search. Energy "
        "landscape theory, developed by Wolynes, Onuchic, and Bryngelson, models folding as diffusion on a "
        "funnel-shaped energy surface where native contacts progressively stabilize the structure. The Critical "
        "Assessment of protein Structure Prediction (CASP) competition, founded by John Moult in 1994, provided "
        "a rigorous benchmark for the field. For decades, homology modeling (using known structures of related "
        "proteins as templates) dominated. Physics-based approaches like Rosetta, developed by David Baker's lab, "
        "combined coarse-grained energy functions with Monte Carlo fragment assembly. The watershed moment came "
        "at CASP13 in 2018 when DeepMind's AlphaFold achieved unprecedented accuracy, and at CASP14 in 2020 when "
        "AlphaFold2 effectively solved the single-chain structure prediction problem, achieving median GDT-TS "
        "scores above 90. AlphaFold2's architecture uses a novel 'Evoformer' module that processes multiple "
        "sequence alignments (MSAs) and pairwise residue features through attention mechanisms, followed by a "
        "structure module that iteratively refines 3D coordinates. The subsequent release of AlphaFold-Multimer "
        "extended predictions to protein complexes, and the AlphaFold Protein Structure Database now contains "
        "predictions for over 200 million proteins."
    ),
    (
        "The history of cartography reflects humanity's evolving understanding of geography and mathematical "
        "representation. The earliest known maps, dating to approximately 2300 BCE, are Babylonian clay tablets "
        "depicting local features and irrigation canals. Greek cartography achieved remarkable sophistication: "
        "Eratosthenes calculated Earth's circumference around 240 BCE with surprising accuracy using shadow "
        "angles at Alexandria and Syene. Ptolemy's Geographia (circa 150 CE) introduced the concepts of latitude "
        "and longitude and provided coordinates for some 8,000 locations, though systematic errors in his estimate "
        "of Earth's size (about 18% too small) would later encourage Columbus's westward voyage. Medieval European "
        "mappae mundi, such as the Hereford Map (circa 1300), prioritized theological narrative over geographic "
        "accuracy, placing Jerusalem at the center. Islamic cartographers like al-Idrisi produced far more "
        "geographically accurate works; his Tabula Rogeriana (1154) remained influential for centuries. The Age "
        "of Exploration demanded practical navigational charts. Portolan charts, originating in 13th-century "
        "Italy, depicted coastlines with remarkable precision using compass roses and rhumb lines. Mercator's "
        "1569 world map introduced the conformal cylindrical projection that preserves angles — essential for "
        "navigation — at the cost of dramatically distorting area at high latitudes. The 18th and 19th centuries "
        "saw the rise of systematic national surveys: the Ordnance Survey in Britain (founded 1791), the Survey "
        "of India (whose Great Trigonometrical Survey measured the height of Everest), and the US Geological "
        "Survey (founded 1879). Thematic cartography emerged with works like John Snow's cholera map of 1854 "
        "London, which pioneered spatial epidemiology. The 20th century brought aerial photography, satellite "
        "imagery, and Geographic Information Systems (GIS). Today, platforms like OpenStreetMap leverage "
        "crowdsourced data, while LiDAR and InSAR provide centimeter-resolution topographic measurements."
    ),
    (
        "Compiler optimization passes transform intermediate representations to produce faster, smaller machine "
        "code. The field traces back to Frances Allen's pioneering work at IBM on control flow analysis and "
        "optimization in the 1960s and 1970s, for which she received the Turing Award in 2006. Modern optimizing "
        "compilers like LLVM, GCC, and MSVC implement dozens of transformation passes organized in carefully "
        "tuned pipelines. Scalar optimizations include constant folding and propagation, dead code elimination, "
        "common subexpression elimination (CSE), strength reduction (replacing expensive operations with cheaper "
        "equivalents), and algebraic simplification. Loop optimizations are particularly important for numerical "
        "code: loop-invariant code motion (LICM) hoists computations that don't change across iterations, loop "
        "unrolling reduces branch overhead, loop fusion combines adjacent loops over the same range to improve "
        "locality, and loop tiling (blocking) partitions iteration spaces to fit working sets in cache. "
        "Vectorization transforms scalar operations into SIMD instructions (SSE, AVX, NEON), requiring dependence "
        "analysis to ensure correctness. Polyhedral compilation, implemented in tools like ISL and Pluto, provides "
        "a mathematical framework for reasoning about loop transformations using integer linear programming. "
        "Interprocedural optimizations span function boundaries: inlining eliminates call overhead and enables "
        "further local optimizations, while link-time optimization (LTO) extends this across translation units. "
        "Profile-guided optimization (PGO) uses runtime profiles to inform decisions about inlining thresholds, "
        "branch prediction hints, and hot/cold code partitioning. Register allocation, typically formulated as "
        "graph coloring, assigns variables to physical registers while minimizing spills to memory. Instruction "
        "scheduling reorders operations to hide latency and maximize pipeline utilization, often using list "
        "scheduling with priority heuristics. The interaction between passes creates phase-ordering problems where "
        "the best sequence of optimizations depends on the specific program, motivating research into machine "
        "learning-guided compilation and iterative optimization."
    ),
    (
        "The thermodynamics of black holes represents one of the most profound connections between gravity, "
        "quantum mechanics, and information theory. Bekenstein's insight in 1972 that black holes must carry "
        "entropy proportional to their horizon area — rather than volume — challenged conventional thermodynamic "
        "intuitions. Hawking's semiclassical calculation in 1974 showed that black holes radiate thermally at "
        "a temperature inversely proportional to their mass: T = ℏc³/(8πGMkB), confirming the thermodynamic "
        "analogy and establishing the four laws of black hole mechanics as genuine thermodynamic laws. The "
        "Bekenstein-Hawking entropy formula S = A/(4ℓP²), where ℓP is the Planck length, implies that the maximum "
        "entropy of a region of space scales with its boundary area rather than its volume — the holographic "
        "principle, formalized by 't Hooft and Susskind. The black hole information paradox, arising from the "
        "apparent conflict between unitary quantum evolution and Hawking's thermal radiation spectrum, has driven "
        "decades of theoretical research. Proposed resolutions include black hole complementarity (Susskind, "
        "Thorlacius, Uglum, 1993), which posits that information is both reflected at the horizon and passes "
        "through it, with no single observer witnessing both copies; the firewall argument (AMPS, 2012), which "
        "challenges complementarity by showing that maintaining entanglement between early and late Hawking "
        "radiation requires either a breakdown of effective field theory at the horizon or the presence of a "
        "high-energy 'firewall'; and the island formula from quantum extremal surfaces, which provides a "
        "gravitational path integral derivation of the Page curve showing entropy initially rising then falling "
        "as a black hole evaporates. The ER=EPR conjecture by Maldacena and Susskind proposes that quantum "
        "entanglement between particles is geometrically realized as Einstein-Rosen bridges (wormholes), "
        "suggesting a deep connection between spacetime geometry and quantum information. These developments "
        "have made black hole physics a central arena for testing candidate theories of quantum gravity."
    ),
    (
        "Industrial fermentation has shaped human civilization from ancient brewing to modern biomanufacturing. "
        "Archaeological evidence suggests beer production in Mesopotamia dates to at least 3500 BCE, with the "
        "Sumerian Hymn to Ninkasi serving as both prayer and brewing recipe. Wine fermentation was practiced in "
        "the South Caucasus by 6000 BCE. Louis Pasteur's studies in the 1850s and 1860s established that "
        "fermentation was a biological process driven by living microorganisms, overturning the prevailing theory "
        "of spontaneous generation. His work on 'diseases' of wine and beer led to pasteurization and the germ "
        "theory of disease. The 20th century transformed fermentation from art to engineering. The production of "
        "acetone-butanol by Clostridium acetobutylicum during World War I (the Weizmann process) demonstrated "
        "strategic industrial applications. Penicillin production during World War II drove the development of "
        "submerged fermentation in stirred-tank bioreactors, replacing the inefficient surface culture method. "
        "This required advances in sterile engineering, aeration, agitation, and process control. The recombinant "
        "DNA revolution of the 1970s enabled heterologous protein expression: human insulin produced in E. coli "
        "(Genentech/Lilly, 1982) became the first recombinant pharmaceutical. Chinese hamster ovary (CHO) cells "
        "emerged as the dominant platform for monoclonal antibody production, with fed-batch processes reaching "
        "titers exceeding 10 g/L. Metabolic engineering, pioneered by Jay Bailey, applies genetic modifications "
        "guided by stoichiometric models to redirect cellular metabolism toward desired products. Recent advances "
        "in synthetic biology enable the construction of entirely novel biosynthetic pathways: artemisinin "
        "precursor production in engineered yeast (Keasling lab), opioid biosynthesis, and cellular agriculture "
        "for cultivated meat. Modern bioreactors range from 2,000-liter single-use systems for biologics to "
        "500,000-liter tanks for amino acid fermentation. Process analytical technology (PAT) employs Raman "
        "spectroscopy, dielectric spectroscopy, and soft sensors for real-time monitoring, moving toward Industry "
        "4.0 digital twins for bioprocess optimization."
    ),
    (
        "The evolution of memory hierarchy in computer architecture reflects the fundamental tradeoff between "
        "speed, capacity, and cost. Von Neumann's original architecture assumed a single uniform memory, but the "
        "growing disparity between processor and memory speeds — the 'memory wall' identified by Wulf and McKee "
        "in 1995 — necessitated increasingly sophisticated caching strategies. The first hardware cache appeared "
        "in the IBM System/360 Model 85 in 1968. Modern processors feature multi-level cache hierarchies: L1 "
        "caches (typically 32-64 KB, 4-5 cycle latency), L2 caches (256 KB - 1 MB, 12-15 cycles), and L3 caches "
        "(shared, 8-64 MB, 30-50 cycles). Cache design involves numerous tradeoffs: direct-mapped caches minimize "
        "hit latency but suffer from conflict misses; set-associative caches (typically 4-16 way) balance hit time "
        "and miss rate; replacement policies like LRU, pseudo-LRU, and RRIP determine eviction decisions. Cache "
        "coherence protocols ensure consistency in multiprocessor systems: snooping protocols (MESI, MOESI) "
        "broadcast invalidations on a shared bus, while directory-based protocols scale to larger systems by "
        "tracking sharers at a central directory. Intel's MESIF and AMD's MOESI variants optimize for different "
        "sharing patterns. Prefetching hides memory latency by fetching data before it is requested: hardware "
        "stride prefetchers detect regular access patterns, while software prefetch instructions (e.g., x86 "
        "PREFETCHT0) allow programmer-guided prefetching. Non-volatile memory technologies are reshaping the "
        "hierarchy: Intel's Optane (3D XPoint) offered byte-addressable persistent memory with latencies between "
        "DRAM and SSD, while Compute Express Link (CXL) enables memory pooling and sharing across hosts. "
        "Processing-in-memory (PIM) architectures like Samsung's HBM-PIM and UPMEM's DPUs move computation "
        "closer to data, addressing the bandwidth wall. The emergence of chiplet-based designs with heterogeneous "
        "memory configurations — combining HBM, LPDDR, and CXL-attached memory — creates increasingly complex "
        "NUMA topologies that challenge operating system memory management and application optimization."
    ),
]

# Additional random filler words to pad and diversify input
FILLER_DOMAINS = [
    "quantum computing", "neural architecture search", "climate modeling", "genomic sequencing",
    "supply chain optimization", "natural language processing", "autonomous vehicles", "drug discovery",
    "materials science", "astrophysics simulations", "financial modeling", "robotics control systems",
    "network security analysis", "ocean current simulation", "epidemiological modeling",
    "computational fluid dynamics", "molecular dynamics", "signal processing", "game theory",
    "operations research", "topological data analysis", "Bayesian optimization", "causal inference",
    "reinforcement learning", "federated learning", "graph neural networks", "diffusion models",
    "sparse matrix computation", "tensor decomposition", "spectral methods",
]

QUESTIONS = [
    "Summarize the above in exactly one sentence.",
    "What is the single most important takeaway from the above text?",
    "Reply with only the word 'done' after reading the above.",
    "In one word, what is the primary topic above?",
    "Respond with a single-paragraph abstract of the above content.",
    "Give a one-line TL;DR of the above material.",
    "List the three most important keywords from the above text.",
    "What field of study does the above text primarily concern?",
]


def _generate_unique_padding(rng: random.Random, target_words: int) -> str:
    """Generate pseudo-random but coherent-looking text to pad input length."""
    chunks = []
    words_so_far = 0
    while words_so_far < target_words:
        domain = rng.choice(FILLER_DOMAINS)
        run_id = uuid.uuid4().hex[:12]
        experiment_num = rng.randint(1, 999999)
        timestamp = rng.randint(1_600_000_000, 1_900_000_000)
        values = [f"{rng.uniform(-100, 100):.6f}" for _ in range(rng.randint(5, 15))]
        param_names = [
            ''.join(rng.choices(string.ascii_lowercase, k=rng.randint(4, 10)))
            for _ in range(len(values))
        ]
        params = ", ".join(f"{n}={v}" for n, v in zip(param_names, values))

        chunk = (
            f"[Experiment {experiment_num} | run={run_id} | ts={timestamp} | domain={domain}] "
            f"Parameters: {params}. "
            f"Observations: The measured throughput for {domain} workload was {rng.uniform(0.1, 1000):.4f} "
            f"ops/sec with a latency of {rng.uniform(0.001, 500):.4f}ms at batch size {rng.choice([1, 2, 4, 8, 16, 32, 64, 128, 256])}. "
            f"Configuration hash: {hashlib.sha256(run_id.encode()).hexdigest()[:32]}. "
            f"Notes: Variance across {rng.randint(3, 50)} trials was {rng.uniform(0.0001, 10):.6f} with "
            f"confidence interval [{rng.uniform(0.8, 0.95):.4f}, {rng.uniform(0.95, 0.999):.4f}]. "
            f"The control group using standard {rng.choice(FILLER_DOMAINS)} baseline showed "
            f"{rng.uniform(-50, 50):.2f}% deviation from expected. "
            f"Memory utilization peaked at {rng.uniform(10, 99):.1f}% with {rng.randint(1, 128)} active threads. "
        )
        chunks.append(chunk)
        words_so_far += len(chunk.split())

    return " ".join(chunks)


def build_high_isl_prompt(request_id: int, target_tokens: int = 10000) -> str:
    """Build a prompt with approximately `target_tokens` tokens of unique input content."""
    rng = random.Random(request_id)

    # Pick a random subset of passage blocks and shuffle
    selected = rng.sample(PASSAGE_BLOCKS, k=min(len(PASSAGE_BLOCKS), rng.randint(5, len(PASSAGE_BLOCKS))))
    rng.shuffle(selected)

    # Combine the curated passages
    curated_text = "\n\n".join(selected)
    curated_words = len(curated_text.split())

    # Rough heuristic: 1 token ≈ 0.75 words for English
    target_total_words = int(target_tokens * 0.75)
    padding_words_needed = max(0, target_total_words - curated_words)
    padding_text = _generate_unique_padding(rng, padding_words_needed)

    # Pick a short question to keep OSL low
    question = rng.choice(QUESTIONS)

    prompt = (
        f"REQUEST-ID: {request_id}-{uuid.uuid4().hex[:8]}\n\n"
        f"Please carefully read all of the following reference material, then answer the question at the end.\n\n"
        f"--- BEGIN REFERENCE MATERIAL ---\n\n"
        f"{curated_text}\n\n"
        f"--- SUPPLEMENTARY DATA ---\n\n"
        f"{padding_text}\n\n"
        f"--- END REFERENCE MATERIAL ---\n\n"
        f"Question: {question}"
    )
    return prompt


@dataclass
class Metrics:
    total_requests: int = 0
    completed: int = 0
    failed: int = 0
    ttfb_values: list = field(default_factory=list)
    total_latency_values: list = field(default_factory=list)
    start_time: float = 0.0
    in_flight: int = 0

    def record(self, ttfb: float, total: float, success: bool):
        self.total_requests += 1
        if success:
            self.completed += 1
            self.ttfb_values.append(ttfb)
            self.total_latency_values.append(total)
        else:
            self.failed += 1

    def print_report(self):
        elapsed = time.time() - self.start_time
        print("\n")
        print("=" * 60)
        print("  HIGH-ISL / LOW-OSL LOAD GENERATOR METRICS")
        print("=" * 60)
        print(f"  Duration:           {elapsed:.1f}s")
        print(f"  Total requests:     {self.total_requests}")
        print(f"  Completed:          {self.completed}")
        print(f"  Failed:             {self.failed}")
        print(f"  Throughput:         {self.completed / elapsed:.2f} req/s")
        print()

        if self.ttfb_values:
            self._print_latency_stats("TTFB", self.ttfb_values)
            self._print_latency_stats("Total Latency", self.total_latency_values)

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
        self.osl_min = args.osl_min
        self.osl_max = args.osl_max
        self.target_tokens = args.target_isl
        self.concurrent = args.concurrent
        self.metrics = Metrics()
        self.shutting_down = False
        self.request_counter = 0

    def next_request_id(self) -> int:
        self.request_counter += 1
        return self.request_counter

    def build_payload(self, prompt: str, max_tokens: int) -> dict:
        return {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "nvext": {
                "agent_hints": {
                    "latency_sensitivity": 0
                }
            }
        }

    async def send_request(self, session: aiohttp.ClientSession, req_id: int):
        prompt = build_high_isl_prompt(req_id, self.target_tokens)
        max_tokens = random.randint(self.osl_min, self.osl_max)
        payload = self.build_payload(prompt, max_tokens)
        headers = {"Content-Type": "application/json"}

        t_start = time.time()
        ttfb = None
        success = False

        try:
            async with session.post(self.url, json=payload, headers=headers) as resp:
                async for chunk in resp.content.iter_any():
                    if ttfb is None:
                        ttfb = time.time() - t_start
                success = resp.status == 200
        except Exception as e:
            if not self.shutting_down:
                print(f"\n  [req {req_id}] Error: {e}", file=sys.stderr)

        t_total = time.time() - t_start
        if ttfb is None:
            ttfb = t_total

        self.metrics.record(ttfb, t_total, success)
        return success

    async def worker(self, session: aiohttp.ClientSession, worker_id: int):
        """Each worker maintains exactly one in-flight request, replacing on completion."""
        while not self.shutting_down:
            req_id = self.next_request_id()
            self.metrics.in_flight += 1
            await self.send_request(session, req_id)
            self.metrics.in_flight -= 1

    async def run(self):
        self.metrics.start_time = time.time()

        loop = asyncio.get_event_loop()
        stop_event = asyncio.Event()

        def on_signal():
            self.shutting_down = True
            stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, on_signal)

        connector = aiohttp.TCPConnector(limit=self.concurrent + 10)
        timeout = aiohttp.ClientTimeout(total=600)

        # Estimate prompt size
        sample_prompt = build_high_isl_prompt(0, self.target_tokens)
        est_words = len(sample_prompt.split())
        est_tokens = int(est_words / 0.75)

        print(f"=== High-ISL / Low-OSL Load Generator ===")
        print(f"  Target:       {self.url}")
        print(f"  Model:        {self.model}")
        print(f"  Concurrent:   {self.concurrent}")
        print(f"  OSL range:    [{self.osl_min}, {self.osl_max}] tokens (output)")
        print(f"  Target ISL:   {self.target_tokens} tokens")
        print(f"  Est. ISL:     ~{est_tokens} tokens (~{est_words} words)")
        print(f"==========================================")
        print(f"  Press Ctrl-C to stop and print metrics\n")

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            workers = [
                asyncio.create_task(self.worker(session, i))
                for i in range(self.concurrent)
            ]

            async def print_status():
                while not self.shutting_down:
                    elapsed = time.time() - self.metrics.start_time
                    rps = self.metrics.completed / elapsed if elapsed > 0 else 0
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

            await stop_event.wait()

            status_task.cancel()
            for w in workers:
                w.cancel()

            await asyncio.gather(*workers, return_exceptions=True)

        self.metrics.print_report()


def main():
    parser = argparse.ArgumentParser(description="Sustained high-ISL / low-OSL load generator")
    parser.add_argument(
        "--concurrent", "-n", type=int, default=12, help="Number of concurrent requests to maintain (default: 12)"
    )
    parser.add_argument(
        "--model", "-m", type=str, default="qwen3", help="Model name"
    )
    parser.add_argument(
        "--url", type=str, default="http://localhost:8099/v1/chat/completions", help="API endpoint URL"
    )
    parser.add_argument(
        "--osl-min", type=int, default=256, help="Minimum output tokens per request (default: 64)"
    )
    parser.add_argument(
        "--osl-max", type=int, default=2048, help="Maximum output tokens per request (default: 256)"
    )
    parser.add_argument(
        "--target-isl", type=int, default=7000, help="Target input tokens per request (default: 10000)"
    )
    args = parser.parse_args()
    asyncio.run(LoadGenerator(args).run())


if __name__ == "__main__":
    main()
