# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generate architecture proposal benchmark dataset for linear LangGraph benchmarking.

Produces 100 architecture review scenarios across 5 architecture styles (20 each):
microservices, monolithic, serverless, event-driven, and hybrid.

Usage:
    cd examples/dynamo_integration/linear_langgraph_benchmark
    python generate_dataset.py
"""

import json
import random
from pathlib import Path

random.seed(42)

# ---------------------------------------------------------------------------
# Variable pools
# ---------------------------------------------------------------------------
INDUSTRIES = [
    "fintech",
    "healthcare",
    "e-commerce",
    "media-streaming",
    "logistics",
    "gaming",
    "edtech",
    "insurtech",
    "govtech",
    "adtech",
]

COMPANY_NAMES = [
    "NovaPay",
    "MedCore",
    "ShopStream",
    "PixelVault",
    "FreightWave",
    "PlayForge",
    "LearnBridge",
    "ClaimSync",
    "CivicEdge",
    "AdPulse",
    "FinLedger",
    "HealthGrid",
    "CartFlow",
    "StreamBase",
    "RouteLink",
    "GuildNet",
    "EduSphere",
    "RiskCanvas",
    "GovStack",
    "BidLogic",
    "PayPath",
    "CareSync",
    "MerchHub",
    "MediaFlux",
    "CargoNet",
    "ArenaVerse",
    "SkillForge",
    "PolicyIQ",
    "DataGov",
    "ClickField",
    "CoinBridge",
    "VitalLink",
    "DealHive",
    "ContentRay",
    "TrackOps",
    "QuestCore",
    "TutorGrid",
    "CoverMesh",
    "PermitHub",
    "ReachMetric",
    "LedgerSync",
    "PharmaNet",
    "StoreLogic",
    "ViewStream",
    "FleetPulse",
    "PixelArena",
    "ClassPath",
    "UnderwriteAI",
    "RegConnect",
    "ImpressAd",
]

CLOUD_PROVIDERS = ["aws", "gcp", "azure", "multi-cloud"]

DB_TYPES = [
    "postgresql",
    "mysql",
    "mongodb",
    "dynamodb",
    "cosmosdb",
    "aurora",
    "cockroachdb",
    "cassandra",
]

TRAFFIC_RPS = [1000, 5000, 10000, 25000, 50000, 100000]

STORAGE_TB = [5, 10, 25, 50, 100, 500]

TEAM_SIZES = [8, 15, 25, 40, 60, 100]

COMPLIANCE_REQS = [
    ["SOC2"],
    ["SOC2", "GDPR"],
    ["HIPAA", "SOC2"],
    ["PCI-DSS", "SOC2"],
    ["GDPR", "HIPAA"],
    ["SOC2", "GDPR", "PCI-DSS"],
]

REGIONS = ["us-east", "us-west", "eu-west", "ap-southeast", "multi-region"]

CACHE_TYPES = ["redis", "memcached", "elasticache", "hazelcast"]

QUEUE_TYPES = ["kafka", "rabbitmq", "sqs", "pulsar", "nats"]

# ---------------------------------------------------------------------------
# Template functions  (each returns the question string)
# ---------------------------------------------------------------------------


def _microservices_template(company, industry, cloud, db, traffic, storage, team, compliance, region, cache, queue):
    comp_str = ", ".join(compliance)
    return (f"Architecture Review Request for {company}\n\n"
            f"Company Background:\n"
            f"{company} is a rapidly growing {industry} company that has been operating for three years "
            f"and recently closed a Series C funding round. The engineering organization has scaled from a "
            f"small founding team to {team} engineers distributed across multiple product squads. The "
            f"company's core platform serves both B2B enterprise clients and a growing B2C user base. "
            f"Revenue has doubled year-over-year and the board is pushing for aggressive international "
            f"expansion within the next 18 months.\n\n"
            f"System Architecture Overview:\n"
            f"The platform is built on a microservices architecture deployed on {cloud} in the {region} "
            f"region. The system currently comprises 45 independently deployable services communicating "
            f"through a combination of synchronous REST/gRPC calls and asynchronous event-driven "
            f"messaging. The API gateway layer handles request routing, authentication, and rate limiting "
            f"for all external traffic. Internal service-to-service communication is secured via mTLS "
            f"through a service mesh. The primary message queue infrastructure uses {queue} for "
            f"event-driven workflows including order processing, notification delivery, and analytics "
            f"data pipelines. The persistence layer is anchored by a {db} cluster handling transactional "
            f"data, supplemented by dedicated datastores for search, time-series metrics, and document "
            f"storage. A distributed {cache} caching layer sits in front of the database tier to reduce "
            f"read latency and protect against traffic spikes. Container orchestration is managed through "
            f"Kubernetes with auto-scaling policies tied to CPU, memory, and custom application metrics.\n\n"
            f"Data Flow Description:\n"
            f"Incoming requests hit the API gateway, which performs JWT validation and routes to the "
            f"appropriate service. Write operations are processed synchronously and then published as "
            f"domain events to {queue} topics. Downstream consumers handle secondary concerns such as "
            f"updating read models, triggering notifications, and feeding the analytics pipeline. The "
            f"analytics pipeline aggregates events into the data warehouse on a near-real-time basis with "
            f"a maximum lag of 30 seconds. Cache invalidation follows an event-driven pattern where "
            f"mutations publish invalidation events consumed by the caching service.\n\n"
            f"Current Scale and Traffic:\n"
            f"The system handles a sustained {traffic:,} requests per second during peak hours with "
            f"burst capacity provisioned for 3x that load. Total data under management is approximately "
            f"{storage} TB across all datastores. The {queue} cluster processes roughly 2 million events "
            f"per hour during normal operations. Average API response latency at the p50 is 45ms and "
            f"p99 is 380ms.\n\n"
            f"Growth Projections:\n"
            f"With the planned international expansion, the engineering team expects traffic to grow 4x "
            f"within 12 months. The data footprint is projected to reach {storage * 5} TB. New market "
            f"entry will require multi-region deployment capabilities and data residency compliance. The "
            f"team plans to onboard 30 additional engineers.\n\n"
            f"Team Structure:\n"
            f"The current {team}-person engineering team is organized into domain-aligned squads: "
            f"platform infrastructure, payments, user experience, data engineering, and security. Each "
            f"squad owns between 5 and 12 services. There is a shared platform team responsible for the "
            f"service mesh, CI/CD pipelines, and observability stack.\n\n"
            f"Compliance Requirements:\n"
            f"The platform must maintain {comp_str} compliance. Audit logs must be immutable and retained "
            f"for seven years. All PII must be encrypted at rest and in transit. Regular penetration "
            f"testing and vulnerability scanning are mandated quarterly.\n\n"
            f"Technical Concerns:\n"
            f"1. Service-to-service call chains are reaching 8-12 hops for some critical user flows, "
            f"causing cascading latency.\n"
            f"2. The {queue} cluster is showing increasing consumer lag during peak periods.\n"
            f"3. Database connection pool exhaustion occurs sporadically under sustained high load.\n"
            f"4. Distributed tracing coverage is only at 60% of services, making root cause analysis "
            f"difficult.\n"
            f"5. The team is concerned about data consistency across services during partial failures.\n\n"
            f"Please review this architecture and provide recommendations for scalability, reliability, "
            f"and operational excellence.")


def _monolithic_template(company, industry, cloud, db, traffic, storage, team, compliance, region, cache, queue):
    comp_str = ", ".join(compliance)
    return (f"Architecture Review Request for {company}\n\n"
            f"Company Background:\n"
            f"{company} is an established {industry} company that has been operating for over eight years. "
            f"The platform was originally built as a monolithic application by a small founding team and "
            f"has grown organically as the business scaled. The company serves approximately 2 million "
            f"monthly active users and processes significant transaction volume. The engineering team of "
            f"{team} engineers maintains the core application along with a handful of ancillary services "
            f"that were extracted over the past two years.\n\n"
            f"System Architecture Overview:\n"
            f"The core application is a monolithic deployment running on {cloud} in the {region} region. "
            f"The application is packaged as a single deployable unit behind an L7 load balancer that "
            f"handles SSL termination, health checking, and traffic distribution across multiple "
            f"application instances. The primary datastore is a {db} database with read replicas "
            f"configured for query-heavy workloads. A {cache} caching layer provides session management "
            f"and frequently accessed data caching with a current hit rate of approximately 78%. A CDN "
            f"serves static assets and caches API responses for public endpoints with a 15-minute TTL. "
            f"The monitoring stack includes centralized logging, application performance monitoring, and "
            f"infrastructure metrics dashboards. Background job processing is handled by a task queue "
            f"backed by {queue}.\n\n"
            f"Data Flow Description:\n"
            f"All traffic enters through the load balancer and is routed to one of the application "
            f"instances. The application handles request processing, business logic execution, and "
            f"database operations within a single process. Read-heavy endpoints first check the {cache} "
            f"cache and fall back to the database on cache miss. Write operations go directly to the "
            f"primary {db} instance and trigger asynchronous cache invalidation. Background jobs such "
            f"as report generation, email delivery, and data aggregation are queued and processed by "
            f"dedicated worker instances running the same application codebase. Static assets and public "
            f"API responses are served through the CDN edge network.\n\n"
            f"Current Scale and Traffic:\n"
            f"The platform handles {traffic:,} requests per second at peak with an average response "
            f"time of 120ms at p50 and 850ms at p99. The {db} database holds {storage} TB of data with "
            f"the largest tables containing over 500 million rows. The task queue processes approximately "
            f"50,000 background jobs per hour. Deployment cadence is twice weekly with a 45-minute "
            f"deployment window during low-traffic periods.\n\n"
            f"Growth Projections:\n"
            f"The business is projecting 2.5x user growth over the next year driven by a new product "
            f"line launch. Traffic is expected to reach {traffic * 3:,} requests per second. The "
            f"database is projected to grow to {storage * 3} TB. The company plans to add {team // 2} "
            f"more engineers, nearly doubling the team size.\n\n"
            f"Team Structure:\n"
            f"The {team}-person team is organized into feature teams, but all teams work within the same "
            f"codebase. There is a dedicated DevOps team of 3 engineers managing infrastructure and "
            f"deployments. Code reviews require approval from at least one member of the architecture "
            f"board, which consists of 4 senior engineers.\n\n"
            f"Compliance Requirements:\n"
            f"The platform must maintain {comp_str} compliance. Database backups are taken every 6 hours "
            f"with point-in-time recovery enabled. Access controls are enforced at the application layer "
            f"with role-based permissions. All sensitive data fields are encrypted using application-level "
            f"encryption.\n\n"
            f"Technical Concerns:\n"
            f"1. Deployment risk is high because every release deploys the entire application, and "
            f"rollbacks take 30 minutes.\n"
            f"2. The monolithic codebase has grown to over 800,000 lines of code and build times exceed "
            f"25 minutes.\n"
            f"3. Database query performance is degrading as table sizes grow, with several queries now "
            f"exceeding 2 seconds.\n"
            f"4. The {cache} cache hit rate has been declining from 85% to 78% over the past quarter.\n"
            f"5. Testing is slow and flaky, with the full test suite taking over 90 minutes to run.\n\n"
            f"Please review this architecture and provide recommendations for modernization, scalability, "
            f"and deployment improvements.")


def _serverless_template(company, industry, cloud, db, traffic, storage, team, compliance, region, cache, queue):
    comp_str = ", ".join(compliance)
    cloud_fn_map = {
        "aws": "AWS Lambda",
        "gcp": "Google Cloud Functions",
        "azure": "Azure Functions",
        "multi-cloud": "a combination of AWS Lambda and Azure Functions",
    }
    cloud_fn = cloud_fn_map[cloud]
    return (f"Architecture Review Request for {company}\n\n"
            f"Company Background:\n"
            f"{company} is a {industry} startup that launched 18 months ago with a serverless-first "
            f"philosophy. The founding team made a deliberate architectural decision to avoid managing "
            f"infrastructure and instead rely entirely on managed cloud services. This approach allowed "
            f"the small team to move quickly and launch their MVP in under three months. The company has "
            f"since grown to {team} engineers and raised a Series B round. The platform now serves over "
            f"500,000 registered users with strong month-over-month growth.\n\n"
            f"System Architecture Overview:\n"
            f"The entire backend is built on serverless primitives running on {cloud} in the {region} "
            f"region. Core business logic is implemented as {cloud_fn} functions triggered by API "
            f"Gateway endpoints, event bus rules, and scheduled cron expressions. The API gateway "
            f"provides REST endpoints with request validation, API key management, and usage throttling. "
            f"Data persistence uses {db} as the primary datastore with managed services for search and "
            f"file storage. Event-driven workflows are orchestrated through {queue} and a managed event "
            f"bus that routes events to the appropriate function handlers. Authentication and "
            f"authorization are handled by a managed identity service with JWT-based token validation at "
            f"the API gateway level. The storage layer manages {storage} TB of data across structured "
            f"databases, object storage, and search indices.\n\n"
            f"Data Flow Description:\n"
            f"User requests arrive at the API gateway, which validates the JWT token and routes the "
            f"request to the corresponding function. Functions execute business logic, interact with "
            f"{db} for data operations, and publish events to the event bus for downstream processing. "
            f"Asynchronous workflows such as document processing, notification delivery, and data "
            f"enrichment are triggered by event rules and executed as separate function invocations. "
            f"File uploads go directly to object storage via pre-signed URLs, which trigger processing "
            f"functions upon upload completion. The analytics pipeline consumes events from the event "
            f"bus, transforms them, and loads them into the data warehouse on a streaming basis.\n\n"
            f"Current Scale and Traffic:\n"
            f"The platform handles {traffic:,} API requests per second at peak. The function fleet "
            f"executes approximately 50 million invocations per day. Average function duration is 180ms "
            f"with a cold start rate of approximately 8%. The {db} datastore handles 15,000 read and "
            f"5,000 write operations per second. Monthly cloud spend is approximately $45,000 with "
            f"compute accounting for 60% of the total.\n\n"
            f"Growth Projections:\n"
            f"The company expects 5x growth in user base over the next year as they expand into new "
            f"markets. API traffic is projected to reach {traffic * 5:,} requests per second. Data "
            f"storage is expected to grow to {storage * 4} TB. The team plans to grow to "
            f"{team + 25} engineers.\n\n"
            f"Team Structure:\n"
            f"The {team}-person team is organized into small, autonomous squads of 3-4 engineers each. "
            f"Each squad owns a bounded domain and its associated functions, events, and data. There is "
            f"no dedicated operations team; all engineers practice a you-build-it-you-run-it model. A "
            f"small platform team maintains shared libraries, deployment tooling, and observability "
            f"infrastructure.\n\n"
            f"Compliance Requirements:\n"
            f"The platform must maintain {comp_str} compliance. All function execution logs are retained "
            f"for 12 months. Data at rest and in transit is encrypted using cloud-managed keys. Access "
            f"to production resources requires MFA and is audited.\n\n"
            f"Technical Concerns:\n"
            f"1. Cold start latency is causing p99 response times to spike to 3.5 seconds for "
            f"user-facing endpoints.\n"
            f"2. Debugging distributed function chains is extremely difficult with current observability "
            f"tooling.\n"
            f"3. The 15-minute function execution timeout is limiting the ability to process large batch "
            f"operations.\n"
            f"4. Vendor lock-in is a growing concern as the architecture relies heavily on {cloud}-"
            f"specific managed services.\n"
            f"5. Cost is becoming less predictable as traffic patterns are highly variable and "
            f"auto-scaling can lead to unexpected spend.\n\n"
            f"Please review this architecture and provide recommendations for performance optimization, "
            f"observability, and cost management.")


def _event_driven_template(company, industry, cloud, db, traffic, storage, team, compliance, region, cache, queue):
    comp_str = ", ".join(compliance)
    return (f"Architecture Review Request for {company}\n\n"
            f"Company Background:\n"
            f"{company} is a data-intensive {industry} company that specializes in real-time analytics "
            f"and decision-making. Founded five years ago, the company processes massive volumes of "
            f"streaming data to provide actionable insights to enterprise customers. The platform ingests "
            f"data from hundreds of sources including APIs, IoT devices, webhooks, and batch file "
            f"uploads. The engineering team of {team} engineers has deep expertise in distributed systems "
            f"and stream processing. The company recently signed several Fortune 500 customers and needs "
            f"to scale accordingly.\n\n"
            f"System Architecture Overview:\n"
            f"The platform is built on an event-driven architecture centered around {queue} as the "
            f"core streaming platform, deployed on {cloud} in the {region} region. All data enters the "
            f"system as events published to {queue} topics partitioned by source, type, and customer. "
            f"Stream processors consume events, apply transformations, enrichment, and business rules, "
            f"then publish derived events back to the streaming platform. The data lake stores raw and "
            f"processed events in columnar format partitioned by time and source for efficient "
            f"analytical queries. The analytics engine provides sub-second query performance over "
            f"terabytes of data using a {db}-backed serving layer with pre-computed materialized views. "
            f"A real-time dashboard service pushes live updates to client applications via WebSocket "
            f"connections. A distributed {cache} layer caches frequently queried aggregations and user "
            f"session state.\n\n"
            f"Data Flow Description:\n"
            f"Data producers publish events to {queue} topics with guaranteed at-least-once delivery. "
            f"The ingestion layer validates event schemas and routes them to the appropriate processing "
            f"pipelines. Stream processors operate in consumer groups to enable parallel processing and "
            f"fault tolerance. Processed events are written to both the data lake for historical analysis "
            f"and the serving layer for real-time queries. The materialized view engine continuously "
            f"updates pre-aggregated datasets as new events arrive, ensuring dashboards reflect data no "
            f"more than 5 seconds old. Alert rules evaluate streaming data against configurable "
            f"thresholds and trigger notifications through multiple channels.\n\n"
            f"Current Scale and Traffic:\n"
            f"The streaming platform processes {traffic:,} events per second sustained, with burst "
            f"capacity for 5x during peak periods. The data lake holds {storage} TB of historical data "
            f"with approximately 2 TB ingested daily. The {queue} cluster runs 120 partitions across "
            f"24 brokers. End-to-end processing latency from ingestion to dashboard update averages "
            f"3.2 seconds at p50 and 8.5 seconds at p99. The serving layer handles 5,000 concurrent "
            f"dashboard sessions.\n\n"
            f"Growth Projections:\n"
            f"With the new Fortune 500 customers, event volume is expected to reach {traffic * 4:,} "
            f"events per second within 6 months. The data lake will grow to {storage * 8} TB. The "
            f"number of concurrent dashboard sessions is expected to triple. The team plans to grow "
            f"from {team} to {team + 35} engineers.\n\n"
            f"Team Structure:\n"
            f"The {team}-person engineering team is split into four groups: data ingestion and "
            f"integration, stream processing and analytics, platform reliability, and customer-facing "
            f"applications. The platform reliability team manages the {queue} clusters, {db} "
            f"infrastructure, and monitoring. Each team operates with significant autonomy but "
            f"coordinates through weekly architecture reviews.\n\n"
            f"Compliance Requirements:\n"
            f"The platform must maintain {comp_str} compliance. All events must be immutable once "
            f"ingested. Data retention policies must be configurable per customer with minimum 3-year "
            f"retention for regulated industries. Personally identifiable information must be tokenized "
            f"at ingestion time.\n\n"
            f"Technical Concerns:\n"
            f"1. Consumer lag on critical processing pipelines is increasing, with some consumers "
            f"falling behind by over 2 minutes during peak periods.\n"
            f"2. The data lake query performance degrades significantly for queries spanning more than "
            f"30 days of data.\n"
            f"3. Exactly-once processing semantics are not fully guaranteed, leading to occasional "
            f"duplicate records in the serving layer.\n"
            f"4. The real-time dashboard WebSocket infrastructure struggles with more than 3,000 "
            f"concurrent connections per node.\n"
            f"5. Schema evolution across producers is causing compatibility issues in downstream "
            f"consumers.\n\n"
            f"Please review this architecture and provide recommendations for scaling the streaming "
            f"platform, improving data processing guarantees, and enhancing real-time capabilities.")


def _hybrid_template(company, industry, cloud, db, traffic, storage, team, compliance, region, cache, queue):
    comp_str = ", ".join(compliance)
    return (f"Architecture Review Request for {company}\n\n"
            f"Company Background:\n"
            f"{company} is a well-established {industry} company with over 15 years of operational "
            f"history. The company has a significant on-premises infrastructure footprint built up over "
            f"many years, including physical data centers in two geographic locations. A cloud migration "
            f"initiative was started two years ago, but regulatory constraints, data sovereignty "
            f"requirements, and legacy system dependencies have necessitated a hybrid approach. The "
            f"engineering team of {team} engineers is a mix of traditional infrastructure specialists "
            f"and cloud-native developers. The company serves government and enterprise clients with "
            f"strict uptime SLAs of 99.99%.\n\n"
            f"System Architecture Overview:\n"
            f"The platform runs a hybrid architecture spanning on-premises data centers and {cloud} "
            f"cloud resources in the {region} region. Core transactional systems and sensitive data "
            f"processing remain on-premises, running on {db} clusters in the primary and disaster "
            f"recovery data centers. Cloud-burst capacity on {cloud} handles traffic overflow during "
            f"peak periods, batch processing workloads, and non-sensitive analytical workloads. A "
            f"dedicated VPN tunnel with redundant connections provides secure connectivity between "
            f"on-premises and cloud environments with sub-10ms latency. Data synchronization between "
            f"on-premises and cloud uses a bidirectional replication pipeline built on {queue} with "
            f"conflict resolution logic. The {cache} caching layer is deployed in both environments "
            f"with cross-site cache coherence for frequently accessed reference data. Disaster recovery "
            f"is configured as active-passive with automated failover triggered by health check "
            f"failures, targeting a recovery time objective of under 15 minutes.\n\n"
            f"Data Flow Description:\n"
            f"Client requests enter through a global traffic manager that routes based on request type "
            f"and data classification. Requests involving sensitive regulated data are routed to the "
            f"on-premises environment, while general compute and analytics requests are directed to "
            f"the cloud. Write operations on-premises are captured via change data capture and replicated "
            f"to the cloud {db} replica with an average lag of 500ms. Cloud-side workloads that need "
            f"regulated data access it through a secure API proxy that enforces data classification "
            f"policies and maintains an audit trail. Batch processing jobs are submitted to the cloud "
            f"environment, pulling anonymized datasets from on-premises, processing them, and pushing "
            f"results back through the secure replication pipeline.\n\n"
            f"Current Scale and Traffic:\n"
            f"The combined platform handles {traffic:,} requests per second, with approximately 60% "
            f"served on-premises and 40% in the cloud. Total data under management is {storage} TB, "
            f"split roughly 70/30 between on-premises and cloud. The VPN link carries approximately "
            f"500 Mbps of sustained traffic. On-premises hardware is at 75% capacity during peak hours. "
            f"The cloud environment auto-scales between 20 and 200 instances based on load.\n\n"
            f"Growth Projections:\n"
            f"The company aims to shift the traffic split to 30% on-premises and 70% cloud within "
            f"18 months. Total traffic is expected to grow to {traffic * 2:,} requests per second. "
            f"Data volume will reach {storage * 4} TB. A new regulatory approval is expected that will "
            f"allow more workloads to move to the cloud. The team plans to hire {team // 3} additional "
            f"cloud-focused engineers.\n\n"
            f"Team Structure:\n"
            f"The {team}-person engineering team is divided between an on-premises infrastructure group, "
            f"a cloud engineering group, a networking and security group, and application development "
            f"teams. The two infrastructure groups are currently siloed with different tooling, "
            f"processes, and monitoring systems. A newly formed cloud center of excellence is working "
            f"to standardize practices across both environments.\n\n"
            f"Compliance Requirements:\n"
            f"The platform must maintain {comp_str} compliance. Data sovereignty requirements mandate "
            f"that certain data categories never leave the on-premises environment. All cross-environment "
            f"data transfers must be logged and auditable. Disaster recovery testing must be conducted "
            f"quarterly with documented results. Encryption must use customer-managed keys for data "
            f"at rest.\n\n"
            f"Technical Concerns:\n"
            f"1. The VPN link is a single point of contention, and latency spikes during peak "
            f"replication cause data synchronization delays of up to 30 seconds.\n"
            f"2. Inconsistent monitoring and alerting between on-premises and cloud environments makes "
            f"incident response fragmented.\n"
            f"3. The disaster recovery failover process has only been tested twice and both tests "
            f"revealed issues that took over 45 minutes to resolve.\n"
            f"4. Data classification enforcement is manual and error-prone, with three incidents in the "
            f"past year where regulated data was inadvertently processed in the cloud.\n"
            f"5. The on-premises hardware refresh cycle is creating budget pressure, with several "
            f"servers approaching end-of-life within 12 months.\n\n"
            f"Please review this architecture and provide recommendations for improving the hybrid "
            f"connectivity, data governance, and disaster recovery posture.")


# ---------------------------------------------------------------------------
# Scenario generation helpers
# ---------------------------------------------------------------------------

ARCH_STYLES = [
    ("microservices", _microservices_template),
    ("monolithic", _monolithic_template),
    ("serverless", _serverless_template),
    ("event_driven", _event_driven_template),
    ("hybrid", _hybrid_template),
]

RISK_INDICATORS_MAP = {
    "microservices": "cascading latency, consumer lag, connection pool exhaustion",
    "monolithic": "deployment risk, codebase complexity, query degradation",
    "serverless": "cold start latency, vendor lock-in, cost unpredictability",
    "event_driven": "consumer lag, duplicate processing, schema evolution issues",
    "hybrid": "VPN contention, data sync delays, DR failover gaps",
}


def _pick(lst):
    """Pick a random element from a list."""
    return random.choice(lst)


def _generate_scenarios():
    """Generate 100 architecture review scenarios (20 per architecture style)."""
    scenarios = []
    scenario_id = 1

    # Shuffle company names so assignments vary
    company_pool = list(COMPANY_NAMES)
    random.shuffle(company_pool)
    company_idx = 0

    for arch_style, template_fn in ARCH_STYLES:
        for _ in range(20):
            company = company_pool[company_idx % len(company_pool)]
            company_idx += 1

            industry = _pick(INDUSTRIES)
            cloud = _pick(CLOUD_PROVIDERS)
            db = _pick(DB_TYPES)
            traffic = _pick(TRAFFIC_RPS)
            storage = _pick(STORAGE_TB)
            team = _pick(TEAM_SIZES)
            compliance = _pick(COMPLIANCE_REQS)
            region = _pick(REGIONS)
            cache = _pick(CACHE_TYPES)
            queue = _pick(QUEUE_TYPES)

            question = template_fn(
                company=company,
                industry=industry,
                cloud=cloud,
                db=db,
                traffic=traffic,
                storage=storage,
                team=team,
                compliance=compliance,
                region=region,
                cache=cache,
                queue=queue,
            )

            comp_str = ", ".join(compliance)
            risk_indicators = RISK_INDICATORS_MAP[arch_style]
            ground_truth = (f"Expected: {arch_style} architecture, "
                            f"{comp_str} requirements, "
                            f"{risk_indicators}")

            entry = {
                "id": f"arch_review_{scenario_id:03d}",
                "question": question,
                "ground_truth": ground_truth,
                "metadata": {
                    "expected_architecture_style": arch_style,
                    "expected_industry": industry,
                    "expected_cloud": cloud,
                    "expected_min_llm_calls": 7,
                },
            }
            scenarios.append(entry)
            scenario_id += 1

    return scenarios


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Generate and write benchmark dataset files."""
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "src" / "linear_langgraph_benchmark" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    scenarios = _generate_scenarios()

    full_path = data_dir / "linear_langgraph_benchmark.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(scenarios, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(scenarios)} entries to {full_path}")

    mini_path = data_dir / "linear_langgraph_benchmark_mini.json"
    with open(mini_path, "w", encoding="utf-8") as f:
        json.dump(scenarios[:3], f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(scenarios[:3])} entries to {mini_path}")


if __name__ == "__main__":
    main()
