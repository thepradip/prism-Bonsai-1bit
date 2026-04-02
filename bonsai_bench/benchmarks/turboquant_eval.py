#!/usr/bin/env python3
"""
Bonsai + TurboQuant IEEE Paper Evaluation
==========================================
Comprehensive benchmark: Bonsai 1-bit weights + TurboQuant KV cache compression.
Tests 3 configs x 3 models x 5 context sizes x 40 questions.

Configs:
  - Baseline: FP16 KV cache
  - TurboQuant-Q8: 8-bit KV cache quantization
  - TurboQuant-Q4: 4-bit KV cache quantization

Metrics: TTFT, Wall Time, PP tok/s, TG tok/s, Memory, Quality (PASS/FAIL + score)
"""

import json
import os
import re
import subprocess
import sys
import time

from ..models import get_llama_cli, get_llama_bench, get_model_path, MODELS

# ── Configurations ─────────────────────────────────────────────────
KV_CONFIGS = [
    {"name": "Baseline (FP16 KV)", "short": "FP16", "ctk": "f16", "ctv": "f16"},
    {"name": "TurboQuant-Q8 KV", "short": "TQ-Q8", "ctk": "q8_0", "ctv": "q8_0"},
    {"name": "TurboQuant-Q4 KV", "short": "TQ-Q4", "ctk": "q4_0", "ctv": "q4_0"},
]

CONTEXT_SIZES = [1024, 2048, 4096, 16384, 32768]
CTX_LABELS = ["1K", "2K", "4K", "16K", "32K"]

# ── 40 Evaluation Questions ────────────────────────────────────────
# Organized by category and difficulty for IEEE-grade evaluation

EVAL_QUESTIONS = [
    # ═══ CATEGORY: Context QA (RAG) ═══════════════════════════════
    # Easy
    {"id": 1, "cat": "Context QA", "diff": "Easy", "desc": "Simple fact lookup",
     "q": 'Based on this text, answer the question.\n\nText: "Tesla reported Q4 2024 revenue of $25.7 billion. Vehicle deliveries reached 495,570 units. Energy storage deployments grew 244% year-over-year to 11.0 GWh."\n\nQuestion: How many vehicles did Tesla deliver in Q4 2024?',
     "fn": lambda a: "495" in a and ("570" in a or "000" in a)},
    {"id": 2, "cat": "Context QA", "diff": "Easy", "desc": "Date extraction",
     "q": 'From the passage: "The merger between Acme Corp and Zenith LLC was finalized on March 12, 2024, creating a combined entity valued at $8.3 billion."\n\nWhen was the merger finalized?',
     "fn": lambda a: "march 12" in a.lower() or "march, 2024" in a.lower() or "12, 2024" in a},
    # Medium
    {"id": 3, "cat": "Context QA", "diff": "Medium", "desc": "Multi-hop extraction",
     "q": 'Context: "Division A generated $340M revenue with 42% margins. Division B generated $520M with 28% margins. Division C generated $180M with 55% margins."\n\nWhich division had the highest absolute profit (revenue x margin)? Show calculation.',
     "fn": lambda a: ("division b" in a.lower() or "B" in a) and ("145" in a or "142" in a or "143" in a or "146" in a)},
    {"id": 4, "cat": "Context QA", "diff": "Medium", "desc": "Negation understanding",
     "q": 'Context: "The board approved all proposals except Proposal 7 (executive compensation revision) and Proposal 12 (stock buyback program). All other proposals passed with over 70% shareholder support."\n\nDid Proposal 7 pass? Did Proposal 5 pass?',
     "fn": lambda a: "no" in a.lower()[:50] and ("yes" in a.lower() or "pass" in a.lower())},
    # Hard
    {"id": 5, "cat": "Context QA", "diff": "Hard", "desc": "Contradiction detection",
     "q": 'Document A (Jan 15): "Server capacity will be expanded to 500 nodes by Q2."\nDocument B (Feb 20): "Due to budget constraints, the server expansion has been capped at 350 nodes maximum."\nDocument C (Mar 5): "We are on track to deliver all 500 nodes by June."\n\nIdentify contradictions and which document is likely outdated.',
     "fn": lambda a: ("contradict" in a.lower() or "inconsisten" in a.lower() or "conflict" in a.lower()) and ("350" in a or "500" in a) and ("C" in a or "document c" in a.lower())},
    {"id": 6, "cat": "Context QA", "diff": "Hard", "desc": "Refuse hallucination",
     "q": 'Based ONLY on this context, answer. If not answerable, say "CANNOT BE DETERMINED".\n\nContext: "GlobalTech has 12,000 employees across 8 countries. Their main R&D center is in Bangalore, India."\n\nQuestion: What is GlobalTech\'s annual R&D budget?',
     "fn": lambda a: "cannot" in a.lower() or "not" in a.lower() and ("determined" in a.lower() or "mentioned" in a.lower() or "found" in a.lower() or "provided" in a.lower())},

    # ═══ CATEGORY: Document QA (Finance) ══════════════════════════
    # Easy
    {"id": 7, "cat": "Document QA", "diff": "Easy", "desc": "Invoice field extraction",
     "q": 'Extract vendor name and total from: "Invoice #8847 from Cloudflare Inc., dated Dec 1, 2024. Services: CDN Pro Plan. Amount: $4,299.00. Tax: $365.42. Total: $4,664.42. Terms: Net 45."',
     "fn": lambda a: "cloudflare" in a.lower() and ("4,664" in a or "4664" in a)},
    {"id": 8, "cat": "Document QA", "diff": "Easy", "desc": "Simple ratio",
     "q": "A company has $2M revenue and $800K costs. What is the profit margin percentage?",
     "fn": lambda a: "60" in a and "%" in a},
    # Medium
    {"id": 9, "cat": "Document QA", "diff": "Medium", "desc": "Earnings sentiment",
     "q": '"We achieved record revenue of $3.2B but operating income declined 15% due to $400M in restructuring charges. Excluding one-time items, adjusted EBITDA grew 8%. We expect headwinds to persist through H1 but see strong recovery indicators for H2."\n\nClassify outlook as positive, negative, or mixed. Justify.',
     "fn": lambda a: "mixed" in a.lower() and ("restructur" in a.lower() or "headwind" in a.lower() or "decline" in a.lower())},
    {"id": 10, "cat": "Document QA", "diff": "Medium", "desc": "Compare two periods",
     "q": "FY2023: Revenue $50M, COGS $30M, OpEx $12M, Net Income $8M.\nFY2024: Revenue $62M, COGS $35M, OpEx $15M, Net Income $12M.\n\nCalculate gross margin for both years and the change in net income percentage.",
     "fn": lambda a: ("40" in a and "43" in a) or ("50" in a and "56" in a.replace("$",""))},
    # Hard
    {"id": 11, "cat": "Document QA", "diff": "Hard", "desc": "Multi-step risk analysis",
     "q": 'Analyze: "Customer concentration: Top client = 45% of revenue ($90M). Contract expires Dec 2025 with no renewal clause. Competitor launched identical product at 20% lower price. Client\'s procurement team has scheduled vendor review for Q3 2025."\n\nRate risk severity (1-10), explain, and suggest mitigation.',
     "fn": lambda a: ("8" in a or "9" in a or "10" in a or "high" in a.lower() or "critical" in a.lower() or "severe" in a.lower()) and ("diversi" in a.lower() or "mitigat" in a.lower() or "renew" in a.lower())},
    {"id": 12, "cat": "Document QA", "diff": "Hard", "desc": "DCF components",
     "q": 'Given: Free Cash Flow Year 1: $10M, Growth rate: 8% for 5 years, Terminal growth: 3%, Discount rate (WACC): 10%.\n\nCalculate the Year 2 and Year 3 free cash flows and explain what the terminal value formula is.',
     "fn": lambda a: ("10.8" in a or "10,800" in a) and ("11.66" in a or "11.664" in a or "11,664" in a or "11.6" in a) and ("terminal" in a.lower())},

    # ═══ CATEGORY: Reasoning ══════════════════════════════════════
    # Easy
    {"id": 13, "cat": "Reasoning", "diff": "Easy", "desc": "Simple deduction",
     "q": "All servers in Rack A run Linux. Server X is in Rack A. What operating system does Server X run?",
     "fn": lambda a: "linux" in a.lower()},
    {"id": 14, "cat": "Reasoning", "diff": "Easy", "desc": "Sequence pattern",
     "q": "What comes next: 2, 6, 18, 54, ?",
     "fn": lambda a: "162" in a},
    # Medium
    {"id": 15, "cat": "Reasoning", "diff": "Medium", "desc": "Causal chain",
     "q": "Event timeline:\n1. DNS provider had outage at 9:00 AM\n2. Website became unreachable at 9:02 AM\n3. Customer support tickets spiked at 9:15 AM\n4. Engineering switched to backup DNS at 9:45 AM\n5. Website restored at 9:47 AM\n\nWhat was the root cause? What was the total customer-facing downtime?",
     "fn": lambda a: "dns" in a.lower() and ("45" in a or "47" in a or "42" in a)},
    {"id": 16, "cat": "Reasoning", "diff": "Medium", "desc": "Syllogistic logic",
     "q": "Premises:\n1. All databases in production use encryption.\n2. MongoDB-5 is in production.\n3. Some encrypted databases use AES-256.\n\nIs it VALID or INVALID to conclude: 'MongoDB-5 uses AES-256'?",
     "fn": lambda a: "invalid" in a.lower()},
    # Hard
    {"id": 17, "cat": "Reasoning", "diff": "Hard", "desc": "Constraint puzzle",
     "q": "Schedule 3 meetings (A, B, C) into 5 time slots (9am-1pm). Constraints:\n- A must be before B\n- C cannot be at 9am or 1pm\n- B and C cannot be adjacent\n- A is at 9am\n\nList ALL valid schedules.",
     "fn": lambda a: ("9" in a and "a" in a.lower()) and ("b" in a.lower() and "c" in a.lower())},
    {"id": 18, "cat": "Reasoning", "diff": "Hard", "desc": "Mathematical proof",
     "q": "A database has 1 million records. A binary search takes at most log2(n) comparisons. A linear scan takes n comparisons in the worst case.\n\nHow many comparisons does each approach need for 1 million records? What is the speedup factor?",
     "fn": lambda a: ("20" in a or "19.9" in a) and ("1,000,000" in a or "1000000" in a or "million" in a.lower()) and ("50" in a and "000" in a)},

    # ═══ CATEGORY: Instruction Following ══════════════════════════
    # Easy
    {"id": 19, "cat": "Instruction", "diff": "Easy", "desc": "Format as list",
     "q": "List the 4 seasons in a numbered list. Nothing else.",
     "fn": lambda a: "1" in a and "2" in a and "3" in a and "4" in a and ("spring" in a.lower() or "winter" in a.lower())},
    {"id": 20, "cat": "Instruction", "diff": "Easy", "desc": "JSON output",
     "q": 'Convert to JSON with keys "name", "age", "city": "John Smith is 34 years old and lives in Boston." Return ONLY JSON.',
     "fn": lambda a: "{" in a and '"name"' in a and '"age"' in a and '"city"' in a},
    # Medium
    {"id": 21, "cat": "Instruction", "diff": "Medium", "desc": "Conditional logic",
     "q": 'Classify each item by the rule: if price > $50 AND category is "electronics", label "HIGH". Otherwise label "LOW".\n\n1. Headphones, $79, electronics\n2. Book, $25, education\n3. Cable, $12, electronics\n4. Tablet, $299, electronics\n5. Pen, $3, office',
     "fn": lambda a: a.lower().count("high") >= 2 and a.lower().count("low") >= 3},
    {"id": 22, "cat": "Instruction", "diff": "Medium", "desc": "Summarize in N words",
     "q": 'Summarize in EXACTLY one sentence (under 25 words): "Kubernetes orchestrates containerized applications across clusters of machines, providing automated deployment, scaling, and management of application containers, originally designed by Google and now maintained by CNCF."',
     "fn": lambda a: "kubernetes" in a.lower() and len(a.split()) < 40 and a.count(".") <= 2},
    # Hard
    {"id": 23, "cat": "Instruction", "diff": "Hard", "desc": "SQL from NL",
     "q": 'Write SQL: "Find users who signed up in 2024 and made at least 3 purchases totaling over $500, excluding cancelled orders. Tables: users(id, name, signup_date), orders(id, user_id, amount, status, created_at). Sort by total spend desc."',
     "fn": lambda a: "select" in a.lower() and "join" in a.lower() and "group by" in a.lower() and "having" in a.lower() and "2024" in a},
    {"id": 24, "cat": "Instruction", "diff": "Hard", "desc": "Code review",
     "q": 'Find the bug:\n```python\ndef merge_sorted(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i])\n            i += 1\n        else:\n            result.append(b[j])\n            j += 1\n    return result\n```\nWhat does this miss? Fix it.',
     "fn": lambda a: ("remaining" in a.lower() or "rest" in a.lower() or "leftover" in a.lower() or "append" in a.lower() or "extend" in a.lower()) and ("a[i:]" in a or "b[j:]" in a or "while i" in a.lower() or "left" in a.lower())},

    # ═══ CATEGORY: Long Context (stress test for KV cache) ════════
    # These use longer prompts to stress KV cache compression
    {"id": 25, "cat": "Long Context", "diff": "Medium", "desc": "Multi-paragraph comprehension",
     "q": """Read this entire passage and answer the question at the end.

The Global Climate Report 2024 highlighted several key findings. First, global average temperatures rose by 1.45 degrees Celsius above pre-industrial levels, marking the hottest year on record. The Arctic experienced unprecedented ice loss, with summer sea ice extent reaching a new minimum of 3.2 million square kilometers. Second, extreme weather events increased by 35% compared to the 2010-2020 average. Hurricane activity in the Atlantic basin was particularly severe, with 22 named storms and 11 reaching hurricane strength. Third, renewable energy installations grew by 42% globally, with solar capacity additions alone reaching 420 GW. China led installations with 180 GW, followed by the US at 65 GW and India at 45 GW. Fourth, global CO2 emissions actually increased by 1.2% despite renewable growth, primarily due to increased coal usage in developing nations. The report concludes that current policies put the world on track for 2.7 degrees of warming by 2100. Fifth, ocean acidification reached pH 8.05, the lowest in 26,000 years, threatening marine ecosystems. The Great Barrier Reef experienced its sixth mass bleaching event.

Question: How many named storms were there in the Atlantic, and what percentage reached hurricane strength? Also, what was the total renewable solar capacity added by China, US, and India combined?""",
     "fn": lambda a: "22" in a and ("11" in a or "50%" in a) and ("290" in a or "289" in a or "285" in a)},
    {"id": 26, "cat": "Long Context", "diff": "Hard", "desc": "Cross-reference 3 sources",
     "q": """Source 1 (HR Policy): "All employees must complete cybersecurity training annually by December 31. Non-compliance results in system access suspension."
Source 2 (IT Log): "As of January 15, 2025: 847 of 900 employees have completed training. 53 employees non-compliant."
Source 3 (Manager Email, Jan 20): "Great news - 100% of our team completed their training on time. All 900 staff are compliant."

Questions:
1. Is Source 3 accurate based on Source 2?
2. What should happen to the 53 non-compliant employees per Source 1?
3. What is the compliance rate from Source 2?""",
     "fn": lambda a: ("inaccurate" in a.lower() or "incorrect" in a.lower() or "false" in a.lower() or "not accurate" in a.lower() or "contradict" in a.lower()) and ("suspend" in a.lower() or "access" in a.lower()) and ("94" in a)},
    {"id": 27, "cat": "Long Context", "diff": "Hard", "desc": "Legal clause interpretation",
     "q": """Contract excerpt: "The Licensee shall pay royalties of 5% on net revenue for the first $10 million, 3.5% on net revenue between $10 million and $50 million, and 2% on net revenue exceeding $50 million. 'Net revenue' is defined as gross revenue minus returns, allowances, and shipping costs. The minimum annual royalty is $250,000 regardless of revenue. Payments are due quarterly within 30 days of quarter end."

If the Licensee had gross revenue of $75 million with $5 million in returns/allowances/shipping:
1. What is the net revenue?
2. Calculate the total royalty owed.
3. What is the quarterly payment amount?""",
     "fn": lambda a: ("70" in a and "million" in a.lower()) and ("1" in a and ("9" in a or "mill" in a.lower()))},

    # ═══ CATEGORY: Math & Numerical Reasoning ═════════════════════
    {"id": 28, "cat": "Math", "diff": "Easy", "desc": "Percentage calculation",
     "q": "A server cluster has 250 nodes. 15 are down for maintenance. What percentage of nodes are operational?",
     "fn": lambda a: "94" in a},
    {"id": 29, "cat": "Math", "diff": "Medium", "desc": "Compound calculation",
     "q": "Cloud costs: $0.10/hour per instance. You run 20 instances for 730 hours/month. You get a 15% reserved instance discount. What is the monthly bill?",
     "fn": lambda a: ("1241" in a or "1,241" in a or "1242" in a or "1,242" in a)},
    {"id": 30, "cat": "Math", "diff": "Hard", "desc": "Optimization",
     "q": "A load balancer distributes requests to 3 servers: A (capacity 100 req/s), B (capacity 150 req/s), C (capacity 200 req/s). Total incoming: 360 req/s. Distribute proportionally to capacity. How many req/s does each server handle?",
     "fn": lambda a: "80" in a and "120" in a and "160" in a},

    # ═══ CATEGORY: Code Understanding ═════════════════════════════
    {"id": 31, "cat": "Code", "diff": "Easy", "desc": "Output prediction",
     "q": 'What does this print?\n```python\nx = [1, 2, 3, 4, 5]\nprint(x[1:4])\nprint(x[-2:])\n```',
     "fn": lambda a: "[2, 3, 4]" in a and "[4, 5]" in a},
    {"id": 32, "cat": "Code", "diff": "Medium", "desc": "Time complexity",
     "q": "What is the time complexity of this code?\n```python\ndef find(arr, target):\n    for i in range(len(arr)):\n        for j in range(i+1, len(arr)):\n            if arr[i] + arr[j] == target:\n                return (i, j)\n    return None\n```\nExpress in Big-O notation and explain why.",
     "fn": lambda a: "O(n" in a and ("2" in a or "^2" in a or "**2" in a or "squared" in a.lower())},
    {"id": 33, "cat": "Code", "diff": "Hard", "desc": "Race condition",
     "q": "Two threads execute concurrently:\n```\nThread 1: read(x) -> x=0; x = x + 1; write(x)\nThread 2: read(x) -> x=0; x = x + 2; write(x)\n```\nx starts at 0. What are all possible final values of x? Explain each scenario.",
     "fn": lambda a: "1" in a and "2" in a and "3" in a and ("race" in a.lower() or "concurrent" in a.lower() or "interleav" in a.lower())},

    # ═══ CATEGORY: Summarization ══════════════════════════════════
    {"id": 34, "cat": "Summarization", "diff": "Medium", "desc": "Key points extraction",
     "q": 'Extract exactly 3 key takeaways from: "Amazon Web Services announced a $10 billion investment in AI infrastructure across three new data center regions. The investment will create 12,000 jobs. AWS CEO Matt Garman stated this positions AWS as the leading cloud for AI workloads. Competitors Google Cloud and Azure have announced similar but smaller investments of $7B and $8B respectively. Industry analysts predict the cloud AI market will reach $200B by 2027."',
     "fn": lambda a: ("10" in a and "billion" in a.lower()) and ("12,000" in a or "12000" in a or "jobs" in a.lower())},
    {"id": 35, "cat": "Summarization", "diff": "Hard", "desc": "Abstractive summary",
     "q": 'Summarize the business impact in 2 sentences: "Our SaaS platform experienced 4 hours of downtime on Black Friday due to a database failover that did not execute properly. During the outage, approximately 23,000 transactions failed to process, representing an estimated $1.8M in lost revenue. Additionally, 340 enterprise customers opened support tickets, and our NPS score dropped 12 points in the following survey. The engineering team has since implemented automated failover testing and increased database redundancy from 2x to 3x."',
     "fn": lambda a: ("1.8" in a or "1.8M" in a or "million" in a.lower()) and ("downtime" in a.lower() or "outage" in a.lower() or "black friday" in a.lower())},

    # ═══ Additional Hard Questions ════════════════════════════════
    {"id": 36, "cat": "Reasoning", "diff": "Hard", "desc": "Estimation",
     "q": "Estimate: If a model processes 50 tokens/second and you need to generate a 500-word response (approximately 650 tokens), with a 2000-token prompt that processes at 300 tokens/second, what is the approximate total response time?",
     "fn": lambda a: ("19" in a or "20" in a or "13" in a) and ("second" in a.lower() or "sec" in a.lower() or "s" in a.lower())},
    {"id": 37, "cat": "Instruction", "diff": "Hard", "desc": "Multi-format output",
     "q": 'Given this data, output it as: (1) a markdown table, and (2) a CSV line.\n\nData: Product "Widget-X", SKU "WX-100", Price $29.99, Stock 1,450 units.',
     "fn": lambda a: ("|" in a or "---" in a) and ("WX-100" in a or "wx-100" in a.lower()) and ("," in a)},
    {"id": 38, "cat": "Document QA", "diff": "Hard", "desc": "Compliance check",
     "q": 'Policy: "All data transfers to third parties require (1) DPA signed, (2) SOC2 certification, (3) encryption in transit and at rest, (4) data residency in EU or US."\n\nVendor profile: "DataSync Ltd. Has signed DPA. SOC2 Type II certified. Uses TLS 1.3 for transit. Data stored in AWS eu-west-1. At-rest encryption: none currently, planned for Q2 2025."\n\nIs this vendor compliant? Which requirements are met and which are not?',
     "fn": lambda a: ("not compliant" in a.lower() or "non-compliant" in a.lower() or "fails" in a.lower() or "not met" in a.lower()) and ("at-rest" in a.lower() or "at rest" in a.lower() or "encryption" in a.lower())},
    {"id": 39, "cat": "Long Context", "diff": "Hard", "desc": "Table interpretation",
     "q": 'Server performance data:\n| Server | CPU% | RAM% | Disk I/O (MB/s) | Error Rate | Region |\n|--------|------|------|-----------------|------------|--------|\n| srv-01 | 45 | 72 | 120 | 0.01% | US-East |\n| srv-02 | 92 | 88 | 340 | 2.30% | US-East |\n| srv-03 | 38 | 55 | 95 | 0.02% | EU-West |\n| srv-04 | 78 | 91 | 210 | 0.15% | EU-West |\n| srv-05 | 95 | 95 | 380 | 5.10% | AP-South |\n\n1. Which servers need immediate attention and why?\n2. Is there a regional pattern?\n3. Recommend specific actions for each flagged server.',
     "fn": lambda a: ("srv-02" in a.lower() or "srv-05" in a.lower()) and ("cpu" in a.lower() or "error" in a.lower() or "overload" in a.lower() or "scale" in a.lower())},
    {"id": 40, "cat": "Reasoning", "diff": "Hard", "desc": "Trade-off analysis",
     "q": "You must choose between two architectures for a real-time trading system:\n\nOption A: Monolith. Latency: 2ms. Deployment: 1 hour. Team: 5 devs. Scaling: vertical only (max 10K req/s).\nOption B: Microservices. Latency: 8ms. Deployment: 10 min per service. Team: 15 devs. Scaling: horizontal (unlimited).\n\nCurrent load: 5K req/s, expected to reach 50K in 18 months. Which do you recommend and why?",
     "fn": lambda a: ("b" in a.lower() or "microservice" in a.lower()) and ("scal" in a.lower() or "50" in a)},
]


def run_llama_with_config(model_path, prompt, kv_config, max_tokens=400):
    """Run llama-cli with specific KV cache config."""
    cli = get_llama_cli()
    cmd = [
        cli, "-fa", "1", "-ngl", "999",
        "-m", model_path,
        "-ctk", kv_config["ctk"], "-ctv", kv_config["ctv"],
        "-p", prompt,
        "-n", str(max_tokens),
        "--single-turn", "--no-display-prompt",
    ]

    t_start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    wall_time = time.perf_counter() - t_start

    full = result.stdout + "\n" + result.stderr

    prompt_tps = gen_tps = 0.0
    m = re.search(r"Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s", full)
    if m:
        prompt_tps, gen_tps = float(m.group(1)), float(m.group(2))

    # Memory
    memory = {"self_mib": 0, "model_mib": 0, "kv_mib": 0, "compute_mib": 0}
    for line in full.split("\n"):
        if "memory_breakdown" in line and ("MTL" in line or "CUDA" in line):
            mm = re.search(r'(\d+)\s*=\s*(\d+)\s*\+\s*\(\s*(\d+)\s*=\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)\)', line)
            if mm:
                memory["self_mib"] = int(mm.group(3))
                memory["model_mib"] = int(mm.group(4))
                memory["kv_mib"] = int(mm.group(5))
                memory["compute_mib"] = int(mm.group(6))

    # Clean answer
    answer = result.stdout
    answer = re.sub(r'\x1b\[[0-9;]*m', '', answer)
    answer = re.sub(r'\[ Prompt:.*?]', '', answer)
    for noise in ["Loading model...", "Exiting...", "llama_memory", "ggml_metal",
                   "ggml_cuda", "build      :", "model      :", "modalities :",
                   "available commands:", "/exit", "/regen", "/clear", "/read"]:
        answer = "\n".join(l for l in answer.split("\n") if noise not in l)
    answer = re.sub(r'^[|/\-\\]+\s*', '', answer)

    return {
        "answer": answer.strip()[:800],
        "prompt_tps": round(prompt_tps, 1),
        "gen_tps": round(gen_tps, 1),
        "wall_time_s": round(wall_time, 2),
        "memory": memory,
    }


def run_memory_sweep(model_path, kv_config):
    """Run memory probe at all context sizes for a model+config."""
    cli = get_llama_cli()
    results = []
    for ctx, label in zip(CONTEXT_SIZES, CTX_LABELS):
        prompt = "Hello " * min(ctx, 500)
        r = subprocess.run(
            [cli, "-m", model_path, "-ngl", "999", "-fa", "1",
             "-c", str(ctx), "-ctk", kv_config["ctk"], "-ctv", kv_config["ctv"],
             "-p", prompt, "-n", "1", "--single-turn"],
            capture_output=True, text=True, timeout=120,
        )
        output = r.stdout + "\n" + r.stderr
        mem = {"label": label, "ctx": ctx, "self_mib": 0, "kv_mib": 0}
        for line in output.split("\n"):
            if "memory_breakdown" in line and ("MTL" in line or "CUDA" in line):
                mm = re.search(r'(\d+)\s*=\s*(\d+)\s*\+\s*\(\s*(\d+)\s*=\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)\)', line)
                if mm:
                    mem["self_mib"] = int(mm.group(3))
                    mem["kv_mib"] = int(mm.group(5))
        results.append(mem)
    return results


def run_throughput_sweep(model_path, kv_config):
    """Run llama-bench at multiple prompt sizes for a model+config."""
    bench = get_llama_bench()
    results = []
    for pp_size in [512, 2048, 4096]:
        r = subprocess.run(
            [bench, "-m", model_path, "-ngl", "999", "-fa", "1",
             "-ctk", kv_config["ctk"], "-ctv", kv_config["ctv"],
             "-p", str(pp_size), "-n", "32", "-r", "1", "-o", "json"],
            capture_output=True, text=True, timeout=300,
        )
        full = r.stdout + "\n" + r.stderr
        pp_tps = tg_tps = 0.0
        try:
            start = full.find("[")
            end = full.rfind("]")
            if start >= 0 and end > start:
                data = json.loads(full[start:end + 1])
                for entry in data:
                    if entry.get("n_prompt", 0) > 0 and entry.get("n_gen", 0) == 0:
                        pp_tps = entry.get("avg_ts", 0.0)
                    elif entry.get("n_gen", 0) > 0:
                        tg_tps = entry.get("avg_ts", 0.0)
        except (json.JSONDecodeError, TypeError):
            pass
        results.append({"prompt_size": pp_size, "pp_tps": round(pp_tps, 1), "tg_tps": round(tg_tps, 1)})
    return results


def run_full_evaluation(model_names, verbose=True):
    """Run the complete Bonsai + TurboQuant evaluation."""
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    all_results = {
        "benchmark": "Bonsai 1-Bit + TurboQuant KV Cache Evaluation",
        "timestamp": timestamp,
        "models": model_names,
        "configs": [c["name"] for c in KV_CONFIGS],
        "qa_results": {},
        "memory_results": {},
        "throughput_results": {},
    }

    for model_name in model_names:
        model_path = get_model_path(model_name)
        all_results["qa_results"][model_name] = {}
        all_results["memory_results"][model_name] = {}
        all_results["throughput_results"][model_name] = {}

        for cfg in KV_CONFIGS:
            cfg_key = cfg["short"]

            if verbose:
                print(f"\n{'='*70}")
                print(f"  {model_name} | {cfg['name']}")
                print(f"{'='*70}")

            # ── QA Benchmark ───────────────────────────────────
            qa_results = []
            for q in EVAL_QUESTIONS:
                r = run_llama_with_config(model_path, q["q"], cfg)
                accepted = q["fn"](r["answer"])
                entry = {
                    "id": q["id"], "cat": q["cat"], "diff": q["diff"],
                    "desc": q["desc"], "question": q["q"][:200],
                    "accepted": accepted,
                    **r,
                }
                qa_results.append(entry)

                if verbose:
                    tag = "PASS" if accepted else "FAIL"
                    print(f"  Q{q['id']:2d} [{q['cat']:>14}] [{q['diff']:>6}] "
                          f"Wall={r['wall_time_s']:>5.1f}s  Gen={r['gen_tps']:>5.0f}t/s  [{tag}]",
                          flush=True)

            all_results["qa_results"][model_name][cfg_key] = qa_results

            # ── Memory Sweep ───────────────────────────────────
            if verbose:
                print(f"  Memory sweep...", end="", flush=True)
            mem = run_memory_sweep(model_path, cfg)
            all_results["memory_results"][model_name][cfg_key] = mem
            if verbose:
                print(" done")

            # ── Throughput Sweep ───────────────────────────────
            if verbose:
                print(f"  Throughput sweep...", end="", flush=True)
            tput = run_throughput_sweep(model_path, cfg)
            all_results["throughput_results"][model_name][cfg_key] = tput
            if verbose:
                print(" done")

    return all_results


def print_ieee_summary(results):
    """Print IEEE-style summary tables."""
    models = results["models"]
    configs = [c["short"] for c in KV_CONFIGS]
    sep = "=" * 100

    print(f"\n{sep}")
    print("  BONSAI + TURBOQUANT: COMPREHENSIVE EVALUATION RESULTS")
    print(f"  {results['timestamp']}")
    print(sep)

    # ── Table 1: Quality by Model x Config ─────────────────────
    print(f"\n  TABLE 1: Overall Quality (Pass Rate)")
    print(f"  {'-'*70}")
    print(f"  {'Model':<15}", end="")
    for c in configs:
        print(f"  | {c:>12}", end="")
    print()
    print(f"  {'-'*15}", end="")
    for _ in configs:
        print(f"--+-{'-'*11}", end="")
    print()

    for m in models:
        print(f"  {m:<15}", end="")
        for c in configs:
            rs = results["qa_results"][m][c]
            passed = sum(1 for r in rs if r["accepted"])
            total = len(rs)
            print(f"  | {passed}/{total} ({passed/total*100:.0f}%)", end="")
        print()
    print()

    # ── Table 2: Quality by Category ───────────────────────────
    cats = {}
    for q in EVAL_QUESTIONS:
        cats.setdefault(q["cat"], []).append(q["id"])

    print(f"  TABLE 2: Quality by Category (Pass Rate)")
    print(f"  {'-'*90}")
    for m in models:
        print(f"\n  {m}:")
        print(f"  {'Category':<18} {'Count':>5}", end="")
        for c in configs:
            print(f"  | {c:>12}", end="")
        print()

        for cat, ids in cats.items():
            print(f"  {cat:<18} {len(ids):>5}", end="")
            for c in configs:
                rs = results["qa_results"][m][c]
                passed = sum(1 for r in rs if r["id"] in ids and r["accepted"])
                pct = passed / len(ids) * 100
                print(f"  | {passed}/{len(ids)} ({pct:.0f}%)", end="  ")
            print()

    # ── Table 3: Memory Savings ────────────────────────────────
    print(f"\n  TABLE 3: KV Cache Memory (MiB) at Different Context Sizes")
    print(f"  {'-'*90}")
    for m in models:
        print(f"\n  {m}:")
        print(f"  {'Context':>8}", end="")
        for c in configs:
            print(f"  | {c+' KV':>12} {c+' Total':>12}", end="")
        print()
        for i, label in enumerate(CTX_LABELS):
            print(f"  {label:>8}", end="")
            for c in configs:
                mem = results["memory_results"][m][c][i]
                print(f"  | {mem['kv_mib']:>9}MiB {mem['self_mib']:>9}MiB", end="")
            print()

    # ── Table 4: KV Cache Compression Ratios ───────────────────
    print(f"\n  TABLE 4: KV Cache Compression Ratio (vs FP16 Baseline)")
    print(f"  {'-'*70}")
    print(f"  {'Model':<15} {'Context':>8} {'FP16 KV':>10} {'TQ-Q8 KV':>10} {'Q8 Ratio':>10} {'TQ-Q4 KV':>10} {'Q4 Ratio':>10}")
    for m in models:
        for i, label in enumerate(CTX_LABELS):
            fp16_kv = results["memory_results"][m]["FP16"][i]["kv_mib"]
            q8_kv = results["memory_results"][m]["TQ-Q8"][i]["kv_mib"]
            q4_kv = results["memory_results"][m]["TQ-Q4"][i]["kv_mib"]
            q8_ratio = fp16_kv / q8_kv if q8_kv else 0
            q4_ratio = fp16_kv / q4_kv if q4_kv else 0
            print(f"  {m:<15} {label:>8} {fp16_kv:>8}MiB {q8_kv:>8}MiB {q8_ratio:>9.2f}x {q4_kv:>8}MiB {q4_ratio:>9.2f}x")

    print(f"\n{sep}")
