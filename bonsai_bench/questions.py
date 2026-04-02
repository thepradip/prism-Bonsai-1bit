"""20 production-quality QA questions for evaluating 1-bit models."""

QA_QUESTIONS = [
    # ── RAG Context Comprehension (1-5) ────────────────────────
    {
        "id": 1, "category": "RAG Context", "difficulty": "Medium",
        "description": "Calculate YoY growth from passage",
        "question": 'Based on the following passage, answer the question.\n\nPassage: "Meridian Technologies reported Q3 2024 revenue of $4.2 billion, up 18% year-over-year. The growth was primarily driven by their cloud infrastructure division, which grew 34% to $1.8 billion. However, their legacy hardware segment declined 12% to $890 million. CEO Sarah Chen noted that operating margins improved to 22.5%, up from 19.1% in Q3 2023. The company revised its full-year guidance upward to $17.2 billion, from the previous estimate of $16.5 billion."\n\nQuestion: What was the dollar amount of revenue growth in the cloud infrastructure division compared to the same quarter last year? Show your calculation.',
        "accept_fn": lambda a: ("1.34" in a or "1,34" in a or "1.342" in a or "340" in a or "1342" in a or "1.8" in a) and ("34%" in a or "34 %" in a or "billion" in a.lower()),
    },
    {
        "id": 2, "category": "RAG Context", "difficulty": "Hard",
        "description": "Multi-step fact cross-reference",
        "question": 'Read the following context and answer precisely.\n\nContext: "Project Aurora has three phases. Phase 1 (completed Jan 2024) migrated 2,400 user accounts from SystemA to SystemB. Phase 2 (completed Apr 2024) migrated 1,800 accounts from SystemC to SystemB. Phase 3 (planned for Jul 2024) will migrate the remaining accounts from SystemD. The total number of accounts across all four legacy systems was 6,100 before migration began. SystemB had 500 existing accounts before any migration started."\n\nQuestions:\n1. How many accounts will Phase 3 migrate?\n2. How many total accounts will SystemB have after all phases complete?',
        "accept_fn": lambda a: ("1900" in a or "1,900" in a) and ("6600" in a or "6,600" in a or "6100" in a.replace(",","") or "6,100" in a),
    },
    {
        "id": 3, "category": "RAG Context", "difficulty": "Hard",
        "description": "Detect contradiction across docs",
        "question": 'Read these two passages and identify the contradiction.\n\nPassage A (Internal Memo, March 5): "The new authentication system will use OAuth 2.0 with PKCE flow. All API endpoints must migrate by April 30. The security team confirmed that legacy API keys will be deprecated and disabled on May 1."\n\nPassage B (Client Communication, March 12): "We want to assure all partners that your existing API key integrations will continue to work throughout 2024. No action is required on your part at this time."\n\nWhat is the specific contradiction between these two documents? Which stakeholders could be impacted?',
        "accept_fn": lambda a: ("api key" in a.lower() or "api keys" in a.lower()) and ("contradict" in a.lower() or "conflict" in a.lower() or "inconsisten" in a.lower() or "deprecated" in a.lower() or "disabled" in a.lower() or "may 1" in a.lower()),
    },
    {
        "id": 4, "category": "RAG Context", "difficulty": "Medium",
        "description": "Refuse when answer not in context",
        "question": 'Based ONLY on the provided context, answer the question. If the answer is not in the context, say "NOT FOUND IN CONTEXT".\n\nContext: "Acme Corp\'s data center in Oregon uses 45MW of power and has a PUE ratio of 1.3. The facility was built in 2019 and houses approximately 50,000 servers. It primarily serves customers in the US West region."\n\nQuestion: What is Acme Corp\'s annual electricity cost for the Oregon data center?',
        "accept_fn": lambda a: "not found" in a.lower() or "not in the context" in a.lower() or "not mentioned" in a.lower() or "doesn't mention" in a.lower() or "does not mention" in a.lower() or "cannot be determined" in a.lower() or "not provided" in a.lower() or "no information" in a.lower(),
    },
    {
        "id": 5, "category": "RAG Context", "difficulty": "Hard",
        "description": "Constrained format summarization",
        "question": 'Summarize the following in EXACTLY 3 bullet points. Each bullet must be one sentence only. Do not use more than 3 bullets.\n\nText: "The European Union\'s AI Act, which came into force in August 2024, establishes a risk-based regulatory framework for artificial intelligence systems. High-risk AI applications in areas like healthcare, law enforcement, and critical infrastructure face mandatory conformity assessments, ongoing monitoring requirements, and transparency obligations. General-purpose AI models, including large language models, must comply with transparency requirements and copyright rules. Systems deemed to pose unacceptable risk, such as social scoring by governments and real-time biometric surveillance in public spaces, are outright banned. Companies face fines of up to 7% of global revenue or 35 million euros for violations. The Act provides a phased implementation timeline, with bans on prohibited practices taking effect first, followed by requirements for high-risk systems over the next 24 months."',
        "accept_fn": lambda a: a.count("- ") >= 2 or a.count("* ") >= 2 or (a.count("\n") >= 2 and a.count("\n") <= 6),
    },

    # ── Finance Document QA (6-10) ─────────────────────────────
    {
        "id": 6, "category": "Finance QA", "difficulty": "Medium",
        "description": "Calculate financial ratios",
        "question": 'From the following financial data, calculate the debt-to-equity ratio and current ratio. Show your work.\n\nBalance Sheet (in millions):\n- Total Assets: $12,400\n- Current Assets: $3,200\n- Total Liabilities: $7,800\n- Current Liabilities: $2,100\n- Shareholders\' Equity: $4,600\n- Long-term Debt: $5,700',
        "accept_fn": lambda a: ("1.7" in a or "1.69" in a or "1.52" in a) and ("1.52" in a or "1.5" in a),
    },
    {
        "id": 7, "category": "Finance QA", "difficulty": "Hard",
        "description": "Interpret earnings call sentiment",
        "question": 'Read this earnings call excerpt and answer: Is the company\'s outlook positive, negative, or mixed? Cite specific phrases that support your assessment.\n\n"While we delivered solid top-line growth of 12%, we want to be transparent about headwinds we\'re seeing. Gross margins compressed 180 basis points due to elevated input costs, and we expect this pressure to persist through Q1 2025. On the positive side, our subscription revenue reached an inflection point, now representing 62% of total revenue versus 41% a year ago. We\'re cautiously optimistic about the second half but have prudently adjusted our full-year EPS guidance down by $0.15 to reflect near-term margin pressures."',
        "accept_fn": lambda a: "mixed" in a.lower() and ("margin" in a.lower() or "headwind" in a.lower() or "compress" in a.lower()) and ("subscription" in a.lower() or "positive" in a.lower() or "growth" in a.lower()),
    },
    {
        "id": 8, "category": "Finance QA", "difficulty": "Medium",
        "description": "Structured data extraction",
        "question": 'Extract the following fields from this invoice description and return them as a structured list. Use the exact field names given.\n\n"Invoice #INV-2024-0847 dated November 15, 2024 from CloudServe Inc. (vendor ID: VS-1192) to Pinnacle Dynamics LLC for Professional Services - Cloud Migration Phase 2. Subtotal: $34,500.00. Tax (8.5%): $2,932.50. Total Due: $37,432.50. Payment terms: Net 30. PO Reference: PO-2024-445."\n\nFields to extract: invoice_number, date, vendor_name, vendor_id, client_name, subtotal, tax_rate, total, payment_terms, po_reference',
        "accept_fn": lambda a: "INV-2024-0847" in a and "VS-1192" in a and "37,432.50" in a and "PO-2024-445" in a and ("Net 30" in a or "net 30" in a),
    },
    {
        "id": 9, "category": "Finance QA", "difficulty": "Hard",
        "description": "Multi-period trend analysis",
        "question": 'Analyze the revenue trend and answer the questions.\n\nQuarterly Revenue (in $M):\nQ1 2023: 120, Q2 2023: 135, Q3 2023: 128, Q4 2023: 152\nQ1 2024: 141, Q2 2024: 158, Q3 2024: 149, Q4 2024: 175\n\n1. What is the year-over-year growth rate for Q4?\n2. Which quarter shows the weakest sequential growth in 2024?\n3. Is there a seasonal pattern? If so, describe it.',
        "accept_fn": lambda a: ("15" in a and "%" in a) and ("q3" in a.lower() or "q1" in a.lower()) and ("season" in a.lower() or "pattern" in a.lower() or "dip" in a.lower() or "decline" in a.lower()),
    },
    {
        "id": 10, "category": "Finance QA", "difficulty": "Medium",
        "description": "Rank risks from disclosure",
        "question": 'Based on this risk disclosure, list the top 3 risks in order of severity and explain your ranking.\n\n"The Company faces several material risks: (1) Concentration risk - our top 3 clients represent 68% of revenue; (2) Regulatory risk - pending legislation in the EU could require significant product modifications estimated at $12M; (3) Currency risk - 40% of revenue is denominated in foreign currencies with unhedged exposure; (4) Key person risk - our CTO holds 14 patents critical to our core product; (5) Cybersecurity risk - we experienced two minor data incidents in 2024 that were contained without material impact."',
        "accept_fn": lambda a: "concentration" in a.lower() and ("68%" in a or "top 3 client" in a.lower() or "three client" in a.lower()),
    },

    # ── Reasoning & Logic (11-15) ──────────────────────────────
    {
        "id": 11, "category": "Reasoning", "difficulty": "Hard",
        "description": "Constraint satisfaction puzzle",
        "question": 'Solve this logic puzzle step by step.\n\nFive people (Alice, Bob, Carol, Dave, Eve) sit in a row of 5 chairs numbered 1-5 (left to right).\n- Bob sits immediately to the right of Alice.\n- Carol is not in chair 1 or chair 5.\n- Dave sits in chair 4.\n- Eve is not adjacent to Dave.\n\nIn which chair does each person sit?',
        "accept_fn": lambda a: ("alice" in a.lower() and "bob" in a.lower()) and (("1" in a and "2" in a) or ("chair 1" in a.lower()) or ("alice" in a.lower() and "1" in a)),
    },
    {
        "id": 12, "category": "Reasoning", "difficulty": "Hard",
        "description": "Root cause analysis from timeline",
        "question": 'A server went down at 3:00 AM. Here\'s the timeline of events:\n- 2:45 AM: Automated deployment pushed version 2.3.1\n- 2:50 AM: Memory usage spiked from 60% to 95%\n- 2:55 AM: Database connection pool exhausted (max 100 connections)\n- 2:58 AM: Error rate jumped from 0.1% to 45%\n- 3:00 AM: Health check failed, server marked unhealthy and removed from load balancer\n- 3:02 AM: Rollback to version 2.3.0 initiated\n- 3:08 AM: Memory usage dropped to 65%, connections normalized\n- 3:10 AM: Health check passed, server re-added to load balancer\n\nWhat is the most likely root cause? What evidence supports this? What would you recommend to prevent recurrence?',
        "accept_fn": lambda a: ("2.3.1" in a or "deploy" in a.lower()) and ("memory" in a.lower() or "leak" in a.lower()) and ("rollback" in a.lower() or "prevent" in a.lower() or "test" in a.lower() or "monitor" in a.lower()),
    },
    {
        "id": 13, "category": "Reasoning", "difficulty": "Medium",
        "description": "Formal logic evaluation",
        "question": 'Evaluate each conclusion as VALID or INVALID based strictly on the given premises.\n\nPremises:\n1. All machine learning engineers know Python.\n2. Some Python developers are not machine learning engineers.\n3. No data analyst at this company knows R.\n4. All data analysts at this company know Python.\n\nConclusions:\nA. "Some Python developers are data analysts." - VALID or INVALID?\nB. "No machine learning engineer knows R." - VALID or INVALID?\nC. "All people who know Python are machine learning engineers." - VALID or INVALID?',
        "accept_fn": lambda a: ("invalid" in a.lower()) and (a.lower().count("invalid") >= 2),
    },
    {
        "id": 14, "category": "Reasoning", "difficulty": "Hard",
        "description": "Constraint optimization problem",
        "question": 'A company has a $500,000 annual budget for cloud computing split across 3 providers. The constraints are:\n- AWS must receive at least 40% of the budget\n- GCP cannot receive more than $150,000\n- Azure must receive at least $50,000 more than GCP\n- The total must equal exactly $500,000\n\nFind a valid allocation. Then determine: what is the maximum possible amount for GCP?',
        "accept_fn": lambda a: ("200,000" in a or "200000" in a or "$200" in a) and ("150,000" in a or "150000" in a or "$150" in a or "100,000" in a or "100000" in a),
    },
    {
        "id": 15, "category": "Reasoning", "difficulty": "Medium",
        "description": "Technical analogy completion",
        "question": 'Complete the analogy and explain your reasoning in one sentence each.\n\n1. Docker container is to application as _____ is to virtual machine.\n2. Git branch is to codebase as _____ is to database.\n3. Load balancer is to servers as _____ is to network traffic.\n4. API rate limiting is to server resources as _____ is to highway traffic.',
        "accept_fn": lambda a: len(a) > 100 and ("hypervisor" in a.lower() or "vmware" in a.lower() or "host" in a.lower() or "image" in a.lower() or "snapshot" in a.lower() or "os" in a.lower()),
    },

    # ── Instruction Following (16-20) ──────────────────────────
    {
        "id": 16, "category": "Instruction", "difficulty": "Medium",
        "description": "Text to structured JSON",
        "question": 'Convert this paragraph into a JSON object with exactly these keys: "title", "date", "amount", "currency", "status". Respond with ONLY the JSON, no explanation.\n\n"The payment of 2,500 euros for the Q3 Marketing Campaign was approved on September 15, 2024 and is currently pending disbursement."',
        "accept_fn": lambda a: '"title"' in a and '"date"' in a and '"amount"' in a and '"currency"' in a and '"status"' in a and ("{" in a and "}" in a),
    },
    {
        "id": 17, "category": "Instruction", "difficulty": "Hard",
        "description": "Apply conditional routing rules",
        "question": 'Process these customer support tickets according to the rules:\n\nRules:\n- If severity is "critical" AND status is "open": assign to "Tier 3" and set priority to "P0"\n- If severity is "high" AND created more than 24h ago: escalate to "Tier 2" and set priority to "P1"\n- All other tickets: assign to "Tier 1" and set priority to "P2"\n\nTickets:\n1. ID: T-001, Severity: critical, Status: open, Created: 2h ago\n2. ID: T-002, Severity: high, Status: open, Created: 36h ago\n3. ID: T-003, Severity: medium, Status: open, Created: 1h ago\n4. ID: T-004, Severity: high, Status: open, Created: 12h ago\n5. ID: T-005, Severity: critical, Status: resolved, Created: 48h ago\n\nList each ticket\'s assignment and priority.',
        "accept_fn": lambda a: "T-001" in a and "T-002" in a and (("tier 3" in a.lower() or "Tier 3" in a) and ("tier 2" in a.lower() or "Tier 2" in a) and ("tier 1" in a.lower() or "Tier 1" in a) and ("P0" in a or "p0" in a) and ("P1" in a or "p1" in a)),
    },
    {
        "id": 18, "category": "Instruction", "difficulty": "Medium",
        "description": "Follow negative constraints",
        "question": 'Answer the following question about Python. You must follow ALL of these rules:\n- Do NOT use code blocks or backticks\n- Do NOT mention the word "function"\n- Do NOT use bullet points or numbered lists\n- Answer in exactly 2 sentences\n\nQuestion: How does Python\'s decorator syntax work?',
        "accept_fn": lambda a: "```" not in a and "function" not in a.lower() and a.count(". ") <= 3 and len(a.split("\n")) <= 4 and "- " not in a and "1." not in a,
    },
    {
        "id": 19, "category": "Instruction", "difficulty": "Hard",
        "description": "Bug identification in code",
        "question": 'Review this code and identify ALL bugs. For each bug, state: (a) the line, (b) what\'s wrong, (c) the fix. Do not rewrite the entire code.\n\n```python\ndef calculate_average(numbers):\n    total = 0\n    for i in range(len(numbers)):\n        total += numbers[i]\n    average = total / len(numbers)\n    return average\n\ndef find_duplicates(lst):\n    seen = {}\n    duplicates = []\n    for item in lst:\n        if item in seen:\n            duplicates.append(item)\n        seen[item] = True\n    return duplicates\n\nresult = calculate_average([])\nprint(f"Average: {result}")\n```',
        "accept_fn": lambda a: ("division" in a.lower() or "zero" in a.lower() or "empty" in a.lower() or "ZeroDivision" in a) and ("len" in a.lower() or "[]" in a or "empty" in a.lower()),
    },
    {
        "id": 20, "category": "Instruction", "difficulty": "Hard",
        "description": "NL-to-SQL with edge cases",
        "question": 'Write a SQL query for this request. Use standard SQL syntax.\n\nDatabase schema:\n- orders(id, customer_id, amount, status, created_at)\n- customers(id, name, email, tier)\n\nRequest: "Find all premium tier customers who placed more than 5 orders in the last 90 days but whose total order amount is less than $1000. Show customer name, order count, and total amount. Sort by total amount descending."\n\nImportant: Handle the case where orders might have status = \'cancelled\' (exclude these).',
        "accept_fn": lambda a: "select" in a.lower() and "join" in a.lower() and "group by" in a.lower() and ("having" in a.lower()) and ("premium" in a.lower()) and ("cancel" in a.lower() or "status" in a.lower()),
    },
]
