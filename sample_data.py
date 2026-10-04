"""
Sample benchmark documents for Vectorless RAG demonstration.
Contains exact terms, technical acronyms, figures, and regulatory codes 
where vectorless BM25/Lexical retrieval outshines semantic embedding drift.
"""

SAMPLE_DOCS = {
    "Quantum_Computing_Specs_2025.txt": """
# Superconducting Quantum Processors: Architecture and Benchmarks (2025)

## Section 1: Executive Overview & NISQ Milestones
In the Noisy Intermediate-Scale Quantum (NISQ) era, processor scaling requires balancing physical qubit count with two-qubit gate fidelities.
The Helios-X9 processor features 1,121 physical superconducting transmon qubits arranged in a heavy-hexagonal lattice architecture.
Its primary objective is the demonstration of fault-tolerant logical qubits via Surface Code Distance-7 (d=7).

## Section 2: Gate Fidelities and Coherence Times
Benchmark measurements conducted at dilution refrigerator base temperatures of 12.5 mK revealed:
- Average single-qubit gate fidelity (F_1Q): 99.96% using randomized benchmarking (RB).
- Average two-qubit cross-resonance gate fidelity (F_2Q): 99.82% across all active couplers.
- Median energy relaxation time (T1): 185 microseconds.
- Median dephasing time (T2*): 142 microseconds.
- Readout error rate per qubit: 0.85% under multiplexed dispersive readout with quantum-limited traveling wave parametric amplifiers (TWPAs).

## Section 3: Algorithmic Complexity & Cryptographic Implications
Quantum algorithms evaluated on the Helios-X9 simulator include:
1. Shor's Algorithm: Requires O((log N)^3) modular exponentiation steps. For factoring standard 2048-bit RSA keys, an estimated 4,096 logical qubits and approximately 20 million physical qubits are required under standard surface-code syndrome extraction cycles.
2. Grover's Algorithm: Provides a quadratic speedup of O(sqrt(N)) for unstructured database search and AES-256 key search. Grover's search on an N-item search space achieves optimal queries using selective phase inversions.
3. Quantum Approximate Optimization Algorithm (QAOA): Evaluated at depth p=6 for Max-Cut problems on 3-regular graphs, demonstrating an approximation ratio of 0.892 without active error mitigation.

## Section 4: Dilution Cryogenics & Pulse Control
The cryogenic stack utilizes a closed-cycle pulse-tube dilution refrigerator delivering 450 microwatts of cooling power at 100 mK.
Control pulses are synthesized using 14-bit arbitrary waveform generators (AWGs) operating at 5.0 GSa/s with microwave up-conversion.
Crosstalk mitigation is achieved through tunable capacitive couplers (Transmon-Coupler-Transmon architecture), suppressing residual ZZ coupling to below 12 kHz.
""",

    "EU_AI_Act_Compliance_Guide.md": """
# European Union AI Act: Enterprise Compliance & Governance Manual

## Article 1: Scope & Risk Classification Framework
The EU Artificial Intelligence Act establishes a risk-based categorization system for all AI software placed on the EU internal market:
1. Unacceptable Risk: AI practices strictly prohibited under Article 5 (e.g., social scoring by public authorities, cognitive behavioral manipulation, untargeted biometric facial scraping).
2. High-Risk AI Systems: Systems identified under Annex II and Annex III (e.g., critical infrastructure safety components, CV screening software, credit scoring algorithms, law enforcement biometric identification).
3. Specific Transparency Risk: AI systems that interact directly with natural persons (Article 52), including chatbots, synthetic deepfake media generators, and emotion recognition systems.
4. Minimal or Low Risk: Unregulated AI applications such as spam filters or AI-enabled video games.

## Article 5: Prohibited AI Practices and Deadlines
Compliance deadlines for Prohibited AI practices took mandatory effect on 2 February 2025.
Any enterprise operating prohibited systems within the European single market faces immediate enforcement notices and cease-and-desist orders from national supervisory authorities.

## Article 10 & 14: High-Risk AI Requirements
High-Risk AI providers must maintain rigorous documentation and automated controls:
- Data Governance (Article 10): Training, validation, and testing datasets must be subject to bias audits, statistical relevance checks, and data lineage tracking.
- Human Oversight (Article 14): Deployers must ensure continuous human-in-the-loop (HITL) or human-on-the-loop (HOTL) kill-switches capable of overriding algorithmic outputs in real-time.
- Fundamental Rights Impact Assessment (FRIA): Mandatory for public entities and critical infrastructure operators prior to production rollout.

## Article 71: Penalties and Fines Schedule
Violations of the EU AI Act incur substantial administrative fines:
- Infringements of Prohibited AI Practices (Article 5): Fines up to €35,000,000 or up to 7% of total worldwide annual turnover for the preceding financial year, whichever is higher.
- Non-compliance with High-Risk AI requirements (Articles 9 through 15): Fines up to €15,000,000 or up to 3% of total worldwide annual turnover.
- Supplying incorrect, incomplete, or misleading information to notified bodies: Fines up to €7,500,000 or up to 1.5% of annual turnover.
- Small and Medium Enterprises (SMEs) benefit from proportionate penalty caps under statutory discretion.
""",

    "NovaCorp_Q3_Financial_Report.txt": """
# NovaCorp Global Technologies - Q3 2025 Financial and Operations Report

## Consolidated Financial Highlights
For the third quarter ended September 30, 2025:
- Total Consolidated Revenue: $482.5 million, representing a 23.4% year-over-year (YoY) increase compared to $391.0 million in Q3 2024.
- Cloud & AI Platform Revenue: $214.8 million (44.5% of total revenue), driven by rapid enterprise adoption of our Vectorless Data Engine.
- Gross Margin: 64.2% on a GAAP basis, expanding 180 basis points YoY due to reduced cloud infrastructure compute overhead.
- Adjusted EBITDA: $128.4 million (26.6% EBITDA margin), compared to $96.1 million in Q3 2024.
- GAAP Operating Income: $89.7 million; Non-GAAP Diluted Earnings Per Share (EPS): $1.42.
- Cash and Cash Equivalents: $1.15 billion as of September 30, 2025. Free Cash Flow generated in Q3 stood at $104.2 million.

## Operational KPIs and Customer Metrics
- Annual Recurring Revenue (ARR): Reached $1.86 billion, growing 21.0% YoY.
- Enterprise Customers (> $100k ARR): Rose to 2,410 clients, an increase of 385 net new enterprise logos.
- Net Revenue Retention (NRR): Stood at 119.5%, reflecting strong product upsells and usage-based expansions.
- Customer Churn Rate: Maintained at a record low of 1.4% annualized.
- Global Headcount: 4,820 full-time employees, with 52% allocated to R&D and Engineering.

## Strategic Outlook & Q4 Full-Year Guidance
- Q4 2025 Revenue Expectation: Projected between $505.0 million and $515.0 million.
- Full Year 2025 Revenue Guidance: Raised to a range of $1.920 billion to $1.935 billion.
- Full Year 2025 Non-GAAP Operating Margin Target: Maintained between 27.0% and 28.5%.
- Capital Expenditures (CapEx): Expected to be approximately $65.0 million in Q4, primarily directed towards low-latency sovereign edge data centers.
"""
}
