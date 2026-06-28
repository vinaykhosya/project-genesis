"""
evolution_journal.py — Phase 9D: Evolution Replay & Scientific Journal

Analyzes population genetics, drives, relationships, and motivations every 100 years (36,000 ticks).
Appends entries to a Markdown research journal and compiles data structures for evolution replay scrubbing.

Journal style: evidence-first, no anthropomorphic language.
Each observation reports:
  - A measured numerical delta
  - Correlated changes (direction + magnitude)
  - A single hedged hypothesis
  - Confidence level with explicit basis
"""

import os
import json
import numpy as np
from typing import Dict, Any, List, Optional
from world.state import WorldState
from world.agents.genetics import GENE_NAMES


def _pct(val: float) -> str:
    """Format a fractional change as a signed percent string, e.g. -27%."""
    return f"{val * 100:+.0f}%"


def _conf_label(score: float) -> str:
    if score >= 0.80:
        return "High"
    if score >= 0.55:
        return "Medium"
    return "Low"

class EvolutionJournal:
    def __init__(self, exp_folder: str):
        self.exp_folder = exp_folder
        self.journal_path = os.path.join(exp_folder, "evolution_journal.md")
        self.history_records: List[Dict[str, Any]] = []
        self.baseline_genes: Optional[np.ndarray] = None
        # Snapshot of previous epoch for epoch-to-epoch delta tracking
        self._prev_record: Optional[Dict[str, Any]] = None
        self.init_journal_file()

    def init_journal_file(self):
        if not os.path.exists(self.journal_path):
            with open(self.journal_path, "w", encoding="utf-8") as f:
                f.write("# Scientific Evolution Journal\n")
                f.write("## Project Genesis — Evidence-Based Evolutionary Telemetry\n\n")
                f.write(
                    "Each entry documents the population state at the end of an epoch.\n"
                    "All claims are grounded in measured simulation data.\n"
                    "Hypotheses are labelled by confidence level and must not be read as causal explanations.\n\n"
                )

    def record_epoch(self, world: WorldState):
        """
        Runs a comprehensive analysis of the population state and appends
        a detailed entry to the Evolution Journal.
        """
        tick = world.tick
        year = tick // 360
        alive_agents = [a for a in world.agents if not a.dead]
        total_agents  = len(world.agents)

        if not alive_agents:
            return

        # ── 1. Genetics ─────────────────────────────────────────────────────────
        living_genes = np.array([a.genome.genes for a in alive_agents])
        gene_means   = living_genes.mean(axis=0)
        gene_vars    = living_genes.var(axis=0)

        if self.baseline_genes is None:
            self.baseline_genes = gene_means.copy()

        # ── 2. Drive tensions ────────────────────────────────────────────────────
        drives_list = []
        for a in alive_agents:
            ds = a.drives
            drives_list.append([
                ds.hunger_tension, ds.thirst_tension, ds.exhaustion_tension,
                ds.pain_tension,   ds.thermal_stress, ds.fear,
                ds.frustration,    ds.contentment,    ds.valence, ds.arousal
            ])
        drives_arr = np.array(drives_list)
        avg_drives = drives_arr.mean(axis=0)

        # ── 3. Motivations ───────────────────────────────────────────────────────
        MOTIV_NAMES = ["Safety", "Family", "Exploration", "Knowledge", "Comfort", "Dominance"]
        motiv_list  = []
        for a in alive_agents:
            m = a.motivation
            motiv_list.append([
                m.safety.current, m.family.current,     m.exploration.current,
                m.knowledge.current, m.comfort.current, m.dominance.current
            ])
        motiv_arr  = np.array(motiv_list)
        avg_motivs = motiv_arr.mean(axis=0)

        # ── 4. Relationships ─────────────────────────────────────────────────────
        rel_trusts = []
        rel_attachments = []
        for a in alive_agents:
            for r in a.relationships.values():
                rel_trusts.append(r.trust)
                rel_attachments.append(r.attachment)

        avg_trust      = float(np.mean(rel_trusts))      if rel_trusts      else 0.0
        avg_attachment = float(np.mean(rel_attachments)) if rel_attachments else 0.0
        rel_count      = len(rel_trusts)

        # ── 5. Cognition ─────────────────────────────────────────────────────────
        pred_acc_vals  = []
        mem_counts     = []
        concept_counts = []
        for a in alive_agents:
            if a.prediction_attempts > 0:
                pred_acc_vals.append(a.prediction_successes / a.prediction_attempts)
            mem_counts.append(len(getattr(a, "episodic_memory", [])))
            concept_counts.append(len(getattr(a, "concepts", {})))

        avg_pred_acc  = float(np.mean(pred_acc_vals))   if pred_acc_vals   else 0.0
        avg_mem       = float(np.mean(mem_counts))       if mem_counts      else 0.0
        avg_concepts  = float(np.mean(concept_counts))   if concept_counts  else 0.0

        # ── 6. Lineage tracking ──────────────────────────────────────────────────
        lineage_counts = {}
        for a in alive_agents:
            founder_id = self.trace_founder(a, world)
            lineage_counts[founder_id] = lineage_counts.get(founder_id, 0) + 1

        sorted_lineages = sorted(lineage_counts.items(), key=lambda x: x[1], reverse=True)
        top_lineages    = [{"founder_id": k, "living_count": v} for k, v in sorted_lineages[:5]]

        founders         = {a.id for a in world.agents if a.generation == 0}
        extinct_founders = sorted(founders - set(lineage_counts.keys()))

        # ── 7. Climate context ───────────────────────────────────────────────────
        climate_state = getattr(world, "climate_epoch_state", {})
        epoch_name    = climate_state.get("current_epoch_name", "Temperate")
        temp_offset   = getattr(world, "global_temp_offset",       0.0)
        water_mult    = getattr(world, "global_water_multiplier",   1.0)

        # ── 8. Colony breakdown ──────────────────────────────────────────────────
        colony_data: Dict[int, Any] = {}
        for c in getattr(world, "colonies", []):
            col_agents = [a for a in alive_agents if getattr(a, "colony_id", -1) == c["id"]]
            if col_agents:
                colony_data[c["id"]] = {
                    "name":       c["name"],
                    "n":          len(col_agents),
                    "gene_means": np.mean([a.genome.genes for a in col_agents], axis=0),
                    "avg_gen":    float(np.mean([a.generation for a in col_agents]))
                }

        avg_gen = float(np.mean([a.generation for a in alive_agents]))
        n_alive = len(alive_agents)

        # ── 9. Build evidence-based insights ────────────────────────────────────
        insights = self._build_insights(
            gene_means, gene_vars, avg_drives, avg_motivs, MOTIV_NAMES,
            avg_pred_acc, avg_trust, avg_attachment, rel_count,
            avg_mem, avg_concepts, colony_data, epoch_name, temp_offset,
            water_mult, avg_gen, n_alive, world
        )

        # ── 10. Write Markdown entry ─────────────────────────────────────────────
        with open(self.journal_path, "a", encoding="utf-8") as f:
            self._write_entry(
                f, tick, year, n_alive, total_agents, epoch_name,
                temp_offset, water_mult, world.generation_number,
                top_lineages, extinct_founders, gene_means, gene_vars,
                avg_drives, avg_motivs, MOTIV_NAMES, avg_trust,
                avg_attachment, avg_pred_acc, insights
            )

        # ── 11. Store record for replay scrubbing ────────────────────────────────
        record = {
            "tick":            tick,
            "year":            year,
            "alive":           n_alive,
            "gene_means":      [float(v) for v in gene_means],
            "gene_vars":       [float(v) for v in gene_vars],
            "avg_drives":      [float(v) for v in avg_drives],
            "avg_motivations": [float(v) for v in avg_motivs],
            "avg_trust":       avg_trust,
            "avg_attachment":  avg_attachment,
            "avg_pred_acc":    avg_pred_acc,
            "avg_memories":    avg_mem,
            "avg_concepts":    avg_concepts,
            "top_lineages":    top_lineages,
            "extinct_count":   len(extinct_founders),
            "insights":        insights,
        }
        self._prev_record = record
        self.history_records.append(record)

        try:
            with open(os.path.join(self.exp_folder, "evolution_replay.json"), "w") as f:
                json.dump(self.history_records, f, indent=2)
        except Exception:
            pass

    # ────────────────────────────────────────────────────────────────────────────
    # Insight builder — evidence first, no anthropomorphic language
    # ────────────────────────────────────────────────────────────────────────────
    def _build_insights(
        self, gene_means, gene_vars, avg_drives, avg_motivs, motiv_names,
        avg_pred_acc, avg_trust, avg_attachment, rel_count,
        avg_mem, avg_concepts, colony_data, epoch_name, temp_offset,
        water_mult, avg_gen, n_alive, world
    ) -> List[Dict[str, Any]]:
        insights = []
        n_cols   = len(colony_data)
        prev     = self._prev_record

        prev_drives = np.array(prev["avg_drives"])      if prev else avg_drives.copy()
        prev_motivs = np.array(prev["avg_motivations"]) if prev else avg_motivs.copy()
        prev_trust  = prev["avg_trust"]     if prev else avg_trust
        prev_acc    = prev["avg_pred_acc"]  if prev else avg_pred_acc
        prev_mem    = prev["avg_memories"]  if prev else avg_mem

        GENE_THRESH  = 0.08
        ACC_THRESH   = 0.05   # 5 percentage-point change

        # ── Insight A: Novelty Seeking (gene 6) ─────────────────────────────────
        G_NOV     = 6
        shift_nov = gene_means[G_NOV] - self.baseline_genes[G_NOV]
        if abs(shift_nov) > GENE_THRESH and n_cols > 0:
            direction = "decreased" if shift_nov < 0 else "increased"
            congruent = sum(
                1 for cd in colony_data.values()
                if (cd["gene_means"][G_NOV] < self.baseline_genes[G_NOV]) == (shift_nov < 0)
            )
            conf_score = 0.40 + 0.40 * (congruent / n_cols) + 0.20 * min(1.0, n_alive / 40.0)

            motiv_expl_idx = 2   # exploration motivation slot
            motiv_delta    = avg_motivs[motiv_expl_idx] - prev_motivs[motiv_expl_idx]
            base_nov       = max(self.baseline_genes[G_NOV], 1e-6)

            assoc = [
                f"Novelty Seeking gene: baseline={self.baseline_genes[G_NOV]:.3f} → current={gene_means[G_NOV]:.3f} ({_pct(shift_nov / base_nov)} from population baseline)",
                f"Exploration motivation: {_pct(motiv_delta / max(abs(prev_motivs[motiv_expl_idx]), 1e-6))} change this epoch",
                f"Colony congruency: {congruent}/{n_cols} active colonies show same direction",
                f"Gene variance: {gene_vars[G_NOV]:.4f} (spread within population)",
            ]
            if water_mult < 0.90:
                assoc.append(f"Water availability: {water_mult:.2f}x (below 1.0x baseline)")

            hyp = (
                f"Reduced exploration may have increased survival during resource scarcity in the '{epoch_name}' epoch."
                if shift_nov < 0 else
                f"Increased resource heterogeneity during '{epoch_name}' may have rewarded wide-range foragers."
            )
            insights.append({
                "category":    "Behavioral Genetics",
                "observation": f"Average Novelty Seeking (Gene 6) {direction} {_pct(shift_nov / base_nov)} from population baseline.",
                "associated":  assoc,
                "hypothesis":  hyp,
                "confidence":  conf_score,
            })

        # ── Insight B: Aggression (gene 8) ──────────────────────────────────────
        G_AGG     = 8
        shift_agg = gene_means[G_AGG] - self.baseline_genes[G_AGG]
        if abs(shift_agg) > GENE_THRESH and n_cols > 0:
            direction = "increased" if shift_agg > 0 else "decreased"
            congruent = sum(
                1 for cd in colony_data.values()
                if (cd["gene_means"][G_AGG] > self.baseline_genes[G_AGG]) == (shift_agg > 0)
            )
            conf_score = 0.40 + 0.40 * (congruent / n_cols) + 0.20 * min(1.0, n_alive / 40.0)

            dom_motiv_idx = 5   # dominance motivation slot
            dom_delta     = avg_motivs[dom_motiv_idx] - prev_motivs[dom_motiv_idx]
            base_agg      = max(self.baseline_genes[G_AGG], 1e-6)

            assoc = [
                f"Aggression gene: baseline={self.baseline_genes[G_AGG]:.3f} → current={gene_means[G_AGG]:.3f} ({_pct(shift_agg / base_agg)})",
                f"Dominance motivation: {_pct(dom_delta / max(abs(prev_motivs[dom_motiv_idx]), 1e-6))} change this epoch",
                f"Average trust: {avg_trust:.3f} (prev epoch: {prev_trust:.3f})",
                f"Colony congruency: {congruent}/{n_cols} active colonies agree",
            ]
            hyp = (
                "Increased competitive pressure over constrained resources may be associated with selection for aggressive phenotypes."
                if shift_agg > 0 else
                "Stable resource access and dense social networks may be associated with reduced inter-agent conflict."
            )
            insights.append({
                "category":    "Social Genetics",
                "observation": f"Aggression gene (Gene 8) {direction} {_pct(shift_agg / base_agg)} from population baseline.",
                "associated":  assoc,
                "hypothesis":  hyp,
                "confidence":  conf_score,
            })

        # ── Insight C: Prediction accuracy shift ─────────────────────────────────
        acc_delta = avg_pred_acc - prev_acc
        if abs(acc_delta) > ACC_THRESH:
            direction = "improved" if acc_delta > 0 else "declined"
            mem_delta = avg_mem - prev_mem
            conf_score = 0.60 + 0.20 * min(1.0, n_alive / 30.0)

            assoc = [
                f"Prediction accuracy: {prev_acc * 100:.1f}% → {avg_pred_acc * 100:.1f}% ({acc_delta * 100:+.1f} pp this epoch)",
                f"Average episodic memory size: {_pct(mem_delta / max(prev_mem, 1e-6))} change",
                f"Average concept count: {avg_concepts:.1f}",
                f"Climate phase: '{epoch_name}' (temp offset {temp_offset:+.1f}C)",
            ]
            hyp = (
                "Agents may have accumulated sufficient spatial data to reliably predict resource locations."
                if acc_delta > 0 else
                "Resource distribution changes during this epoch may have outpaced stored spatial predictions."
            )
            insights.append({
                "category":    "Cognitive Performance",
                "observation": f"Population-average spatial prediction accuracy {direction} by {abs(acc_delta) * 100:.1f} pp this epoch.",
                "associated":  assoc,
                "hypothesis":  hyp,
                "confidence":  conf_score,
            })

        # ── Insight D: Social cohesion (trust) ───────────────────────────────────
        if prev and rel_count > 0:
            trust_delta = avg_trust - prev_trust
            if abs(trust_delta) > 0.05:
                direction     = "increased" if trust_delta > 0 else "decreased"
                fam_motiv_idx = 1   # family motivation slot
                fam_delta     = avg_motivs[fam_motiv_idx] - prev_motivs[fam_motiv_idx]
                conf_score    = 0.50 + 0.30 * min(1.0, n_alive / 30.0)

                assoc = [
                    f"Average trust: {prev_trust:.3f} → {avg_trust:.3f} ({trust_delta:+.3f})",
                    f"Average attachment: {avg_attachment:.3f}",
                    f"Total relationship samples: {rel_count}",
                    f"Family motivation: {_pct(fam_delta / max(abs(prev_motivs[fam_motiv_idx]), 1e-6))} change",
                ]
                hyp = (
                    "Persistent co-habitation or shelter sharing may be associated with elevated trust values."
                    if trust_delta > 0 else
                    "Increased competition or territorial disputes may be associated with eroded inter-agent trust."
                )
                insights.append({
                    "category":    "Social Dynamics",
                    "observation": f"Population-average trust {direction} by {abs(trust_delta):.3f} units this epoch.",
                    "associated":  assoc,
                    "hypothesis":  hyp,
                    "confidence":  conf_score,
                })

        return insights

    # ────────────────────────────────────────────────────────────────────────────
    # Markdown writer
    # ────────────────────────────────────────────────────────────────────────────
    def _write_entry(
        self, f, tick, year, n_alive, total_agents, epoch_name,
        temp_offset, water_mult, max_gen, top_lineages, extinct_founders,
        gene_means, gene_vars, avg_drives, avg_motivs, motiv_names,
        avg_trust, avg_attachment, avg_pred_acc, insights
    ):
        f.write(f"## Year {year}  (Tick {tick})\n\n")

        # Summary table
        f.write("### Population Summary\n\n")
        f.write("| Metric | Value |\n")
        f.write("| :----- | :---- |\n")
        f.write(f"| Agents alive | {n_alive} / {total_agents} |\n")
        f.write(f"| Maximum generation | Gen {max_gen} |\n")
        f.write(f"| Climate epoch | {epoch_name} (temp {temp_offset:+.1f}C, water {water_mult:.2f}x) |\n")
        f.write(f"| Avg trust | {avg_trust:.3f} |\n")
        f.write(f"| Avg attachment | {avg_attachment:.3f} |\n")
        f.write(f"| Avg prediction accuracy | {avg_pred_acc * 100:.1f}% |\n")
        if top_lineages:
            lin_str = "  ".join([f"Founder #{l['founder_id']} ({l['living_count']} alive)" for l in top_lineages])
            f.write(f"| Dominant lineages | {lin_str} |\n")
        if extinct_founders:
            ext_str = ", ".join([f"#{fid}" for fid in extinct_founders[:10]])
            if len(extinct_founders) > 10:
                ext_str += f" (+{len(extinct_founders) - 10} more)"
            f.write(f"| Extinct lineages | {ext_str} |\n")

        # Genetic shift table
        f.write("\n### Genetic Means\n\n")
        f.write("| Gene / Trait | Baseline | Current | Shift | Variance |\n")
        f.write("| :----------- | :------- | :------ | :---- | :------- |\n")
        for idx, name in enumerate(GENE_NAMES):
            shift = gene_means[idx] - self.baseline_genes[idx]
            f.write(f"| {name} | {self.baseline_genes[idx]:.3f} | {gene_means[idx]:.3f} | {shift:+.3f} | {gene_vars[idx]:.4f} |\n")

        # Motivation averages table
        f.write("\n### Motivation Averages\n\n")
        f.write("| Motivation | Level |\n")
        f.write("| :--------- | :---- |\n")
        for name, val in zip(motiv_names, avg_motivs):
            f.write(f"| {name} | {val:.3f} |\n")

        # Scientific observations (evidence-first format)
        if insights:
            f.write("\n### Scientific Observations\n\n")
            for ins in insights:
                f.write(f"#### Observation\n{ins['observation']}\n\n")
                f.write(f"**Category:** {ins['category']}\n\n")
                f.write("**Associated Changes:**\n\n")
                for item in ins["associated"]:
                    f.write(f"- {item}\n")
                f.write(f"\n**Possible Hypothesis:**\n{ins['hypothesis']}\n\n")
                conf_label = _conf_label(ins["confidence"])
                f.write(f"**Confidence:** {conf_label} ({ins['confidence'] * 100:.0f}%) — based on colony congruency and population size.\n\n")

        f.write("---\n\n")

    def trace_founder(self, agent, world: WorldState) -> int:
        """Recursively trace an agent back to its Gen 0 founder ID."""
        curr  = agent
        limit = 200   # safeguard against reference cycles
        while curr.parent_ids is not None and limit > 0:
            parent_id = curr.parent_ids[0]   # trace through parent A
            parent    = next((a for a in world.agents if a.id == parent_id), None)
            if parent is None:
                return parent_id   # founder is dead and purged
            curr = parent
            limit -= 1
        return curr.id
