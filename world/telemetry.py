import numpy as np

class TelemetryCollector:
    """
    Centralized collector for project Genesis simulation telemetry.
    Maintains O(1) running statistics (Welford's algorithm) and causal funnels.
    """
    def __init__(self, world):
        self.world = world
        
        # --- 1. Reproduction Funnel Stats ---
        self.repro_potential_opportunities = 0
        self.repro_actual_opportunities = 0
        self.repro_mate_exists_world_ticks = 0
        self.repro_mate_exists_radius_ticks = 0
        self.repro_relationship_ok_ticks = 0
        self.repro_mate_eligible_ticks = 0
        self.repro_wanted_reproduction_ticks = 0
        self.repro_mutual_will_ticks = 0
        self.repro_successes = 0

        # Detailed Reproduction Biological Failures
        self.repro_fail_cooldown = 0
        self.repro_fail_health = 0
        self.repro_fail_fat = 0
        self.repro_fail_hunger = 0
        self.repro_fail_thirst = 0
        self.repro_fail_shelter = 0
        self.repro_fail_injury = 0

        # Detailed Reproduction Mate/Distance Failures
        self.repro_fail_distance = 0
        self.repro_fail_no_mate = 0
        self.repro_fail_relationship = 0
        self.repro_fail_mate_ineligible = 0

        # Detailed Reproduction Coordination Failures
        self.repro_fail_low_utility = 0
        self.repro_fail_mate_unwilling = 0
        self.repro_fail_no_mate_nearby = 0

        # Reproduction Lost To Reasons (Action name -> count)
        self.repro_lost_to_counts = {}
        self.repro_lost_margin_sum = 0.0
        self.repro_lost_margin_count = 0

        # Online running statistics for reproduction utility by age bracket
        self.utility_stats_by_bracket = {
            b: {
                "count": 0,
                "mean_repro": 0.0,
                "M2_repro": 0.0,
                "mean_win": 0.0,
                "M2_win": 0.0,
                "min_repro": 99999.0,
                "max_repro": -99999.0,
                "min_win": 99999.0,
                "max_win": -99999.0
            } for b in ("juvenile", "young_adult", "mature_adult", "elder")
        }

        # --- 2. Water Economy Audit Metrics ---
        self.water_searches = 0
        self.water_search_successes = 0
        self.water_search_failures = 0
        self.water_search_distance_sum = 0.0
        self.water_search_distance_count = 0
        
        # Time to first water metrics (ticks from start to drink)
        self.water_time_to_first_sum = 0
        self.water_time_to_first_count = 0

        # Path efficiency metrics: sum of (straight_distance / actual_distance)
        self.water_path_efficiency_sum = 0.0
        self.water_path_efficiency_count = 0

        # Dehydration Deaths Telemetry
        self.dehydration_deaths = 0
        self.died_carrying_water = 0
        self.died_beside_water = 0
        self.died_with_remembered_water = 0
        self.died_after_dry_visit = 0

        # Births per generation (Gen -> births count)
        self.births_by_generation = {}
        
        # Resource timeline snapshots (every 500 ticks)
        self.resource_timeline = []

    def record_utility_decision(self, age_ticks, repro_utility, winning_action, winning_utility):
        """Updates O(1) online utility statistics for age brackets using Welford's algorithm."""
        age_yrs = age_ticks / 360.0
        if age_yrs < 14.0:
            b = "juvenile"
        elif age_yrs < 30.0:
            b = "young_adult"
        elif age_yrs < 60.0:
            b = "mature_adult"
        else:
            b = "elder"

        stats = self.utility_stats_by_bracket[b]
        stats["count"] += 1
        n = stats["count"]
        
        # Update repro utility online stats
        delta = repro_utility - stats["mean_repro"]
        stats["mean_repro"] += delta / n
        stats["M2_repro"] += delta * (repro_utility - stats["mean_repro"])
        if repro_utility < stats["min_repro"]: stats["min_repro"] = repro_utility
        if repro_utility > stats["max_repro"]: stats["max_repro"] = repro_utility

        # Update winning utility online stats
        delta_win = winning_utility - stats["mean_win"]
        stats["mean_win"] += delta_win / n
        stats["M2_win"] += delta_win * (winning_utility - stats["mean_win"])
        if winning_utility < stats["min_win"]: stats["min_win"] = winning_utility
        if winning_utility > stats["max_win"]: stats["max_win"] = winning_utility
        
        # Record why reproduction lost
        if b in ("young_adult", "mature_adult") and winning_action != "Reproduce" and repro_utility > 0.0:
            self.repro_lost_to_counts[winning_action] = self.repro_lost_to_counts.get(winning_action, 0) + 1
            margin = winning_utility - repro_utility
            self.repro_lost_margin_sum += margin
            self.repro_lost_margin_count += 1

    def record_repro_terminal(self, category):
        """Records terminal causal outcomes to guarantee they sum up to potential opportunities."""
        if category == "cooldown": self.repro_fail_cooldown += 1
        elif category == "health": self.repro_fail_health += 1
        elif category == "fat": self.repro_fail_fat += 1
        elif category == "hunger": self.repro_fail_hunger += 1
        elif category == "thirst": self.repro_fail_thirst += 1
        elif category == "shelter": self.repro_fail_shelter += 1
        elif category == "injury": self.repro_fail_injury += 1
        elif category == "distance": self.repro_fail_distance += 1
        elif category == "no_mate": self.repro_fail_no_mate += 1
        elif category == "relationship": self.repro_fail_relationship += 1
        elif category == "mate_ineligible": self.repro_fail_mate_ineligible += 1
        elif category == "low_utility": self.repro_fail_low_utility += 1
        elif category == "mate_unwilling": self.repro_fail_mate_unwilling += 1
        elif category == "no_mate_nearby": self.repro_fail_no_mate_nearby += 1
        elif category == "birth": self.repro_successes += 1

    def record_water_search_start(self, straight_dist):
        self.water_searches += 1
        self.water_search_distance_sum += straight_dist
        self.water_search_distance_count += 1

    def record_water_search_outcome(self, success, ticks_taken, efficiency):
        if success:
            self.water_search_successes += 1
            self.water_time_to_first_sum += ticks_taken
            self.water_time_to_first_count += 1
            self.water_path_efficiency_sum += efficiency
            self.water_path_efficiency_count += 1
        else:
            self.water_search_failures += 1

    def record_dehydration_death(self, carrying_water, beside_water, remembered_water, recently_visited_dry):
        self.dehydration_deaths += 1
        if carrying_water: self.died_carrying_water += 1
        if beside_water: self.died_beside_water += 1
        if remembered_water: self.died_with_remembered_water += 1
        if recently_visited_dry: self.died_after_dry_visit += 1

    def record_birth(self, child_gen):
        self.births_by_generation[child_gen] = self.births_by_generation.get(child_gen, 0) + 1

    def generate_dashboard_report(self):
        """Generates self-interpreting health diagnostics and flags critical anomalies."""
        total_failures = (
            self.repro_fail_cooldown + self.repro_fail_health + self.repro_fail_fat +
            self.repro_fail_hunger + self.repro_fail_thirst + self.repro_fail_shelter +
            self.repro_fail_injury + self.repro_fail_distance + self.repro_fail_no_mate +
            self.repro_fail_relationship + self.repro_fail_mate_ineligible +
            self.repro_fail_low_utility + self.repro_fail_mate_unwilling +
            self.repro_fail_no_mate_nearby
        )
        total_repro_recorded = total_failures + self.repro_successes
        
        replacement_rate = self.repro_successes / max(1, len(self.world.agents))
        status_repro = "🟢 Healthy" if replacement_rate >= 1.0 else "🔴 Unsustainable"
        
        dehydration_deaths_pct = (self.dehydration_deaths / max(1, getattr(self.world, "total_deaths", 0))) * 100.0 if getattr(self.world, "total_deaths", 0) > 0 else 0.0
        status_water = "🟢 Normal" if dehydration_deaths_pct < 25.0 else ("🟡 Moderate" if dehydration_deaths_pct < 50.0 else "🔴 Critical Bottleneck")

        mate_exists_pct = (self.repro_mate_exists_radius_ticks / max(1, self.repro_actual_opportunities)) * 100.0
        status_mate = "🟢 Healthy" if mate_exists_pct > 70.0 else "🔴 Isolated Populations"
        
        report = []
        report.append("=" * 60)
        report.append("                 CIVILIZATION HEALTH MONITOR")
        report.append("=" * 60)
        report.append(f"  Replacement Rate status  : {status_repro} (R={replacement_rate:.2f})")
        report.append(f"  Water economy stress     : {status_water} ({dehydration_deaths_pct:.1f}% of deaths)")
        report.append(f"  Mate availability status : {status_mate} ({mate_exists_pct:.1f}% in radius)")
        report.append("-" * 60)
        report.append("  CAUSAL REPRODUCTION FUNNEL CONVERSIONS:")
        report.append(f"    Potential Opportunities (Fertile Age) : {self.repro_potential_opportunities}")
        report.append(f"    Biologically Eligible Opportunities  : {self.repro_actual_opportunities} ({self.repro_actual_opportunities / max(1, self.repro_potential_opportunities)*100:.1f}%)")
        report.append(f"    Mate Existed in World Ticks           : {self.repro_mate_exists_world_ticks} ({self.repro_mate_exists_world_ticks / max(1, self.repro_actual_opportunities)*100:.1f}%)")
        report.append(f"    Mate inside Search Radius Ticks       : {self.repro_mate_exists_radius_ticks} ({self.repro_mate_exists_radius_ticks / max(1, self.repro_actual_opportunities)*100:.1f}%)")
        report.append(f"    Acceptable Relationship trust Ticks   : {self.repro_relationship_ok_ticks} ({self.repro_relationship_ok_ticks / max(1, self.repro_mate_exists_radius_ticks)*100:.1f}%)")
        report.append(f"    Mate was Biologically Eligible Ticks  : {self.repro_mate_eligible_ticks} ({self.repro_mate_eligible_ticks / max(1, self.repro_relationship_ok_ticks)*100:.1f}%)")
        report.append(f"    Planner Chose Reproduction Action     : {self.repro_wanted_reproduction_ticks} ({self.repro_wanted_reproduction_ticks / max(1, self.repro_mate_eligible_ticks)*100:.1f}%)")
        report.append(f"    Mutual Consent Coordination Ticks     : {self.repro_mutual_will_ticks} ({self.repro_mutual_will_ticks / max(1, self.repro_wanted_reproduction_ticks)*100:.1f}%)")
        report.append(f"    Successful Births                     : {self.repro_successes} ({self.repro_successes / max(1, self.repro_mutual_will_ticks)*100:.1f}%)")
        report.append("-" * 60)
        report.append("  REPRODUCTION LOSS TERMINAL OUTCOMES:")
        report.append(f"    Successful Births                     : {self.repro_successes}")
        report.append(f"    Biological Rejection (Cooldown)       : {self.repro_fail_cooldown}")
        report.append(f"    Biological Rejection (Low Health)     : {self.repro_fail_health}")
        report.append(f"    Biological Rejection (Low Fat)        : {self.repro_fail_fat}")
        report.append(f"    Biological Rejection (Hunger)         : {self.repro_fail_hunger}")
        report.append(f"    Biological Rejection (Thirst)         : {self.repro_fail_thirst}")
        report.append(f"    Biological Rejection (Low Shelter)     : {self.repro_fail_shelter}")
        report.append(f"    Biological Rejection (Injury)         : {self.repro_fail_injury}")
        report.append(f"    Unreachable Partner (Distance)        : {self.repro_fail_distance}")
        report.append(f"    No Mate Existed (Colony Extinct)      : {self.repro_fail_no_mate}")
        report.append(f"    Relationship Rejection (Low Trust)     : {self.repro_fail_relationship}")
        report.append(f"    Mate Biologically Ineligible          : {self.repro_fail_mate_ineligible}")
        report.append(f"    Drive Selection (Other Priorities)    : {self.repro_fail_low_utility}")
        report.append(f"    Mate Unwilling (Chose other action)   : {self.repro_fail_mate_unwilling}")
        report.append(f"    Mate Moved/Died before coordination   : {self.repro_fail_no_mate_nearby}")
        report.append(f"    Causal Balance Total Recorded         : {total_repro_recorded} (Expected potential: {self.repro_potential_opportunities})")
        report.append("-" * 60)
        
        # Reproduction lost to details
        if self.repro_lost_to_counts:
            report.append("  REPRODUCTION ACTION LOSS DISTRIBUTION:")
            for act, count in sorted(self.repro_lost_to_counts.items(), key=lambda x: x[1], reverse=True):
                pct = (count / max(1, self.repro_fail_low_utility)) * 100.0
                report.append(f"    - Lost to {act:20} : {count:6d} times ({pct:5.1f}%)")
            avg_margin = self.repro_lost_margin_sum / max(1, self.repro_lost_margin_count)
            report.append(f"    Average winning margin utility diff : {avg_margin:.2f}")
            report.append("-" * 60)
            
        # Water Economy Telemetry
        avg_search_dist = self.water_search_distance_sum / max(1, self.water_search_distance_count)
        avg_time = self.water_time_to_first_sum / max(1, self.water_time_to_first_count)
        avg_eff = self.water_path_efficiency_sum / max(1, self.water_path_efficiency_count)
        success_rate = (self.water_search_successes / max(1, self.water_searches)) * 100.0
        
        report.append("  WATER ECONOMY AND PATH TELEMETRY:")
        report.append(f"    Total Searches Started                : {self.water_searches}")
        report.append(f"    Search Successes / Failures           : {self.water_search_successes} / {self.water_search_failures} ({success_rate:.1f}% success)")
        report.append(f"    Average Target Straight Distance      : {avg_search_dist:.2f} cells")
        report.append(f"    Average Time to Hydration (Ticks)     : {avg_time:.2f} ticks")
        report.append(f"    Average Path Traversal Efficiency     : {avg_eff*100:.1f}%")
        report.append(f"    Dehydration Deaths Telemetry          : {self.dehydration_deaths} deaths")
        report.append(f"      - Carrying Water at death           : {self.died_carrying_water}")
        report.append(f"      - Beside Water source               : {self.died_beside_water}")
        report.append(f"      - Had water locations in memory     : {self.died_with_remembered_water}")
        report.append(f"      - Visited dried source recently     : {self.died_after_dry_visit}")
        report.append("-" * 60)

        # Births by generation
        if self.births_by_generation:
            report.append("  BIRTHS BY GENERATION LINEAGES:")
            for gen in sorted(self.births_by_generation.keys()):
                report.append(f"    - Generation {gen:2d} Births                 : {self.births_by_generation[gen]}")
            report.append("-" * 60)

        # Anomaly Warnings Detection
        anomalies = []
        if self.dehydration_deaths > 0:
            pct_carrying = (self.died_carrying_water / self.dehydration_deaths) * 100.0
            if pct_carrying > 50.0:
                anomalies.append(f"⚠️ ANOMALY: {pct_carrying:.1f}% of dehydration deaths occurred while carrying water. Hydration intake delay.")
            pct_beside = (self.died_beside_water / self.dehydration_deaths) * 100.0
            if pct_beside > 50.0:
                anomalies.append(f"⚠️ ANOMALY: {pct_beside:.1f}% of dehydration deaths occurred adjacent to a water source. Reachability/obstacle failure.")
            pct_dry = (self.died_after_dry_visit / self.dehydration_deaths) * 100.0
            if pct_dry > 50.0:
                anomalies.append(f"⚠️ ANOMALY: {pct_dry:.1f}% of dehydration deaths occurred after visiting a dry source. Extreme environmental stress.")
                
        if self.repro_potential_opportunities > 0:
            repro_actual_fail_pct = (self.repro_fail_low_utility / max(1, self.repro_actual_opportunities)) * 100.0 if self.repro_actual_opportunities > 0 else 0.0
            if repro_actual_fail_pct > 75.0:
                anomalies.append(f"⚠️ ANOMALY: {repro_actual_fail_pct:.1f}% of eligible opportunities rejected because other drives took priority.")
                
            dist_fail_pct = (self.repro_fail_distance / max(1, self.repro_potential_opportunities)) * 100.0
            if dist_fail_pct > 50.0:
                anomalies.append(f"⚠️ ANOMALY: {dist_fail_pct:.1f}% of opportunities failed due to spatial distance. Population is too dispersed.")

        if anomalies:
            report.append("  DETECTED SYSTEM ANOMALIES:")
            for a in anomalies:
                report.append(f"    {a}")
            report.append("-" * 60)
            
        report.append("=" * 60 + "\n")
        return "\n".join(report)

    def generate_hypotheses(self):
        """
        Analyzes simulation history to construct structured hypotheses with confidence scores,
        supporting evidence, and alternative explanations.
        Returns a list of dicts.
        """
        hypotheses = []
        if len(self.resource_timeline) < 2:
            return hypotheses
            
        # 1. Group snapshots by climate epoch
        epochs_data = []
        current_epoch_snapshots = []
        last_epoch_name = None
        
        for snap in self.resource_timeline:
            epoch_name = snap.get("climate", "Temperate")
            if last_epoch_name is None:
                last_epoch_name = epoch_name
            if epoch_name != last_epoch_name:
                if current_epoch_snapshots:
                    epochs_data.append({
                        "name": last_epoch_name,
                        "snapshots": current_epoch_snapshots
                    })
                current_epoch_snapshots = []
                last_epoch_name = epoch_name
            current_epoch_snapshots.append(snap)
            
        if current_epoch_snapshots:
            epochs_data.append({
                "name": last_epoch_name,
                "snapshots": current_epoch_snapshots
            })
            
        # Fallback to block partitions if climate doesn't change
        if len(epochs_data) == 1:
            epochs_data = []
            block_size = 4  # 2000 ticks per block
            for i in range(0, len(self.resource_timeline), block_size):
                chunk = self.resource_timeline[i:i+block_size]
                if chunk:
                    start_year = chunk[0]["tick"] // 360
                    end_year = chunk[-1]["tick"] // 360
                    epochs_data.append({
                        "name": f"Years {start_year}-{end_year}",
                        "snapshots": chunk
                    })
                    
        # 2. Summarize each epoch block
        summaries = []
        for ep in epochs_data:
            snaps = ep["snapshots"]
            mean_pop = np.mean([s["alive"] for s in snaps])
            mean_hunger = np.mean([s["mean_hunger"] for s in snaps])
            mean_thirst = np.mean([s["mean_thirst"] for s in snaps])
            mean_shelter = np.mean([s["mean_shelter"] for s in snaps])
            stored_food = np.mean([s["stored_food"] for s in snaps])
            stored_water = np.mean([s["stored_water"] for s in snaps])
            
            ticks_elapsed = snaps[-1]["tick"] - snaps[0]["tick"] + 500
            births_count = snaps[-1]["births"] - snaps[0]["births"]
            birth_rate = (births_count / ticks_elapsed) * 360.0 if ticks_elapsed > 0 else 0.0
            
            deaths_count = snaps[-1]["deaths"] - snaps[0]["deaths"]
            death_rate = (deaths_count / ticks_elapsed) * 360.0 if ticks_elapsed > 0 else 0.0
            
            summaries.append({
                "name": ep["name"],
                "start_tick": snaps[0]["tick"],
                "end_tick": snaps[-1]["tick"],
                "mean_pop": float(mean_pop),
                "mean_hunger": float(mean_hunger),
                "mean_thirst": float(mean_thirst),
                "mean_shelter": float(mean_shelter),
                "stored_food": float(stored_food),
                "stored_water": float(stored_water),
                "birth_rate": float(birth_rate),
                "death_rate": float(death_rate)
            })

        # 3. Analyze adjacent transitions to build hypotheses
        h_idx = 1
        for i in range(1, len(summaries)):
            prev = summaries[i-1]
            curr = summaries[i]
            
            birth_change_pct = (curr["birth_rate"] - prev["birth_rate"]) / max(0.01, prev["birth_rate"]) * 100.0
            pop_change_pct = (curr["mean_pop"] - prev["mean_pop"]) / max(0.01, prev["mean_pop"]) * 100.0
            thirst_change_pct = (curr["mean_thirst"] - prev["mean_thirst"]) / max(0.01, prev["mean_thirst"]) * 100.0
            hunger_change_pct = (curr["mean_hunger"] - prev["mean_hunger"]) / max(0.01, prev["mean_hunger"]) * 100.0
            water_stored_change_pct = (curr["stored_water"] - prev["stored_water"]) / max(0.01, prev["stored_water"]) * 100.0
            food_stored_change_pct = (curr["stored_food"] - prev["stored_food"]) / max(0.01, prev["stored_food"]) * 100.0
            
            start_yr = prev["start_tick"] // 360
            end_yr = curr["end_tick"] // 360
            
            # Hypothesis A: Water Scarcity / Drought impact
            if birth_change_pct < -15.0 or pop_change_pct < -15.0:
                is_dry_climate = "Drought" in curr["name"] or "Heatwave" in curr["name"]
                is_thirst_stress = thirst_change_pct > 20.0
                is_water_depletion = water_stored_change_pct < -20.0
                
                if is_dry_climate or is_thirst_stress or is_water_depletion:
                    # Calculate statistical confidence based on effect size overlap
                    confidence = 0.5
                    evidence = []
                    alternatives = []
                    
                    if is_dry_climate:
                        confidence += 0.15
                        evidence.append(f"Climate epoch transitioned to arid state ({curr['name']}).")
                    if is_thirst_stress:
                        confidence += 0.15
                        evidence.append(f"Average agent thirst increased by {thirst_change_pct:.1f}%.")
                    if is_water_depletion:
                        confidence += 0.1
                        evidence.append(f"Colony water reserves decreased by {abs(water_stored_change_pct):.1f}%.")
                    if self.dehydration_deaths > 0:
                        confidence += 0.05
                        evidence.append(f"Dehydration mortality occurred ({self.dehydration_deaths} deaths).")
                        
                    # Alternatives checks
                    if abs(hunger_change_pct) < 15.0:
                        alternatives.append("Nutritional stress (hunger) remained stable.")
                    else:
                        alternatives.append(f"Nutritional stress (hunger) changed by {hunger_change_pct:.1f}%.")
                    if curr["mean_shelter"] >= prev["mean_shelter"] * 0.9:
                        alternatives.append("Shelter infrastructure remained functional.")
                    else:
                        alternatives.append("Shelter durability degradation could contribute.")
                        
                    hypotheses.append({
                        "id": f"H-{h_idx:04d}",
                        "category": "Ecology",
                        "title": f"Water scarcity reduced survival and reproduction utility during {curr['name']}",
                        "confidence": min(0.95, float(confidence)),
                        "evidence": evidence,
                        "alternatives": alternatives,
                        "affected_metrics": ["population", "birth_rate", "thirst", "stored_water"],
                        "epoch": [int(start_yr), int(end_yr)],
                        "related_events": [f"{curr['name']} began"],
                        "unexplained_variance": round(1.0 - min(0.95, float(confidence)), 2)
                    })
                    h_idx += 1
                    
                # Hypothesis B: Nutritional Scarcity / Famine impact
                is_famine_climate = "Famine" in curr["name"]
                is_hunger_stress = hunger_change_pct > 20.0
                is_food_depletion = food_stored_change_pct < -20.0
                
                if is_famine_climate or is_hunger_stress or is_food_depletion:
                    confidence = 0.5
                    evidence = []
                    alternatives = []
                    
                    if is_famine_climate:
                        confidence += 0.2
                        evidence.append(f"Climate epoch transitioned to Famine.")
                    if is_hunger_stress:
                        confidence += 0.15
                        evidence.append(f"Average agent hunger increased by {hunger_change_pct:.1f}%.")
                    if is_food_depletion:
                        confidence += 0.1
                        evidence.append(f"Colony food reserves decreased by {abs(food_stored_change_pct):.1f}%.")
                        
                    if abs(thirst_change_pct) < 15.0:
                        alternatives.append("Hydration stress (thirst) remained stable.")
                    else:
                        alternatives.append(f"Hydration stress (thirst) changed by {thirst_change_pct:.1f}%.")
                        
                    hypotheses.append({
                        "id": f"H-{h_idx:04d}",
                        "category": "Nutrition",
                        "title": f"Nutritional scarcity impaired population growth during {curr['name']}",
                        "confidence": min(0.95, float(confidence)),
                        "evidence": evidence,
                        "alternatives": alternatives,
                        "affected_metrics": ["population", "birth_rate", "hunger", "stored_food"],
                        "epoch": [int(start_yr), int(end_yr)],
                        "related_events": [f"{curr['name']} began"],
                        "unexplained_variance": round(1.0 - min(0.95, float(confidence)), 2)
                    })
                    h_idx += 1
                    
                # Hypothesis C: Density-dependent overcrowding
                if prev["mean_pop"] > 25.0:
                    hypotheses.append({
                        "id": f"H-{h_idx:04d}",
                        "category": "Social",
                        "title": "Territorial friction and density-dependent pressures reduced birth conversions",
                        "confidence": 0.65,
                        "evidence": [
                            f"Initial population size was high ({prev['mean_pop']:.1f} agents).",
                            f"Birth rates declined by {abs(birth_change_pct):.1f}% as population density peaked."
                        ],
                        "alternatives": [
                            "Climatic stress acts as a primary or synergistic driver.",
                            "Average relationship trust shifts could modify mating frequency."
                        ],
                        "affected_metrics": ["population", "birth_rate"],
                        "epoch": [int(start_yr), int(end_yr)],
                        "related_events": ["Population peak reached"],
                        "unexplained_variance": 0.35
                    })
                    h_idx += 1

            # Recovery hypothesis
            if birth_change_pct > 20.0 or pop_change_pct > 15.0:
                if "Temperate" in curr["name"]:
                    hypotheses.append({
                        "id": f"H-{h_idx:04d}",
                        "category": "Ecology",
                        "title": f"Restoration of temperate seasons revived reproduction utility in {curr['name']}",
                        "confidence": 0.85,
                        "evidence": [
                            f"Temperate conditions returned in {curr['name']}.",
                            f"Population increased by {pop_change_pct:.1f}%.",
                            f"Birth rate improved by {birth_change_pct:.1f}%."
                        ],
                        "alternatives": [
                            "Epigenetic adaptation or selection of high-planning genomes could assist recovery."
                        ],
                        "affected_metrics": ["population", "birth_rate"],
                        "epoch": [int(start_yr), int(end_yr)],
                        "related_events": ["Temperate conditions returned"],
                        "unexplained_variance": 0.15
                    })
                    h_idx += 1
                    
        return hypotheses
