import math
import collections

class Policy:
    def __init__(self, info: dict):
        self.info = info
        self.arena = info["arena_size"]
        self.budget = info["energy_budget"]
        self.base_xy = info["base_xy"]
        self.cell_size = info["cell_size"]
        
        # Dynamically extract hardware capabilities from info
        self.alt_min, self.alt_max = info["altitude_range"]
        self.speed_min, self.speed_max = info["speed_range"]
        self.max_climb = info["max_climb_rate"]
        
        # Extract environment radii
        self.land_radius = info.get("land_radius", 15.0)
        self.match_radius = info.get("match_radius", 5.0)
        
        # Adaptive flight parameters
        self.CRUISE_ALT = self.alt_max * 0.5        # Mid-range for good coverage vs noise balance
        self.INVESTIGATE_ALT = self.alt_min + 2.0   # Just above minimum floor to clearly see triage
        self.CRUISE_SPEED = self.speed_max          # Maximize sweep efficiency
        
        # Adaptive tracking radii based on the simulator's scoring tolerance
        self.DEDUP_RADIUS = self.match_radius * 2.5
        self.DECLARE_RADIUS = self.match_radius * 1.5
        
        # Dynamic ghost filter timeout: time to descend + time to hover + 5s padding
        alt_diff = abs(self.CRUISE_ALT - self.INVESTIGATE_ALT)
        self.GHOST_TIMEOUT = (alt_diff / self.max_climb) + 5.0
        
        # State tracking
        self.state = "SEARCH"
        self.power_est = 2.0
        self.last_action = None
        self.declared = [] 
        self.pending_targets = []
        self.current_investigation = None
        self.returning = False
        
        # Build intelligent sweep path based on the prior map probabilities
        self.waypoints = collections.deque()
        self._build_waypoints(info["prior_map"])
        
    def _build_waypoints(self, prior_map):
        """Calculates a TSP path using a statistical baseline instead of hardcoded thresholds."""
        points = []
        
        # Calculate the dynamic mean probability of the entire map
        all_probs = [prob for row in prior_map for prob in row]
        mean_prob = sum(all_probs) / len(all_probs) if all_probs else 0
        
        for ix in range(len(prior_map)):
            for iy in range(len(prior_map[ix])):
                prob = prior_map[ix][iy]
                # Dynamically filter: keep cells strictly above the map's average probability
                if prob > mean_prob and prob > 0.0:
                    cx = (ix * self.cell_size) + (self.cell_size / 2.0)
                    cy = (iy * self.cell_size) + (self.cell_size / 2.0)
                    points.append((prob, cx, cy))
        
        # Sort by highest probability
        points.sort(key=lambda x: x[0], reverse=True)
        top_points = [pt[1:] for pt in points]
        
        # Greedy Traveling Salesperson route
        curr = tuple(self.base_xy)
        while top_points:
            best_idx = 0
            best_dist = float('inf')
            for i, pt in enumerate(top_points):
                d = math.hypot(pt[0] - curr[0], pt[1] - curr[1])
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            
            curr = top_points.pop(best_idx)
            self.waypoints.append(curr)

    def act(self, obs: dict) -> dict:
        t = obs["t"]
        x, y = obs["x"], obs["y"]
        alt = obs["altitude"]
        
        # 1. Update dynamic power estimate (accounts for wind variation)
        if self.last_action and self.last_action[0] == "fly_to" and t > self.last_action[1]:
            dt = t - self.last_action[1]
            if dt > 0.01:
                pwr = obs["energy_used_last_action"] / dt
                self.power_est = 0.7 * self.power_est + 0.3 * pwr

        # 2. Check dynamic battery safety margin
        home_dist = math.hypot(x - self.base_xy[0], y - self.base_xy[1])
        ret_cost = (home_dist / self.CRUISE_SPEED) * self.power_est * 1.5 + (0.04 * self.budget)
        
        if not self.returning and obs["energy_remaining"] < ret_cost:
            self.returning = True
            
        if self.returning:
            # Safely land if inside the environment's dynamic land_radius
            if home_dist <= self.land_radius * 0.9: 
                return {"type": "land"}
            self.last_action = ("fly_to", t)
            return {"type": "fly_to", "x": self.base_xy[0], "y": self.base_xy[1], "altitude": self.CRUISE_ALT, "speed": self.CRUISE_SPEED}

        # 3. Process new detections
        for d in obs["detections"]:
            dx, dy, dtri = d["x"], d["y"], d["triage"]
            
            # Prevent false declarations by checking distance against already declared
            if all(math.hypot(dx - px, dy - py) > self.DECLARE_RADIUS for px, py in self.declared):
                if dtri is not None:
                    # Confirmed triage -> Declare immediately
                    self.declared.append((dx, dy))
                    self.last_action = ("declare", t)
                    
                    # Clear from pending lists
                    if self.current_investigation and math.hypot(dx - self.current_investigation["x"], dy - self.current_investigation["y"]) < self.DEDUP_RADIUS:
                        self.current_investigation = None
                    self.pending_targets = [pt for pt in self.pending_targets if math.hypot(dx - pt["x"], dy - pt["y"]) > self.DEDUP_RADIUS]
                    
                    return {"type": "declare", "x": dx, "y": dy, "triage": dtri}
                else:
                    # Blip detected -> Add to pending investigations if unique
                    is_new = True
                    if self.current_investigation and math.hypot(dx - self.current_investigation["x"], dy - self.current_investigation["y"]) < self.DEDUP_RADIUS:
                        is_new = False
                    for pt in self.pending_targets:
                        if math.hypot(dx - pt["x"], dy - pt["y"]) < self.DEDUP_RADIUS:
                            is_new = False
                            break
                    if is_new:
                        self.pending_targets.append({"x": dx, "y": dy, "t_added": t, "t_reached": None})

        # 4. Manage active investigation
        if self.current_investigation:
            cx, cy = self.current_investigation["x"], self.current_investigation["y"]
            
            # Dynamic ghost filter timeout based on descent physics
            if t - self.current_investigation["t_added"] > self.GHOST_TIMEOUT:
                self.current_investigation = None
            else:
                d_horiz = math.hypot(x - cx, y - cy)
                
                # Check if arrived at investigation zone
                if d_horiz < 2.0 and abs(alt - self.INVESTIGATE_ALT) < 1.0:
                    if self.current_investigation["t_reached"] is None:
                        self.current_investigation["t_reached"] = t
                    elif t - self.current_investigation["t_reached"] > 2.0:
                        # Hovered over the spot for 2 seconds and still no triage -> discard
                        self.current_investigation = None
                
                if self.current_investigation:
                    if d_horiz < 2.0 and abs(alt - self.INVESTIGATE_ALT) < 1.0:
                        self.last_action = ("hover", t)
                        return {"type": "hover", "duration": 1.0}
                    else:
                        # Dynamic approach altitude
                        tgt_alt = self.INVESTIGATE_ALT if d_horiz < (self.CRUISE_ALT * 1.5) else self.CRUISE_ALT
                        self.last_action = ("fly_to", t)
                        return {"type": "fly_to", "x": cx, "y": cy, "altitude": tgt_alt, "speed": self.CRUISE_SPEED}

        # 5. Route to next pending target if one exists
        if self.pending_targets:
            self.pending_targets.sort(key=lambda pt: math.hypot(x - pt["x"], y - pt["y"]))
            self.current_investigation = self.pending_targets.pop(0)
            self.current_investigation["t_added"] = t 
            
            cx, cy = self.current_investigation["x"], self.current_investigation["y"]
            self.last_action = ("fly_to", t)
            return {"type": "fly_to", "x": cx, "y": cy, "altitude": self.CRUISE_ALT, "speed": self.CRUISE_SPEED}

        # 6. Follow standard TSP sweep waypoints
        if self.waypoints:
            tx, ty = self.waypoints[0]
            # Use dynamic cell size to detect waypoint arrival
            if math.hypot(x - tx, y - ty) < (self.cell_size * 0.8):
                self.waypoints.popleft()
                if self.waypoints:
                    tx, ty = self.waypoints[0]
                else:
                    self.returning = True
                    self.last_action = ("hover", t)
                    return {"type": "hover", "duration": 1.0}
            
            self.last_action = ("fly_to", t)
            return {"type": "fly_to", "x": tx, "y": ty, "altitude": self.CRUISE_ALT, "speed": self.CRUISE_SPEED}

        # 7. Fallback if map is cleared
        self.returning = True
        self.last_action = ("hover", t)
        return {"type": "hover", "duration": 1.0}
