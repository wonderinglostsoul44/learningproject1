import random
import math
import os
import heapq # For A* priority queue
import time # For AI turn delay (optional)
import pickle # For saving/loading
import datetime # For save file names

# --- Constants ---
MAP_WIDTH = 12
MAP_HEIGHT = 10
INITIAL_GOLD = 100
GOLD_PER_TURN = 25
BASE_STARTING_HP = 100 # Bases can be attacked
POISON_DAMAGE = 2 # Damage per turn for poison status

# --- Terrain Definitions ---
TERRAIN_TYPES = {
    "P": {"name": "Plains", "move_cost": 1, "defense_bonus": 0, "vision_cost": 1, "symbol": "."},
    "M": {"name": "Mountain", "move_cost": 2, "defense_bonus": 2, "vision_cost": 2, "symbol": "^"},
    "F": {"name": "Forest", "move_cost": 2, "defense_bonus": 1, "vision_cost": 2, "symbol": "#"},
    "G": {"name": "Gold Mine", "move_cost": 1, "defense_bonus": 0, "vision_cost": 1, "symbol": "G", "income": 10}, # Provides income if unit waits on it
    "B": {"name": "Base", "move_cost": 1, "defense_bonus": 1, "vision_cost": 1, "symbol": "B"}, # Player Base building location
}

# --- Unit Definitions ---
# Format: "Name": {stats...}
# --- Added Scout ---
# --- Modified Warrior (Bash ability), Mage (Fireball applies Poison) ---
UNIT_STATS = {
    "Warrior": {"hp": 25, "attack": 6, "defense": 3, "attack_range": 1, "move_range": 3, "vision_range": 2, "cost": 50, "symbol": "W", "xp_value": 10, "ability": "Bash", "ability_cooldown": 5}, # Changed ability to Bash
    "Archer": {"hp": 15, "attack": 4, "defense": 1, "attack_range": 4, "move_range": 2, "vision_range": 4, "cost": 60, "symbol": "A", "xp_value": 12, "ability": "Long Shot", "ability_cooldown": 4},
    "Cavalry": {"hp": 30, "attack": 7, "defense": 2, "attack_range": 1, "move_range": 5, "vision_range": 3, "cost": 80, "symbol": "C", "xp_value": 15, "ability": "Charge", "ability_cooldown": 5, "ability_duration": 1}, # Charge gives bonus this turn
    "Mage": {"hp": 12, "attack": 5, "defense": 0, "attack_range": 3, "move_range": 2, "vision_range": 3, "cost": 70, "symbol": "M", "xp_value": 15, "ability": "Fireball", "ability_cooldown": 6}, # Fireball applies Poison chance
    "Healer": {"hp": 15, "attack": 1, "defense": 1, "attack_range": 1, "move_range": 3, "vision_range": 3, "cost": 75, "symbol": "H", "xp_value": 8, "ability": "Heal", "ability_cooldown": 3},
    "Scout": {"hp": 12, "attack": 2, "defense": 0, "attack_range": 1, "move_range": 6, "vision_range": 5, "cost": 40, "symbol": "S", "xp_value": 8, "ability": "Evade", "ability_cooldown": 4, "ability_duration": 1}, # New Unit
    "Base": {"hp": BASE_STARTING_HP, "attack": 0, "defense": 2, "attack_range": 0, "move_range": 0, "vision_range": 2, "cost": 0, "symbol": "B", "xp_value": 50, "ability": None} # Static structure unit
}

# --- Experience Levels ---
# Level: (XP Threshold, Stat Bonus) - Bonus applied cumulatively
XP_LEVELS = {
    1: (0, {"hp": 0, "attack": 0, "defense": 0}),
    2: (20, {"hp": 5, "attack": 1, "defense": 0}),
    3: (50, {"hp": 5, "attack": 1, "defense": 1}),
    4: (100, {"hp": 10, "attack": 2, "defense": 1}),
    # Add more levels as needed
}
MAX_LEVEL = max(XP_LEVELS.keys())

# --- Status Effects ---
STATUS_EFFECTS_INFO = {
    "Poison": {"symbol": "(P)", "desc": f"Takes {POISON_DAMAGE} damage per turn."},
    "Stun": {"symbol": "(S)", "desc": "Cannot act next turn."},
    "Evade": {"symbol": "(E)", "desc": "Increased defense."},
    "Shield Wall": {"symbol": "(SW)", "desc": "Increased defense."}, # Old ability effect, good to list
    "Charge": {"symbol": "(Ch)", "desc": "Increased move range."}, # Old ability effect
    # Add more descriptive names if needed
}


# --- Utility Functions ---
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) # Manhattan distance

# --- A* Pathfinding ---
class Node:
    """Node for A* Pathfinding"""
    def __init__(self, position, parent=None, g_cost=0, h_cost=0):
        self.position = position
        self.parent = parent
        self.g_cost = g_cost # Cost from start
        self.h_cost = h_cost # Heuristic cost to end
        self.f_cost = g_cost + h_cost # Total estimated cost

    def __lt__(self, other): # For priority queue comparison
        return self.f_cost < other.f_cost

    def __eq__(self, other):
        return self.position == other.position

    def __hash__(self): # Required for storing nodes in sets/dictionaries
        return hash(self.position)

def a_star_pathfinding(game_map, start_pos, end_pos, unit_move_costs):
    """
    Finds the shortest path using A*.
    unit_move_costs: A function(terrain_key) -> cost for the specific unit.
    Returns a list of positions (path) or None if no path found.
    """
    start_node = Node(start_pos, g_cost=0, h_cost=distance(start_pos, end_pos))
    end_node = Node(end_pos) # We only need the position for comparison

    open_list = [] # Priority queue (min-heap)
    closed_set = set() # Positions already evaluated

    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)

        if current_node == end_node:
            # Path found, reconstruct it
            path = []
            temp = current_node
            while temp:
                path.append(temp.position)
                temp = temp.parent
            return path[::-1] # Return reversed path (start to end)

        closed_set.add(current_node.position)

        # Explore neighbors
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            next_x, next_y = current_node.position[0] + dx, current_node.position[1] + dy
            next_pos = (next_x, next_y)

            if not game_map.is_valid_coordinate(next_pos):
                continue # Out of bounds

            if next_pos in closed_set:
                continue # Already evaluated

            neighbor_tile = game_map.get_tile(next_pos)
            # Check if tile is passable (cannot path through occupied tiles, except the destination)
            # Check visibility for pathing? No, pathfinding assumes knowledge of map terrain.
            if neighbor_tile.unit and neighbor_tile.unit.is_alive and next_pos != end_pos:
                 continue

            terrain_key = game_map.get_terrain_key(next_pos)
            move_cost_to_neighbor = unit_move_costs(terrain_key)
            if move_cost_to_neighbor is None: # Impassable terrain for this unit
                continue

            g_cost = current_node.g_cost + move_cost_to_neighbor
            h_cost = distance(next_pos, end_pos)
            neighbor_node = Node(next_pos, parent=current_node, g_cost=g_cost, h_cost=h_cost)

            # Check if neighbor is in open_list and if this path is better
            found_in_open = False
            for i, node in enumerate(open_list):
                 if node == neighbor_node:
                      found_in_open = True
                      if neighbor_node.g_cost < node.g_cost:
                           # Update the node in the heap - tricky, easier to replace/re-add
                           open_list[i] = neighbor_node
                           heapq.heapify(open_list) # Re-sort the heap
                      break

            if not found_in_open:
                 heapq.heappush(open_list, neighbor_node)

    return None # No path found


# --- Tile Class ---
class Tile:
    def __init__(self, terrain_key):
        self.terrain_key = terrain_key
        self.terrain_info = TERRAIN_TYPES[terrain_key]
        self.name = self.terrain_info["name"]
        self.move_cost = self.terrain_info["move_cost"]
        self.defense_bonus = self.terrain_info["defense_bonus"]
        self.vision_cost = self.terrain_info["vision_cost"]
        self.symbol = self.terrain_info["symbol"]
        self.provides_income = self.terrain_info.get("income", 0)
        self.unit = None # Unit currently on the tile
        self.is_visible = False # For Fog of War (player's perspective)
        self.is_discovered = False # Has the player ever seen this tile?

        # --- Added for Highlighting ---
        self.highlight_move = False
        self.highlight_attack = False
        # -----------------------------

    def display(self, player_perspective=True):
        """How the tile should be displayed"""
        # --- Highlighting takes precedence ---
        if self.highlight_attack: return "!"
        if self.highlight_move: return "*"
        # -----------------------------------

        if player_perspective:
            if self.is_visible:
                if self.unit and self.unit.is_alive:
                    # Show unit symbol, maybe color-coded for player/enemy
                    p_symbol = self.unit.symbol.upper() if self.unit.player.id == 0 else self.unit.symbol.lower()
                    return p_symbol
                else:
                    return self.symbol # Show terrain
            elif self.is_discovered:
                return self.symbol.lower() # Show known terrain, but faded/lowercase
            else:
                return " " # Undiscovered fog
        else: # Objective view (debugging or no fog of war)
            if self.unit and self.unit.is_alive:
                 return self.unit.symbol
            else:
                 return self.symbol


# --- Unit Class ---
class Unit:
    def __init__(self, unit_id, player, unit_type, position, base_stats):
        self.id = unit_id
        self.player = player
        self.type = unit_type
        self.position = position # tuple (x, y)
        self.is_alive = True

        # Base stats from definition
        self.base_hp = base_stats["hp"]
        self.base_attack = base_stats["attack"]
        self.base_defense = base_stats["defense"]
        self.attack_range = base_stats["attack_range"]
        self.base_move_range = base_stats["move_range"]
        self.vision_range = base_stats["vision_range"]
        self.symbol = base_stats["symbol"]
        self.xp_value = base_stats["xp_value"] # XP awarded for defeating this unit
        self.ability_name = base_stats.get("ability")
        self.max_ability_cooldown = base_stats.get("ability_cooldown", 0)
        self.ability_duration = base_stats.get("ability_duration", 0) # For temp effects

        # Dynamic stats
        self.level = 1
        self.xp = 0
        self.xp_to_next_level = XP_LEVELS[2][0] if 2 in XP_LEVELS else float('inf')
        self.hp = self.base_hp
        self.attack = self.base_attack
        self.defense = self.base_defense
        self.move_range = self.base_move_range
        self.ability_cooldown_timer = 0
        self.ability_active_timer = 0 # For duration effects

        # Turn actions
        self.has_moved = False
        self.has_attacked = False
        self.has_used_ability = False

        # --- Added for Status Effects ---
        self.status_effects = {} # Format: {"effect_name": duration}
        # -------------------------------

    @property
    def max_hp(self):
        # Calculate max HP based on level bonuses
        bonus = sum(XP_LEVELS[lvl][1]["hp"] for lvl in range(2, self.level + 1) if lvl in XP_LEVELS)
        return self.base_hp + bonus

    def _update_stats_for_level(self):
        """Recalculates stats based on current level"""
        hp_bonus = sum(XP_LEVELS[lvl][1]["hp"] for lvl in range(2, self.level + 1) if lvl in XP_LEVELS)
        attack_bonus = sum(XP_LEVELS[lvl][1]["attack"] for lvl in range(2, self.level + 1) if lvl in XP_LEVELS)
        defense_bonus = sum(XP_LEVELS[lvl][1]["defense"] for lvl in range(2, self.level + 1) if lvl in XP_LEVELS)

        # Note: We only update the base for max_hp, current stats are modified directly
        # self.hp = self.max_hp # Option: fully heal on level up? Usually yes.
        self.attack = self.base_attack + attack_bonus
        self.defense = self.base_defense + defense_bonus
        # Could potentially increase move/range/vision too
        print(f"{self.player.name}'s {self.type} (ID: {self.id}) stats updated for Level {self.level}!")


    def gain_xp(self, amount):
        if self.level >= MAX_LEVEL:
            return
        self.xp += amount
        print(f"{self.player.name}'s {self.type} (ID: {self.id}) gained {amount} XP.")
        while self.xp >= self.xp_to_next_level and self.level < MAX_LEVEL:
            self.level_up()

    def level_up(self):
        if self.level >= MAX_LEVEL:
            return
        print(f"{self.player.name}'s {self.type} (ID: {self.id}) leveled up to Level {self.level + 1}!")
        self.level += 1
        # Fully heal on level up
        hp_before = self.hp
        self.hp = self.max_hp # Recalculates max HP based on new level
        print(f"  HP restored from {hp_before} to {self.hp}.")

        self._update_stats_for_level() # Apply stat bonuses

        if self.level < MAX_LEVEL:
            self.xp_to_next_level = XP_LEVELS[self.level + 1][0]
        else:
            self.xp = 0 # Or keep accumulating for score?
            self.xp_to_next_level = float('inf')
            print(f"{self.type} reached Max Level!")


    def take_damage(self, damage, attacker=None):
        if not self.is_alive: return # Can't damage dead units

        effective_defense = self.defense
        # Check for temporary defense buffs
        if self.ability_name == "Shield Wall" and self.ability_active_timer > 0:
             effective_defense += 3 # Shield Wall bonus defense - REMOVED Warrior Ability
             print(f"  (Shield Wall active! Defense: {effective_defense})")
        # --- Add Evade Bonus (Scout) ---
        if "Evade" in self.status_effects:
            effective_defense += 2 # Evade bonus defense
            print(f"  (Evade active! Defense: {effective_defense})")
        # ----------------------------------

        # Add terrain bonus
        tile = self.player.game.map.get_tile(self.position)
        terrain_bonus = tile.defense_bonus if tile else 0
        total_defense = effective_defense + terrain_bonus
        if terrain_bonus > 0:
             print(f"  (Terrain bonus: +{terrain_bonus} Defense)")

        actual_damage = max(1, damage - total_defense) # Always at least 1 damage
        self.hp -= actual_damage

        print(f"{self.player.name}'s {self.type} (ID: {self.id}) took {actual_damage} damage. HP: {self.hp}/{self.max_hp}")

        if self.hp <= 0:
            self.hp = 0
            self.is_alive = False
            print(f"{self.player.name}'s {self.type} (ID: {self.id}) has been defeated!")
            # Grant XP to the attacker if provided
            if attacker and attacker.is_alive:
                 attacker.gain_xp(self.xp_value)
            # Remove unit from tile (handled in handle_action now)
            # self.player.game.map.remove_unit(self)
        # --- Retaliation Logic ---
        elif attacker and attacker.is_alive and self.can_retaliate():
            dist = distance(self.position, attacker.position)
            if dist <= self.attack_range: # Ensure attacker is in range for retaliation (usually 1)
                print(f"  {self.type} (ID: {self.id}) retaliates!")
                time.sleep(0.3) # Small pause for clarity
                # Pass self as attacker, attacker as target
                attacker.take_damage(self.attack, attacker=self)
                self.has_attacked = True # Retaliation uses the attack action
        # -----------------------

    # --- Added Retaliation Check ---
    def can_retaliate(self):
        """Checks if the unit can perform a counter-attack."""
        return (self.is_alive and
                self.attack_range >= 1 and # Must have a melee attack at least
                not self.has_attacked and # Cannot retaliate if already attacked
                "Stun" not in self.status_effects) # Cannot retaliate if stunned
    # ----------------------------

    def can_attack(self, target_unit, game_map):
        if (not target_unit or
            not target_unit.is_alive or
            target_unit.player == self.player or
            "Stun" in self.status_effects): # Cannot attack if stunned
            return False
        dist = distance(self.position, target_unit.position)

        attack_range = self.attack_range
        # Handle Long Shot ability for Archer
        if self.ability_name == "Long Shot" and self.ability_active_timer > 0:
             attack_range += 2 # Temp range increase
             print(" (Long Shot active!)")


        if dist > attack_range:
             return False

        # Line of Sight Check (Optional but good with ranged units/terrain)
        # Basic check: No blocking terrain (Mountains/Forests) directly between
        # More complex: Check all tiles on the line using Bresenham's or similar
        if self.attack_range > 1: # Only check LoS for ranged attacks
             # Simplified LoS: check tiles directly adjacent to target in line from attacker
             dx = target_unit.position[0] - self.position[0]
             dy = target_unit.position[1] - self.position[1]
             # Check the tile before the target if not adjacent
             if dist > 1:
                  check_x, check_y = target_unit.position[0], target_unit.position[1]
                  # Determine the tile adjacent to the target along the line from the attacker
                  # This simplified check looks one step back from the target
                  if abs(dx) > abs(dy): # More horizontal line
                      check_x -= int(math.copysign(1, dx))
                  elif abs(dy) > abs(dx): # More vertical line
                      check_y -= int(math.copysign(1, dy))
                  else: # Diagonal - check the step back along the diagonal
                      check_x -= int(math.copysign(1, dx))
                      check_y -= int(math.copysign(1, dy))

                  block_pos = (check_x, check_y)
                  # Ensure the checked position isn't the attacker's own position for adjacent checks
                  if block_pos != self.position and game_map.is_valid_coordinate(block_pos):
                      blocking_tile = game_map.get_tile(block_pos)
                      # Mountains and Forests block LoS
                      if blocking_tile.terrain_key in ["M", "F"]:
                          print(" (Line of sight blocked!)")
                          return False
        return True

    def get_valid_moves(self, game_map):
        """Use BFS to find all reachable tiles within move_range."""
        if "Stun" in self.status_effects: # Cannot move if stunned
            return {self.position} # Only the current position is 'reachable'

        q = [(self.position, 0)] # (position, cost)
        visited = {self.position: 0} # pos: cost
        reachable_tiles = {self.position} # Include starting position

        move_range = self.move_range
        # Apply Charge ability effect for Cavalry
        if "Charge" in self.status_effects: # Check status effect for Charge
             move_range += 2 # Temp move bonus

        while q:
            curr_pos, curr_cost = q.pop(0)

            # Optimization: if current cost is already >= move_range, no need to check neighbors
            # But neighbors might have cost 1, so check new_cost instead
            # if curr_cost >= move_range:
            #     continue

            # Explore neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_x, next_y = curr_pos[0] + dx, curr_pos[1] + dy
                next_pos = (next_x, next_y)

                if not game_map.is_valid_coordinate(next_pos):
                    continue

                next_tile = game_map.get_tile(next_pos)
                terrain_cost = next_tile.move_cost
                new_cost = curr_cost + terrain_cost

                # Check if valid move
                if new_cost <= move_range:
                     # Cannot move into occupied tiles (unless it's the unit itself in visited)
                     if next_tile.unit and next_tile.unit.is_alive: # and next_pos != self.position:
                          continue
                     # Check if already visited with a lower or equal cost
                     if next_pos in visited and visited[next_pos] <= new_cost:
                          continue

                     visited[next_pos] = new_cost
                     reachable_tiles.add(next_pos)
                     q.append((next_pos, new_cost))

        return reachable_tiles

    def can_use_ability(self):
        return (self.ability_name and
                self.ability_cooldown_timer <= 0 and
                "Stun" not in self.status_effects) # Cannot use ability if stunned

    def use_ability(self, target=None):
        """ Target can be position or unit depending on ability """
        if not self.can_use_ability():
             print("Ability not ready or unit stunned!")
             return False

        print(f"{self.player.name}'s {self.type} (ID: {self.id}) uses {self.ability_name}!")
        self.ability_cooldown_timer = self.max_ability_cooldown
        # Assume ability takes the 'attack' action slot unless specified otherwise
        self.has_used_ability = True
        self.has_attacked = True

        # --- Implement Ability Effects ---
        # --- Updated: Bash(Warrior), Fireball(Mage), Evade(Scout) ---
        if self.ability_name == "Bash": # Warrior - Target adjacent enemy
             if isinstance(target, Unit) and target.player != self.player and target.is_alive:
                 if distance(self.position, target.position) == 1:
                     print(f"  Bashes {target.type} (ID: {target.id})!")
                     target.apply_status("Stun", 1) # Stun for 1 turn duration
                     # Maybe deal small damage too?
                     # target.take_damage(self.attack // 2, attacker=self)
                     return True
                 else:
                     print("  Target is not adjacent.")
             else:
                 print("  Invalid target for Bash (must be living adjacent enemy unit).")

        elif self.ability_name == "Long Shot": # Archer - Passive activation for next shot?
            # Let's make Long Shot a self-buff that lasts 1 turn affecting the next attack
            self.ability_active_timer = 1 + 1 # Activate for this action phase + next turn start decrement
            print("  Taking careful aim for the next shot (increased range).")
            # The range check happens in can_attack. Doesn't consume attack action itself.
            self.has_attacked = False # Activating buff doesn't count as attack
            return True # Activation successful

        elif self.ability_name == "Charge": # Cavalry - Activate for bonus move this turn
            self.apply_status("Charge", 1) # Apply Charge status for 1 turn
            print("  Preparing to charge! (Increased move range this turn)")
            # Doesn't consume attack action itself.
            self.has_attacked = False # Activating buff doesn't count as attack
            return True

        elif self.ability_name == "Fireball": # Mage - Area Effect, now applies Poison chance
            if isinstance(target, tuple) and self.player.game.map.is_valid_coordinate(target):
                 if distance(self.position, target) > self.attack_range:
                     print(f"  Target position {target} is out of range ({self.attack_range}).")
                     self.ability_cooldown_timer = 0 # Refund cooldown
                     self.has_used_ability = False # Reset flags
                     self.has_attacked = False
                     return False

                 aoe_radius = 1 # Tiles around target
                 print(f"  Casting Fireball at {target}!")
                 affected_units = []
                 for x in range(target[0] - aoe_radius, target[0] + aoe_radius + 1):
                      for y in range(target[1] - aoe_radius, target[1] + aoe_radius + 1):
                           pos = (x,y)
                           # Check distance and map bounds using Chebyshev distance for square AoE
                           # if max(abs(target[0]-pos[0]), abs(target[1]-pos[1])) <= aoe_radius and self.player.game.map.is_valid_coordinate(pos):
                           # Let's use Manhattan distance for AoE to match movement/range
                           if distance(target, pos) <= aoe_radius and self.player.game.map.is_valid_coordinate(pos):
                               unit_on_tile = self.player.game.map.get_unit_at(pos, check_visibility=False) # AI needs objective view for AoE
                               if unit_on_tile and unit_on_tile.is_alive:
                                   # AoE often hits friendlies too! Be careful.
                                   # if unit_on_tile.player != self.player: # Uncomment to avoid friendly fire
                                   affected_units.append(unit_on_tile)

                 if not affected_units: print("  ...but hit nothing.")
                 fireball_damage = self.attack + 2 # Fireball deals slightly more damage
                 poison_chance = 0.3 # 30% chance to poison

                 for hit_unit in affected_units:
                      print(f"  Hit {hit_unit.player.name}'s {hit_unit.type}!")
                      hit_unit.take_damage(fireball_damage, attacker=self) # Pass self for XP gain
                      # Apply poison chance
                      if hit_unit.is_alive and random.random() < poison_chance:
                           print(f"  {hit_unit.type} is Poisoned!")
                           hit_unit.apply_status("Poison", 3) # Poison for 3 turns
                 return True
            else:
                 print("  Invalid target position for Fireball.")

        elif self.ability_name == "Heal": # Healer
            if isinstance(target, Unit) and target.is_alive and target.player == self.player:
                 if distance(self.position, target.position) <= self.attack_range: # Heal needs range check
                    heal_amount = 10 + self.level # Healing scales slightly with level
                    actual_healed = min(heal_amount, target.max_hp - target.hp) # Cannot heal above max HP
                    target.hp += actual_healed
                    print(f"  Healed {target.type} (ID: {target.id}) for {actual_healed} HP. (Current: {target.hp}/{target.max_hp})")
                    return True
                 else:
                    print(f"  Target {target.type} is out of range.")
            else:
                 print("  Invalid target for Heal (must be living friendly unit in range).")

        # --- Added Evade (Scout) ---
        elif self.ability_name == "Evade": # Scout - Self buff
            self.apply_status("Evade", self.ability_duration) # Use status effect system
            print("  Using evasive maneuvers! (Defense increased)")
            self.has_attacked = False # Activating buff doesn't count as attack
            return True
        # -------------------------

        else:
             print("  Ability effect not implemented.")

        # If ability failed (e.g., invalid target), refund cooldown and action flags
        self.ability_cooldown_timer = 0
        self.has_used_ability = False
        self.has_attacked = False
        return False


    # --- Added Status Effect Methods ---
    def apply_status(self, effect_name, duration):
        """Applies a status effect for a given duration."""
        if effect_name not in STATUS_EFFECTS_INFO:
            print(f"Warning: Unknown status effect '{effect_name}'")
            return
        # Apply effect or refresh duration if already present
        self.status_effects[effect_name] = duration
        print(f"{self.type} (ID: {self.id}) is now affected by {effect_name} for {duration} turns.")

    def tick_status_effects(self):
        """Applies effects like DoT and decrements durations. Called at turn start."""
        if not self.is_alive: return

        effects_to_remove = []
        current_effects = list(self.status_effects.items()) # Iterate over a copy

        for effect, duration in current_effects:
            # Apply effects active at start of turn
            if effect == "Poison":
                print(f"{self.type} (ID: {self.id}) takes {POISON_DAMAGE} damage from Poison.")
                self.take_damage(POISON_DAMAGE, attacker=None) # No attacker for poison source
                if not self.is_alive: break # Stop processing if poison killed the unit

            # Decrement duration
            new_duration = duration - 1
            if new_duration <= 0:
                effects_to_remove.append(effect)
            else:
                self.status_effects[effect] = new_duration # Update duration

        # Remove expired effects
        for effect in effects_to_remove:
            if effect in self.status_effects:
                del self.status_effects[effect]
                print(f"{self.type} (ID: {self.id}) is no longer affected by {effect}.")

    # --- Modified for Stun ---
    def tick_cooldowns(self):
        """Called at the start of the player's turn. Also handles temporary effect timer"""
        if self.ability_cooldown_timer > 0:
             self.ability_cooldown_timer -= 1

        # Using status effect system for Evade/Charge now, this timer is less needed
        # Kept for Long Shot example activation
        if self.ability_active_timer > 0:
             self.ability_active_timer -= 1
             if self.ability_active_timer == 0:
                  print(f"{self.type} (ID:{self.id})'s {self.ability_name} passive effect wore off.")
                  # Reset any temporary stat changes here if needed (e.g., if Long Shot added attack)


    def reset_turn(self):
        # --- Check for Stun before resetting ---
        if "Stun" in self.status_effects:
            print(f"{self.type} (ID: {self.id}) is Stunned and cannot act!")
            # Do not reset flags if stunned, effectively skipping the turn
        else:
            self.has_moved = False
            self.has_attacked = False
            self.has_used_ability = False
        # Do NOT reset cooldowns here, handled by tick_cooldowns/tick_status_effects

    # --- Modified for Stun ---
    def can_act(self):
        if "Stun" in self.status_effects:
            return False
        return not (self.has_moved and self.has_attacked) # Simplified: Can act if move OR attack is available

    def get_possible_targets(self, players, game_map):
        """Find all enemy units this unit could potentially attack."""
        if "Stun" in self.status_effects: return [] # Cannot target if stunned

        targets = []
        # Find opponent based on self.player object reference
        opponent = None
        for p in players:
            if p != self.player:
                opponent = p
                break
        if not opponent: return [] # Should not happen in 2-player game

        for enemy_unit in opponent.get_alive_units():
             # Visibility check should happen *before* calling can_attack ideally,
             # but can_attack includes the Line of Sight check
             # We need visibility from the *attacking player's* perspective
             is_visible_to_attacker = self.player.visibility_map[enemy_unit.position[1]][enemy_unit.position[0]] == 2
             if is_visible_to_attacker and self.can_attack(enemy_unit, game_map):
                  targets.append(enemy_unit)
        return targets

    def __str__(self):
        status_list = []
        if "Stun" in self.status_effects:
            status_list.append(STATUS_EFFECTS_INFO["Stun"]["symbol"])
        else:
            if not self.has_moved: status_list.append("Mv")
            if not self.has_attacked: status_list.append("Atk")
            if self.can_use_ability() and not self.has_used_ability: status_list.append("Abl")

        level_str = f" L{self.level}" if self.level > 1 else ""
        cooldown_str = f" CD:{self.ability_cooldown_timer}" if self.ability_name and self.max_ability_cooldown > 0 and self.ability_cooldown_timer > 0 else ""
        # Show active status effects symbols
        active_effects_str = "".join(STATUS_EFFECTS_INFO.get(eff, {}).get("symbol", "") for eff in self.status_effects)


        return (f"{self.type}{level_str} (ID:{self.id} HP:{self.hp}/{self.max_hp} "
                f"Atk:{self.attack} Def:{self.defense}{active_effects_str}{cooldown_str}) "
                f"Actions: {','.join(status_list) if status_list else 'Done'}")


# --- Player Class ---
class Player:
    def __init__(self, id, name, game, is_ai=False, base_position=(0,0)):
        self.id = id # 0 for player 1, 1 for player 2
        self.name = name
        self.game = game # Reference to the main game object
        self.is_ai = is_ai
        self.units = []
        self.gold = INITIAL_GOLD
        self.next_unit_id_counter = 1 # Unique IDs within the player
        self.base_unit = None # Reference to the player's Base unit
        self._create_base(base_position)
        # Fog of War: 2D array matching map, stores visibility state
        # 0 = Undiscovered, 1 = Discovered (but not visible), 2 = Currently Visible
        self.visibility_map = [[0 for _ in range(MAP_WIDTH)] for _ in range(MAP_HEIGHT)]


    def _create_base(self, position):
         base_stats = UNIT_STATS["Base"]
         unit_id = f"{self.id}-{self.next_unit_id_counter}"
         self.next_unit_id_counter += 1
         self.base_unit = Unit(unit_id, self, "Base", position, base_stats)
         self.units.append(self.base_unit)
         if self.game.map.is_valid_coordinate(position):
             # Make sure base tile type matches if needed, or force place
             base_tile = self.game.map.get_tile(position)
             if base_tile:
                  # base_tile.terrain_key = "B" # Optional: change terrain under base
                  # base_tile.terrain_info = TERRAIN_TYPES["B"]
                  # base_tile.symbol = TERRAIN_TYPES["B"]["symbol"]
                  base_tile.unit = self.base_unit
             else: print(f"Warning: Base position {position} invalid on map.")
         else: print(f"Warning: Base position {position} out of bounds.")


    def get_next_unit_id(self):
         nid = f"{self.id}-{self.next_unit_id_counter}"
         self.next_unit_id_counter += 1
         return nid

    def add_unit(self, unit_type, position):
        if unit_type not in UNIT_STATS:
             print(f"Error: Unknown unit type {unit_type}")
             return None
        if not self.game.map.is_valid_coordinate(position):
             print(f"Error: Cannot place unit at {position}, out of bounds.")
             return None
        tile = self.game.map.get_tile(position)
        if tile.unit and tile.unit.is_alive:
             print(f"Error: Cannot place unit at {position}, tile occupied by {tile.unit.type}.")
             return None

        stats = UNIT_STATS[unit_type]
        unit_id = self.get_next_unit_id()
        unit = Unit(unit_id, self, unit_type, position, stats)
        self.units.append(unit)
        self.game.map.place_unit(unit, position)
        print(f"{self.name} placed {unit_type} (ID: {unit.id}) at {position}.")
        return unit

    def build_unit(self, unit_type):
        if not self.base_unit or not self.base_unit.is_alive:
             print(f"{self.name}: Cannot build, Base is destroyed!")
             return False
        if unit_type not in UNIT_STATS:
             print(f"{self.name}: Unknown unit type '{unit_type}'")
             return False
        if UNIT_STATS[unit_type]["cost"] == 0: # Cannot build Bases directly
             print(f"{self.name}: Cannot build {unit_type}.")
             return False

        cost = UNIT_STATS[unit_type]["cost"]
        if self.gold < cost:
             print(f"{self.name}: Cannot build {unit_type}. Need {cost} gold, have {self.gold}.")
             return False

        # Find valid adjacent spot to place unit
        spawn_pos = None
        base_pos = self.base_unit.position
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1,1), (1,-1), (-1,1), (-1,-1)]: # Check adjacent tiles
            check_pos = (base_pos[0] + dx, base_pos[1] + dy)
            if self.game.map.is_valid_coordinate(check_pos):
                 tile = self.game.map.get_tile(check_pos)
                 # Allow building on basic terrain, not mountains/forests/mines initially
                 if tile and (not tile.unit or not tile.unit.is_alive) and tile.terrain_key == "P":
                      spawn_pos = check_pos
                      break

        if spawn_pos:
             self.gold -= cost
             print(f"{self.name} spent {cost} gold.")
             new_unit = self.add_unit(unit_type, spawn_pos)
             if new_unit:
                  print(f"Successfully built {unit_type} for {cost} gold. Gold remaining: {self.gold}")
                  self.update_visibility() # Update FoW after building
                  return True
             else:
                  # Should not happen if checks above pass, but good practice
                  print(f"Error building {unit_type} despite passing checks.")
                  self.gold += cost # Refund
                  return False
        else:
             print(f"{self.name}: Cannot build {unit_type}. No valid empty space next to Base.")
             return False


    def collect_income(self):
         turn_income = GOLD_PER_TURN # Base income
         mine_income = 0
         for unit in self.get_alive_units():
             # Check if unit is on a Gold Mine tile
             tile = self.game.map.get_tile(unit.position)
             if tile and tile.provides_income > 0:
                  # Maybe require unit to 'wait' or 'garrison' on the tile?
                  # Simple version: just being on it provides income
                  mine_income += tile.provides_income
                  print(f"  +{tile.provides_income} gold from {unit.type} on Gold Mine.")

         total_income = turn_income + mine_income
         self.gold += total_income
         print(f"{self.name} gained {total_income} gold (Base: {turn_income}, Mines: {mine_income}). Total: {self.gold}")


    def update_visibility(self):
        """Recalculates the player's visibility map based on unit positions."""
        # 1. Reset currently visible tiles (but keep discovered ones)
        for y in range(MAP_HEIGHT):
            for x in range(MAP_WIDTH):
                if self.visibility_map[y][x] == 2: # If currently visible
                    self.visibility_map[y][x] = 1 # Mark as discovered but not visible now
                # Also reset visibility flag on the tile itself (for player 1 perspective)
                if self.id == 0:
                    self.game.map.tiles[y][x].is_visible = False

        # 2. Calculate new visibility using BFS from each unit
        for unit in self.get_alive_units():
            q = [(unit.position, 0)] # (position, vision_cost_spent)
            visited_for_unit = {unit.position} # Avoid cycles for this unit's vision

            # Mark unit's own tile as visible
            ux, uy = unit.position
            if 0 <= uy < MAP_HEIGHT and 0 <= ux < MAP_WIDTH: # Bounds check
                self.visibility_map[uy][ux] = 2
                if self.id == 0: # Only update tile properties for player 1's view
                    self.game.map.tiles[uy][ux].is_visible = True
                    self.game.map.tiles[uy][ux].is_discovered = True

            while q:
                curr_pos, cost_spent = q.pop(0)

                # Vision range check: <= allows seeing tile exactly AT vision range limit
                if cost_spent >= unit.vision_range:
                    continue

                # Explore neighbors (including diagonals for vision)
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1,1), (1,-1), (-1,1), (-1,-1)]:
                    next_x, next_y = curr_pos[0] + dx, curr_pos[1] + dy
                    next_pos = (next_x, next_y)

                    if not self.game.map.is_valid_coordinate(next_pos):
                        continue

                    if next_pos in visited_for_unit:
                         continue

                    next_tile = self.game.map.get_tile(next_pos)
                    vision_cost = next_tile.vision_cost # Terrain affects vision cost
                    new_cost = cost_spent + vision_cost

                    # Allow vision into tiles even if cost exceeds range, but don't spread from them
                    if new_cost <= unit.vision_range:
                         visited_for_unit.add(next_pos)
                         # Mark tile as visible (2) and discovered
                         self.visibility_map[next_y][next_x] = 2
                         # Update tile directly FOR PLAYER 1 VIEW ONLY
                         if self.id == 0:
                              self.game.map.tiles[next_y][next_x].is_visible = True
                              self.game.map.tiles[next_y][next_x].is_discovered = True

                         # Only continue spreading vision if terrain doesn't completely block?
                         # Current model allows seeing *past* blocking terrain if range permits, which is simpler.
                         # Add the neighbor to the queue to explore from it
                         q.append((next_pos, new_cost))
                    # Allow seeing the tile *at* the edge of vision range even if its cost pushes over
                    # But don't spread vision *from* it if it's over budget
                    elif cost_spent < unit.vision_range and new_cost > unit.vision_range:
                         # Can see this tile, but cannot see past it
                         if next_pos not in visited_for_unit: # Avoid marking already visited tiles again
                            visited_for_unit.add(next_pos)
                            self.visibility_map[next_y][next_x] = 2
                            if self.id == 0:
                                self.game.map.tiles[next_y][next_x].is_visible = True
                                self.game.map.tiles[next_y][next_x].is_discovered = True
                         # Do not add to q


    def get_alive_units(self):
        return [u for u in self.units if u.is_alive]

    def has_units_left(self):
        # Win condition might be destroying the Base
        # return any(u.is_alive for u in self.units)
        return self.base_unit and self.base_unit.is_alive

    def units_can_act(self):
         return any(u.can_act() for u in self.get_alive_units() if u.type != "Base") # Bases don't act

    def start_turn_updates(self):
        """Actions to perform at the very start of the player's turn."""
        print(f"\n--- {self.name}'s Turn Start ---")
        self.collect_income()
        print("Applying status effects / ticking cooldowns...")
        for unit in self.get_alive_units():
            unit.tick_status_effects() # Apply DoT, decrement status duration
            if unit.is_alive: # Unit might die from poison
                unit.tick_cooldowns() # Update ability cooldowns and temp effects like Long Shot aim
                unit.reset_turn() # Reset action flags (unless stunned)
        self.update_visibility() # Crucial: Update FoW at turn start


# --- Map Class ---
class GameMap:
    def __init__(self, width, height, terrain_layout):
        self.width = width
        self.height = height
        self.tiles = self._create_map(terrain_layout)

    def _create_map(self, terrain_layout):
        if len(terrain_layout) != self.height or any(len(row) != self.width for row in terrain_layout):
             raise ValueError("Terrain layout dimensions do not match map size.")
        tiles = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                terrain_key = terrain_layout[y][x]
                if terrain_key not in TERRAIN_TYPES:
                     raise ValueError(f"Invalid terrain key '{terrain_key}' at ({x},{y})")
                row.append(Tile(terrain_key))
            tiles.append(row)
        return tiles

    # --- Modified for Highlighting ---
    def display(self, player1_pov, highlight_move=None, highlight_attack=None): # Pass the player whose POV we are showing
        """Displays the map from the perspective of Player 1 (Human)"""

        # Reset highlights from previous display
        for y in range(self.height):
            for x in range(self.width):
                self.tiles[y][x].highlight_move = False
                self.tiles[y][x].highlight_attack = False

        # Apply new highlights (respecting FoW for player 1)
        if highlight_move:
            for pos in highlight_move:
                if self.is_valid_coordinate(pos) and player1_pov.visibility_map[pos[1]][pos[0]] > 0: # Check discovered or visible
                     self.tiles[pos[1]][pos[0]].highlight_move = True
        if highlight_attack:
            for pos in highlight_attack:
                 if self.is_valid_coordinate(pos) and player1_pov.visibility_map[pos[1]][pos[0]] == 2: # Must be currently visible to highlight attack target
                      self.tiles[pos[1]][pos[0]].highlight_attack = True # Attack highlight overrides move


        print("    " + " ".join(f"{i:<2}" for i in range(self.width))) # Column numbers
        print("  +" + "--" * self.width + "-+")
        for y in range(self.height):
            row_str = f"{y:<2}|" # Row number
            for x in range(self.width):
                # Use the tile's display method which checks visibility & highlighting
                tile = self.tiles[y][x]
                display_char = tile.display(player_perspective=True) # Always use P1 perspective for display
                row_str += " " + display_char
            row_str += " |"
            print(row_str)
        print("  +" + "--" * self.width + "-+")
        print("Legend: . = Plains, ^ = Mtn, # = Forest, G = Mine, B = Base")
        print("        * = Move Range, ! = Attack Range (lowercase = discovered fog)")
        print("Units: Player 1 (UPPERCASE), Player 2 AI (lowercase)")
        print("Status: (P)Poison (S)Stun (E)Evade (Ch)Charge (SW)ShieldWall")


    def is_valid_coordinate(self, pos):
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def get_tile(self, pos):
        if self.is_valid_coordinate(pos):
            return self.tiles[pos[1]][pos[0]]
        return None

    def get_unit_at(self, pos, check_visibility=True, asking_player=None):
        tile = self.get_tile(pos)
        if tile:
            unit = tile.unit
            if unit and unit.is_alive:
                # If checking visibility, ensure the asking_player (usually P1) can see it
                if check_visibility and asking_player:
                    if asking_player.visibility_map[pos[1]][pos[0]] == 2: # Currently visible
                        return unit
                    else:
                        return None # Unit present but not visible
                else: # If not checking visibility (e.g., for AI logic, AoE)
                    return unit
        return None


    def get_terrain_key(self, pos):
         tile = self.get_tile(pos)
         return tile.terrain_key if tile else None


    def place_unit(self, unit, pos):
        if self.is_valid_coordinate(pos):
            tile = self.get_tile(pos)
            # Should be checked before calling, but double-check
            if not tile.unit or not tile.unit.is_alive:
                tile.unit = unit
                unit.position = pos
            else:
                print(f"Error: Cannot place unit at {pos}, already occupied by {tile.unit.type}")

    def remove_unit(self, unit):
         # Find the unit on the map and remove its reference from the tile
         if self.is_valid_coordinate(unit.position):
             tile = self.get_tile(unit.position)
             if tile.unit == unit:
                  tile.unit = None
         # The unit object might still exist in the player's list until pruned,
         # but setting is_alive to False is the primary check.

    def move_unit(self, unit, new_pos):
        if self.is_valid_coordinate(unit.position):
            self.get_tile(unit.position).unit = None # Clear old tile
        self.place_unit(unit, new_pos) # Place on new tile


# --- Game Class ---
class Game:
    def __init__(self, map_layout):
        self.map = GameMap(MAP_WIDTH, MAP_HEIGHT, map_layout)
        # Player IDs: 0 = Human, 1 = AI
        # Assign base positions here
        p1_base_pos = (1, 1)
        p2_base_pos = (MAP_WIDTH - 2, MAP_HEIGHT - 2)

        # Ensure base positions match the map layout 'B' tiles if needed, or overwrite
        # For simplicity, we assume the player init places the base unit correctly
        # on whatever terrain is at the position.

        self.player1 = Player(0, "Player 1", self, is_ai=False, base_position=p1_base_pos)
        self.player2 = Player(1, "Player 2 (AI)", self, is_ai=True, base_position=p2_base_pos)
        self.players = [self.player1, self.player2]
        self.current_player_index = 0
        self.turn_number = 1
        self._setup_initial_units()
        # Calculate initial visibility AFTER units are placed
        self.player1.update_visibility()
        self.player2.update_visibility() # AI also needs initial visibility


    def _setup_initial_units(self):
        # Bases are created by Player init
        # Add some starting units near bases
        self.player1.add_unit("Warrior", (1, 2))
        self.player1.add_unit("Scout", (2, 1)) # Added Scout
        self.player2.add_unit("Warrior", (MAP_WIDTH - 2, MAP_HEIGHT - 3))
        self.player2.add_unit("Archer", (MAP_WIDTH - 3, MAP_HEIGHT - 2))

    def get_current_player(self):
        return self.players[self.current_player_index]

    def get_opponent(self):
         return self.players[1] if self.current_player_index == 0 else self.players[0]


    def next_turn(self):
        self.current_player_index = (self.current_player_index + 1) % len(self.players)
        new_player = self.get_current_player()

        if self.current_player_index == 0: # Back to Player 1
            self.turn_number += 1
            print(f"\n===== Starting Turn {self.turn_number} =====")

        # Perform start-of-turn updates for the new current player
        new_player.start_turn_updates()


    def display_game_state(self, selected_unit=None):
        clear_screen()
        current_player = self.get_current_player()
        print(f"===== Turn {self.turn_number} - {current_player.name}'s Turn =====")
        print(f"Gold: {self.player1.gold}") # Show human player's gold

        # --- Prepare highlight data if unit selected ---
        highlight_moves = None
        highlight_attacks = None
        if selected_unit and selected_unit.player == self.player1: # Only highlight for human player's selected unit
            if not selected_unit.has_moved:
                 highlight_moves = selected_unit.get_valid_moves(self.map)
            if not selected_unit.has_attacked:
                # Get potential targets respecting player 1's visibility
                targets = selected_unit.get_possible_targets(self.players, self.map)
                highlight_attacks = {t.position for t in targets}
        # ---------------------------------------------

        self.map.display(self.player1, highlight_move=highlight_moves, highlight_attack=highlight_attacks) # Always display from Player 1's POV

        print("\n--- Your Units (Player 1) ---")
        for unit in sorted(self.player1.get_alive_units(), key=lambda u: u.id): # Sort for consistent order
             # Only show info the player should know
             print(f"  {unit}") # Unit __str__ includes actions/cooldowns/status

        # Optionally show visible enemy units
        print("\n--- Visible Enemy Units ---")
        visible_enemies = 0
        # Sort by ID for consistency
        sorted_enemy_units = sorted(self.player2.get_alive_units(), key=lambda u: u.id)
        for unit in sorted_enemy_units:
             # Check Player 1's visibility map
             if self.player1.visibility_map[unit.position[1]][unit.position[0]] == 2:
                 # Show basic info, including status effects player can see
                 status_str = "".join(STATUS_EFFECTS_INFO.get(eff, {}).get("symbol", "") for eff in unit.status_effects)
                 print(f"  {unit.type} (ID:{unit.id}) at {unit.position} HP:?/{unit.max_hp} {status_str}") # Hide exact HP? Show status.
                 visible_enemies +=1
        if visible_enemies == 0: print("  None")


    def get_player_input(self):
        player = self.get_current_player()
        while True:
            print("\n" + "="*20)
            print(f"{player.name}'s Action Phase")
            print(f"Gold: {player.gold}")
            # --- Added save/load ---
            prompt = "Enter command (select [id], build [type], wait, save, load, help, quit): "
            command = input(prompt).lower().strip()

            if command == "quit":
                return ("quit", None)
            elif command == "wait":
                 # End the player's action phase for this turn
                 return ("end_turn", None)
            elif command == "help":
                 self.show_help()
                 continue
            # --- Added save/load ---
            elif command == "save":
                return ("save", None)
            elif command == "load":
                return ("load", None)
            # ----------------------

            elif command.startswith("select"):
                parts = command.split()
                if len(parts) == 2:
                    unit_id_str = parts[1]
                    selected_unit = None
                    for unit in player.get_alive_units():
                        # Match the full ID string (e.g., "0-1")
                        if unit.id == unit_id_str and unit.type != "Base": # Cannot select base for actions
                            selected_unit = unit
                            break
                    if selected_unit:
                         if not selected_unit.can_act():
                              print(f"{selected_unit.type} (ID: {unit_id_str}) has no actions left or is stunned.")
                              continue
                         return ("select", selected_unit)
                    else:
                         print(f"Invalid or unavailable unit ID: {unit_id_str}")
                else:
                    print("Invalid select command. Use: select [unit_id] (e.g., select 0-1)")

            elif command.startswith("build"):
                parts = command.split()
                if len(parts) == 2:
                     unit_type_str = parts[1].capitalize() # Allow warrior, Warrior, etc.
                     # Attempt to build the unit
                     if player.build_unit(unit_type_str):
                          # Building uses the turn? Or just gold? Assume just gold for now.
                          # If building takes the whole turn, return "end_turn"
                          # Let's allow other actions after building.
                          return ("build_success", None) # Indicate success, redraw state
                     else:
                          # Build failed, message already printed by build_unit
                          continue # Ask for input again
                else:
                     print("Invalid build command. Use: build [unit_type] (e.g., build Warrior)")

            else:
                 print("Unknown command. Type 'help' for options.")

    def show_help(self):
        print("\n--- Help ---")
        print("Turn Structure: Income -> Status Effects -> Build -> Actions -> End Turn")
        print("Commands (Action Phase):")
        print("  select [unit_id]   - Choose a unit to command (e.g., select 0-1).")
        print("  build [unit_type]  - Build a unit at your Base (e.g., build Warrior). Costs gold.")
        print("  wait               - End your action phase for this turn.")
        print("  save               - Save the current game state.")
        print("  load               - Load the last saved game (restarts current game).")
        print("  quit               - Exit the game.")
        print("\nUnit Actions (after selecting):")
        print("  move [x] [y]       - Move selected unit to (x,y) using A* pathfinding.")
        print("  attack [target_id] - Attack an enemy unit (e.g., attack 1-1).")
        print("  ability [target]   - Use special ability. Target depends on ability:")
        print("                     - Heal/Bash: ability [target_unit_id]")
        print("                     - Fireball (AoE): ability [x] [y]")
        print("                     - Self/Passive activate (Evade/Charge/LongShot): ability")
        print("  info               - Show detailed info about the selected unit.")
        print("  wait               - End selected unit's turn (uses remaining actions).")
        print("  cancel             - Deselect the unit.")
        print("Map Legend: * = Move Range, ! = Attack Range")
        print("Status Legend: (P)Poison (S)Stun (E)Evade (Ch)Charge")
        print("------------")


    def get_unit_action(self, unit):
        player = self.get_current_player()
        while True:
            print(f"\nSelected: {unit}") # __str__ shows available actions
            options = []
            can_move = not unit.has_moved and "Stun" not in unit.status_effects
            can_attack = not unit.has_attacked and "Stun" not in unit.status_effects
            can_abil = unit.can_use_ability() and not unit.has_used_ability

            if can_move: options.append("move [x] [y]")
            if can_attack: options.append("attack [target_id]")
            if can_abil:
                 options.append(f"ability ({unit.ability_name})")
            options.append("info")
            options.append("wait") # Always possible to wait out the turn
            options.append("cancel")
            print(f"Unit actions: {', '.join(options)}")

            command = input(f"Enter action for {unit.type} (ID: {unit.id}): ").lower().strip()

            if command == "cancel":
                 return ("cancel", None)
            elif command == "wait":
                 # Waiting uses up all actions for the turn
                 unit.has_moved = True
                 unit.has_attacked = True
                 unit.has_used_ability = True
                 return ("wait", None)
            elif command == "info":
                 self.show_unit_info(unit)
                 continue # Show info and re-prompt

            # --- MOVE ---
            elif command.startswith("move") and can_move:
                 parts = command.split()
                 if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                     x, y = int(parts[1]), int(parts[2])
                     target_pos = (x, y)

                     # Check if target position is within calculated valid moves
                     valid_moves = unit.get_valid_moves(self.map) # Recalculate here? Or store? Recalc is safer.
                     if target_pos in valid_moves:
                         # Check if destination is occupied *just before* moving
                         dest_tile = self.map.get_tile(target_pos)
                         if dest_tile.unit and dest_tile.unit.is_alive:
                              print(f"Cannot move to {target_pos}. Destination occupied.")
                         else:
                              return ("move", target_pos) # Return target position
                     else:
                         print(f"Cannot move to {target_pos}. Not within move range or path blocked.")
                 else:
                      print("Invalid move command. Use: move [x] [y]")

            # --- ATTACK ---
            elif command.startswith("attack") and can_attack:
                 parts = command.split()
                 if len(parts) == 2:
                     target_id_str = parts[1]
                     target_unit = None
                     opponent = self.get_opponent()
                     for u in opponent.get_alive_units():
                          if u.id == target_id_str:
                              target_unit = u
                              break

                     # Important: Check visibility for attack command from Player 1's perspective
                     if target_unit:
                          if not self.player1.visibility_map[target_unit.position[1]][target_unit.position[0]] == 2:
                               print(f"Cannot target unit {target_id_str}. Not currently visible.")
                               continue # Re-prompt

                          if unit.can_attack(target_unit, self.map):
                               return ("attack", target_unit)
                          else:
                               print(f"Cannot attack {target_unit.type} (ID: {target_id_str}). Out of range or line of sight blocked.")
                     else:
                          print(f"Invalid or non-visible enemy unit ID: {target_id_str}")
                 else:
                      print("Invalid attack command. Use: attack [target_unit_id] (e.g., attack 1-1)")

            # --- ABILITY ---
            elif command.startswith("ability") and can_abil:
                 parts = command.split()
                 target_data = None # For ability target (unit ID, position, or None)

                 # Determine required target type based on ability
                 ability_needs_target = unit.ability_name in ["Heal", "Bash", "Fireball"] # Add other targeted abilities
                 ability_needs_pos = unit.ability_name in ["Fireball"]
                 ability_needs_unit = unit.ability_name in ["Heal", "Bash"]
                 ability_is_self = unit.ability_name in ["Evade", "Charge", "Long Shot"] # Self-cast or passive activation

                 if ability_needs_target:
                     if len(parts) < 2 and not ability_is_self: # Need target unless self-cast
                          print(f"Ability '{unit.ability_name}' requires a target. Use: ability [target_id/x y]")
                          continue

                     if ability_needs_pos and len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                          target_data = (int(parts[1]), int(parts[2])) # Position tuple
                          # Add range check here? Or in use_ability? Let's do basic check here.
                          if distance(unit.position, target_data) > unit.attack_range:
                              print(f"Target position {target_data} is out of range ({unit.attack_range}).")
                              continue
                     elif ability_needs_unit and len(parts) == 2:
                          target_id_str = parts[1]
                          # Find unit (can be friendly for Heal, enemy for Bash)
                          found_target = None
                          # Check both players for the target ID
                          for p in self.players:
                              for u in p.get_alive_units():
                                   if u.id == target_id_str:
                                        found_target = u
                                        break
                              if found_target: break

                          if found_target:
                              # Visibility check if targeting enemy
                              is_visible = True
                              if found_target.player != player:
                                   if not self.player1.visibility_map[found_target.position[1]][found_target.position[0]] == 2:
                                        print(f"Cannot target unit {target_id_str}. Not currently visible.")
                                        is_visible = False
                                        continue # Re-prompt if not visible

                              # Range check for unit-targeted abilities
                              if distance(unit.position, found_target.position) > unit.attack_range:
                                   # Check specific range if needed (e.g., Bash is range 1)
                                   req_range = 1 if unit.ability_name == "Bash" else unit.attack_range
                                   if distance(unit.position, found_target.position) > req_range:
                                        print(f"Target unit {target_id_str} is out of ability range ({req_range}).")
                                        continue

                              if is_visible:
                                   target_data = found_target # Unit object
                          else:
                               print(f"Invalid target ID for ability: {target_id_str}")
                               continue
                     elif ability_is_self and len(parts) == 1:
                         target_data = None # Explicitly None for self-cast
                     else: # Mismatched parameters
                          print(f"Invalid target format for '{unit.ability_name}'. Use ID, X Y, or no target as needed.")
                          continue
                 elif ability_is_self: # Self-cast or passive activation
                     if len(parts) > 1:
                          print(f"Ability '{unit.ability_name}' does not take parameters.")
                          continue
                     target_data = None # E.g., for Evade, Charge

                 # If target format is valid (or not needed)
                 return ("ability", target_data)

            # --- Handle invalid state actions ---
            elif command.startswith("move") and not can_move:
                 print(f"{unit.type} (ID: {unit.id}) cannot move (already moved or stunned).")
            elif command.startswith("attack") and not can_attack:
                 print(f"{unit.type} (ID: {unit.id}) cannot attack (already attacked or stunned).")
            elif command.startswith("ability") and not can_abil:
                 if "Stun" in unit.status_effects:
                    print(f"{unit.type} (ID: {unit.id}) cannot use ability (stunned).")
                 elif unit.has_used_ability:
                    print(f"{unit.type} (ID: {unit.id}) has already used an ability this turn.")
                 elif not unit.can_use_ability():
                     print(f"Ability '{unit.ability_name}' is on cooldown ({unit.ability_cooldown_timer} turns).")

            else:
                 print("Unknown or invalid action. Try again.")

    def show_unit_info(self, unit):
        """Displays detailed information about a unit."""
        print(f"\n--- Unit Info: {unit.type} (ID: {unit.id}) ---")
        print(f"  Player: {unit.player.name}")
        print(f"  Level: {unit.level} (XP: {unit.xp}/{unit.xp_to_next_level})")
        print(f"  HP: {unit.hp}/{unit.max_hp}")
        print(f"  Attack: {unit.attack}")
        print(f"  Defense: {unit.defense}")
        print(f"  Move Range: {unit.move_range}")
        print(f"  Attack Range: {unit.attack_range}")
        print(f"  Vision Range: {unit.vision_range}")
        if unit.ability_name:
             cooldown_status = "Ready" if unit.ability_cooldown_timer <= 0 else f"{unit.ability_cooldown_timer} turns"
             print(f"  Ability: {unit.ability_name} (Cooldown: {cooldown_status})")
             if unit.ability_active_timer > 0: # For Long Shot aim duration
                  print(f"    Passive Effect Active: {unit.ability_active_timer -1} more turns")
        # --- Show Status Effects ---
        if unit.status_effects:
            print("  Status Effects:")
            for effect, duration in unit.status_effects.items():
                info = STATUS_EFFECTS_INFO.get(effect, {"desc": "Unknown effect"})
                print(f"    - {effect}: {duration} turns ({info['desc']})")
        # ------------------------
        print(f"  Position: {unit.position}")
        tile = self.map.get_tile(unit.position)
        if tile:
             print(f"  Terrain: {tile.name} (Move Cost: {tile.move_cost}, Def Bonus: {tile.defense_bonus})")
        actions = []
        can_move = not unit.has_moved and "Stun" not in unit.status_effects
        can_attack = not unit.has_attacked and "Stun" not in unit.status_effects
        can_abil = unit.can_use_ability() and not unit.has_used_ability

        if can_move: actions.append("Move")
        if can_attack: actions.append("Attack")
        if can_abil: actions.append("Ability")
        if "Stun" in unit.status_effects: actions = ["Stunned"]

        print(f"  Actions Left: {', '.join(actions) if actions else 'None'}")
        print("-------------------------------")


    def handle_action(self, unit, action, data):
        """Executes the chosen action. Returns True if action was successful."""
        action_taken = False
        if action == "move":
            old_pos = unit.position
            self.map.move_unit(unit, data)
            unit.has_moved = True
            print(f"{unit.player.name}'s {unit.type} (ID: {unit.id}) moved from {old_pos} to {data}.")
            action_taken = True
            # Moving reveals new area - update FoW for the player who moved
            unit.player.update_visibility()

        elif action == "attack":
            target_unit = data
            attack_power = unit.attack
             # Apply ability modifiers if active (e.g., Long Shot might reduce power?)
            print(f"{unit.player.name}'s {unit.type} (ID: {unit.id}) attacks {target_unit.player.name}'s {target_unit.type} (ID: {target_unit.id})!")
            target_unit.take_damage(attack_power, attacker=unit) # Pass attacker for XP & retaliation check
            unit.has_attacked = True # Attacking uses the attack action
            action_taken = True
            # Retaliation happens within take_damage, no need to handle here
            if not target_unit.is_alive:
                # Don't remove Base unit representation, just mark as dead/destroyed
                if target_unit.type != "Base":
                     self.map.remove_unit(target_unit) # Remove from tile
                else:
                     print(f"*** {target_unit.player.name}'s Base has been destroyed! ***")
                # Re-calculate visibility as losing a unit can reduce vision
                # Both players might need updates if FoW is strict, but P1 update is most important for display
                self.player1.update_visibility()
                if self.player2.is_ai: self.player2.update_visibility()


        elif action == "ability":
            if unit.use_ability(data): # Target data passed here, method sets flags
                 action_taken = True
                 # Update visibility if ability reveals area or affects units (e.g., AoE kills)
                 self.player1.update_visibility()
                 if self.player2.is_ai: self.player2.update_visibility()
            else:
                 # Ability use failed (e.g., invalid target), message printed in use_ability
                 action_taken = False


        elif action == "wait":
            print(f"{unit.player.name}'s {unit.type} (ID: {unit.id}) waits.")
            unit.has_moved = True
            unit.has_attacked = True
            unit.has_used_ability = True
            action_taken = True

        # After any action, check if the unit's turn is now over
        if action_taken and not unit.can_act():
             print(f"{unit.type} (ID: {unit.id}) has finished its actions.")

        return action_taken

    # --- AI Turn Enhancement needed for new units/abilities/status ---
    def perform_ai_turn(self):
        """Enhanced AI Logic"""
        player = self.get_current_player()
        opponent = self.get_opponent()
        print(f"\n--- {player.name}'s Turn ---")
        time.sleep(0.5) # Small delay

        # --- AI Decision Making ---

        # 1. Resource Management / Building
        # --- Added Scout to potential builds ---
        build_priority = ["Warrior", "Archer", "Scout", "Healer", "Mage", "Cavalry"] # Simple build order
        if player.gold >= UNIT_STATS["Scout"]["cost"]: # Minimum cost check (Scout is cheapest)
            # Simple build condition: if fewer units than opponent or below a threshold, or needs vision
            num_my_units = len([u for u in player.get_alive_units() if u.type != "Base"])
            num_opp_units = len([u for u in opponent.get_alive_units() if u.type != "Base"])
            num_scouts = len([u for u in player.get_alive_units() if u.type == "Scout"])

            # Build scout if lacking vision or heavily outnumbered, otherwise standard units
            build_scout_condition = num_scouts == 0 and num_my_units < 3
            build_unit_condition = num_my_units < num_opp_units or num_my_units < 4

            unit_to_build = None
            if build_scout_condition and player.gold >= UNIT_STATS["Scout"]["cost"]:
                unit_to_build = "Scout"
            elif build_unit_condition:
                 for unit_type in build_priority:
                     if unit_type == "Scout": continue # Already handled above
                     cost = UNIT_STATS[unit_type]["cost"]
                     if player.gold >= cost:
                          unit_to_build = unit_type
                          break # Build the first affordable priority unit

            if unit_to_build:
                print(f"AI: Considering building {unit_to_build} (Cost: {UNIT_STATS[unit_to_build]['cost']}, Gold: {player.gold})")
                time.sleep(0.5)
                if player.build_unit(unit_to_build):
                    print(f"AI: Built {unit_to_build}.")
                    time.sleep(0.5)
                    # Assume allows other actions.


        # 2. Unit Actions (Iterate through units)
        ai_units = player.get_alive_units()
        random.shuffle(ai_units) # Prevent units always acting in the same order

        for unit in ai_units:
             if unit.type == "Base" or not unit.can_act(): # Skips stunned units too
                 continue # Skip Base and units that already acted or are stunned

             print(f"\nAI: Considering action for {unit.type} (ID: {unit.id}) at {unit.position}")
             time.sleep(0.5)

             acted_this_cycle = False # Flag if unit took any action this cycle

             # --- AI Ability Usage ---
             if not unit.has_used_ability and unit.can_use_ability():
                 ability_used = False
                 # Healer AI: Heal nearby damaged allies < 70% HP
                 if unit.type == "Healer":
                      heal_target = None
                      best_heal_score = 0.7 # Heal if below 70% HP
                      # Check nearby allies
                      for friendly in player.get_alive_units():
                          if friendly.is_alive and friendly != unit and friendly.hp < friendly.max_hp:
                              if distance(unit.position, friendly.position) <= unit.attack_range: # Heal range
                                   hp_percent = friendly.hp / friendly.max_hp
                                   if hp_percent < best_heal_score:
                                        best_heal_score = hp_percent
                                        heal_target = friendly
                      if heal_target:
                          print(f"AI: {unit.type} using Heal on {heal_target.type}")
                          time.sleep(0.5)
                          if self.handle_action(unit, "ability", heal_target):
                                ability_used = True

                 # Mage AI: Fireball visible clusters or high-priority targets
                 elif unit.type == "Mage" and unit.ability_name == "Fireball":
                      best_fireball_target_pos = None
                      best_fireball_score = 1 # Min units hit to consider
                      visible_enemies = [e for e in opponent.get_alive_units() if player.visibility_map[e.position[1]][e.position[0]] == 2]
                      potential_targets = []
                      aoe_radius = 1

                      # Find all valid target positions in range
                      for tx in range(MAP_WIDTH):
                          for ty in range(MAP_HEIGHT):
                              target_pos = (tx, ty)
                              if distance(unit.position, target_pos) <= unit.attack_range:
                                  hits = 0
                                  score = 0
                                  for enemy in visible_enemies:
                                      if distance(target_pos, enemy.position) <= aoe_radius:
                                          hits += 1
                                          score += 10 - enemy.hp # Prioritize low HP targets in blast
                                  if hits >= best_fireball_score:
                                      # Prefer hitting more units, then lower HP units
                                      if hits > best_fireball_score or score > potential_targets[-1][0] if potential_targets else -1:
                                        best_fireball_score = hits
                                        best_fireball_target_pos = target_pos
                                        # Store score for tie-breaking if needed, simple approach takes first best
                                        # potential_targets.append((score, target_pos))


                      if best_fireball_target_pos:
                          print(f"AI: {unit.type} using Fireball at {best_fireball_target_pos} (hitting {best_fireball_score} units)")
                          time.sleep(0.5)
                          if self.handle_action(unit, "ability", best_fireball_target_pos):
                              ability_used = True

                 # Warrior AI: Bash adjacent high-threat/low-HP enemy if ready
                 elif unit.type == "Warrior" and unit.ability_name == "Bash":
                      bash_target = None
                      best_bash_score = 1000 # Lower is better (HP)
                      visible_enemies = [e for e in opponent.get_alive_units() if player.visibility_map[e.position[1]][e.position[0]] == 2]
                      for enemy in visible_enemies:
                            if distance(unit.position, enemy.position) == 1:
                                # Prioritize stunning low HP enemies or high threat (e.g., Mage, Healer)
                                score = enemy.hp
                                if enemy.type in ["Mage", "Healer"]: score -= 50 # Add bonus value to stunning these
                                if score < best_bash_score:
                                    best_bash_score = score
                                    bash_target = enemy
                      if bash_target:
                           print(f"AI: {unit.type} using Bash on {bash_target.type}")
                           time.sleep(0.5)
                           if self.handle_action(unit, "ability", bash_target):
                                ability_used = True

                 # Scout AI: Use Evade if enemies are nearby and ability ready
                 elif unit.type == "Scout" and unit.ability_name == "Evade":
                     enemies_nearby = False
                     for enemy in opponent.get_alive_units():
                         if player.visibility_map[enemy.position[1]][enemy.position[0]] == 2:
                            if distance(unit.position, enemy.position) <= 3: # Check if enemies close
                                enemies_nearby = True
                                break
                     if enemies_nearby:
                        print(f"AI: {unit.type} using Evade.")
                        time.sleep(0.5)
                        if self.handle_action(unit, "ability", None):
                            ability_used = True


                 if ability_used:
                     acted_this_cycle = True
                     # Check if unit's turn ended due to ability
                     if not unit.can_act(): continue


             # --- AI Attack Logic ---
             if not unit.has_attacked:
                 potential_targets = unit.get_possible_targets(self.players, self.map) # Already checks visibility via can_attack

                 if potential_targets:
                      # Targeting priority:
                      # 1. Lowest HP absolute value
                      # 2. Base > Healer > Mage > Archer > Cavalry > Warrior > Scout (simple priority list)
                      potential_targets.sort(key=lambda t: (
                          t.hp,
                          {"Base":0, "Healer":1, "Mage":2, "Archer":3, "Cavalry":4, "Warrior":5, "Scout": 6}.get(t.type, 99)
                      ))
                      target = potential_targets[0]
                      print(f"AI: {unit.type} attacking {target.type} (HP: {target.hp})")
                      time.sleep(0.5)
                      if self.handle_action(unit, "attack", target):
                          acted_this_cycle = True
                          # Check win condition immediately after attack
                          winner = self.check_win_condition()
                          if winner: return True # End AI turn early
                          # Check if unit's turn ended due to attack/retaliation
                          if not unit.can_act(): continue


             # --- AI Move Logic ---
             if not unit.has_moved:
                 # Find nearest visible enemy or opponent base
                 target_enemy_obj = None
                 min_dist = float('inf')
                 visible_enemies = [e for e in opponent.get_alive_units() if player.visibility_map[e.position[1]][e.position[0]] == 2]

                 # --- Scout Move Logic: Prioritize exploring unseen areas or spotting ---
                 if unit.type == "Scout":
                     # Very simple: move towards opponent base if nothing seen, otherwise flank/spot
                     if not visible_enemies and opponent.base_unit and opponent.base_unit.is_alive:
                        target_enemy_obj = opponent.base_unit
                        min_dist = distance(unit.position, target_enemy_obj.position)
                     elif visible_enemies:
                         # Move towards average position of enemies or nearest? Let's try nearest.
                         for enemy_unit in visible_enemies:
                             d = distance(unit.position, enemy_unit.position)
                             if d < min_dist:
                                 min_dist = d
                                 target_enemy_obj = enemy_unit
                     # If still no target, move towards center map? Or random explore?
                     if not target_enemy_obj:
                         # Simple explore: move towards center tile
                         center_pos = (MAP_WIDTH // 2, MAP_HEIGHT // 2)
                         target_enemy_obj = Node(center_pos) # Fake target node for pathing goal
                         min_dist = distance(unit.position, center_pos)

                 # --- Default Move Logic ---
                 else:
                     if visible_enemies:
                          for enemy_unit in visible_enemies:
                              d = distance(unit.position, enemy_unit.position)
                              if d < min_dist:
                                   min_dist = d
                                   target_enemy_obj = enemy_unit
                     # No visible enemies, move towards opponent's base
                     elif opponent.base_unit and opponent.base_unit.is_alive:
                          target_enemy_obj = opponent.base_unit # Move towards base if no units seen
                          min_dist = distance(unit.position, target_enemy_obj.position)

                 if target_enemy_obj:
                      target_pos = target_enemy_obj.position
                      # Find the best tile to move to: closest to target using path distance heuristic
                      best_move_pos = unit.position
                      # Prefer tiles that allow attacking the target after moving, then closest distance
                      best_path_cost = float('inf') # A* cost to reach target from move pos
                      can_attack_from_best = False

                      possible_moves = unit.get_valid_moves(self.map)

                      for move_pos in possible_moves:
                          if move_pos == unit.position: continue

                          # Check if can attack target from this move_pos
                          temp_unit_state = {"position": move_pos, "attack_range": unit.attack_range} # Simplified mock state
                          can_attack_after_move = False
                          # Check only if the primary target is a real unit
                          if isinstance(target_enemy_obj, Unit):
                              if distance(move_pos, target_pos) <= unit.attack_range:
                                   # Basic LoS check from potential move spot
                                   # TODO: Reuse can_attack's LoS logic more directly? This is simplified.
                                   line_clear = True
                                   if unit.attack_range > 1 and distance(move_pos, target_pos) > 1:
                                        dx = target_pos[0] - move_pos[0]
                                        dy = target_pos[1] - move_pos[1]
                                        check_x, check_y = target_pos[0], target_pos[1]
                                        if abs(dx) > abs(dy): check_x -= int(math.copysign(1, dx))
                                        elif abs(dy) > abs(dx): check_y -= int(math.copysign(1, dy))
                                        else:
                                             check_x -= int(math.copysign(1, dx)); check_y -= int(math.copysign(1, dy))
                                        block_pos = (check_x, check_y)
                                        if block_pos != move_pos and self.map.is_valid_coordinate(block_pos):
                                            if self.map.get_tile(block_pos).terrain_key in ["M", "F"]:
                                                line_clear = False
                                   if line_clear:
                                       can_attack_after_move = True


                          dist_from_move = distance(move_pos, target_pos) # Manhattan distance as heuristic

                          # Prioritize spots enabling attack, then closest spots
                          if can_attack_after_move:
                              if not can_attack_from_best: # First spot found that allows attack
                                   best_move_pos = move_pos
                                   best_path_cost = dist_from_move
                                   can_attack_from_best = True
                              elif dist_from_move < best_path_cost: # Better attack spot (closer)
                                   best_move_pos = move_pos
                                   best_path_cost = dist_from_move
                          elif not can_attack_from_best: # If no attack spot found yet, just find closest
                              if dist_from_move < best_path_cost:
                                   best_move_pos = move_pos
                                   best_path_cost = dist_from_move

                      if best_move_pos != unit.position:
                          print(f"AI: {unit.type} moving from {unit.position} to {best_move_pos} towards {target_pos}")
                          time.sleep(0.5)
                          if self.handle_action(unit, "move", best_move_pos):
                              acted_this_cycle = True
                              # --- AI Attack after Move ---
                              # Check if the unit *can still act* (e.g. didn't retaliate during move)
                              # and *can attack* the original target from the new position
                              if unit.can_act() and not unit.has_attacked and isinstance(target_enemy_obj, Unit):
                                  # Re-check visibility and can_attack from new position
                                  is_visible = player.visibility_map[target_enemy_obj.position[1]][target_enemy_obj.position[0]] == 2
                                  if is_visible and unit.can_attack(target_enemy_obj, self.map):
                                      print(f"AI: {unit.type} attacking {target_enemy_obj.type} after moving.")
                                      time.sleep(0.5)
                                      if self.handle_action(unit, "attack", target_enemy_obj):
                                          # Check win condition
                                          winner = self.check_win_condition()
                                          if winner: return True
                                          if not unit.can_act(): continue # End turn if attack finished it
                                      else: # Attack failed? Should be rare here
                                          pass
                              # -------------------------
                          # Check if unit's turn ended after move/attack
                          if not unit.can_act(): continue
                      else: # No better move found
                          print(f"AI: {unit.type} at {unit.position} cannot find a better position or is blocked.")
                          # If no action taken at all this cycle, wait
                          if not acted_this_cycle:
                              print(f"AI: {unit.type} waiting.")
                              self.handle_action(unit,"wait",None)
                              acted_this_cycle = True # Mark as acted

                 else: # No target enemy found (no visible units, base destroyed/unreachable?)
                     print(f"AI: {unit.type} sees no targets. Waiting.")
                     if not acted_this_cycle:
                          self.handle_action(unit,"wait",None)
                          acted_this_cycle = True

             # Final check if unit has actions left after all phases - if not, move to next unit
             if not unit.can_act(): continue

             # If unit still has actions but didn't do anything useful (e.g., moved but couldn't attack)
             if not acted_this_cycle:
                 print(f"AI: {unit.type} finished turn without optimal action. Waiting.")
                 self.handle_action(unit,"wait",None)


             winner = self.check_win_condition()
             if winner: return True # Check win condition after each unit

        print(f"--- {player.name} Turn End ---")
        return False # Game not over

    def check_win_condition(self):
        # Win condition is destroying the enemy Base
        if not self.player2.base_unit or not self.player2.base_unit.is_alive:
            return self.player1 # Player 1 wins
        if not self.player1.base_unit or not self.player1.base_unit.is_alive:
            return self.player2 # Player 2 (AI) wins
        return None

    # --- Added Save/Load Methods ---
    def save_game(self, filename=None):
        """Saves the current game state to a file using pickle."""
        if filename is None:
             timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
             filename = f"tbs_save_{timestamp}.pkl"
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
            print(f"Game saved successfully to {filename}")
            return True
        except Exception as e:
            print(f"Error saving game: {e}")
            return False

    @staticmethod
    def load_game(filename="tbs_save.pkl"):
        """Loads a game state from a file using pickle. Returns a Game object or None."""
        # Find the most recent save file if default is used
        if filename == "tbs_save.pkl":
             try:
                 save_files = [f for f in os.listdir('.') if f.startswith('tbs_save_') and f.endswith('.pkl')]
                 if save_files:
                     filename = max(save_files, key=lambda f: os.path.getmtime(f))
                     print(f"Loading most recent save: {filename}")
                 else:
                     print("No default save files found (tbs_save_*.pkl).")
                     return None
             except Exception as e:
                 print(f"Error finding save files: {e}")
                 # Fall back to trying the default literal name anyway
                 filename = "tbs_save.pkl"

        try:
            with open(filename, 'rb') as f:
                loaded_game = pickle.load(f)
            if isinstance(loaded_game, Game):
                print(f"Game loaded successfully from {filename}")
                # Need to re-establish the link from players back to the loaded game object
                # Pickle might handle this, but good to double check or reset
                for player in loaded_game.players:
                    player.game = loaded_game
                print("Verifying player links...") # Add more checks if needed
                return loaded_game
            else:
                print(f"Error loading game: File '{filename}' does not contain a valid Game object.")
                return None
        except FileNotFoundError:
            print(f"Error loading game: Save file '{filename}' not found.")
            return None
        except Exception as e:
            print(f"Error loading game from {filename}: {e}")
            return None
    # ----------------------------


    def run(self):
        game_instance = self # Start with the initial instance

        while True: # Outer loop to handle loading a new game instance
            game_over = False
            winner = None
            selected_unit = None
            load_requested = False

            # Perform initial turn setup for whoever starts (Player 1)
            # Don't call if just loaded, state should be correct
            # This requires knowing if we just loaded. Let's handle start_turn inside the loop.

            while not game_over and not load_requested:
                current_player = game_instance.get_current_player()

                # Make sure turn start updates happen for the current player
                # This might run twice on turn 1 if not handled carefully.
                # Let's move start_turn_updates to *after* next_turn call.

                if current_player.is_ai:
                    game_instance.display_game_state(selected_unit=None) # Show state before AI moves
                    input("Press Enter to begin AI turn...") # Pause before AI acts
                    game_over = game_instance.perform_ai_turn()
                    winner = game_instance.check_win_condition()
                    if winner: break
                    game_instance.next_turn() # Advances to player 1
                    # Player 1's start_turn_updates is called by next_turn now
                    selected_unit = None # Clear selection for P1 start
                    continue

                # --- Human Player Turn ---
                game_instance.display_game_state(selected_unit) # Pass selected unit for highlighting
                action_phase_over = False

                while not action_phase_over and not load_requested:
                    # Check if player can still act
                    units_can_act = current_player.units_can_act()
                    if not units_can_act and not selected_unit: # If no unit selected and none can act
                         print("\nAll your units have finished their actions for this turn.")
                         action_phase_over = True
                         break

                    # If no unit selected, get global command
                    if not selected_unit:
                        action, data = game_instance.get_player_input()

                        if action == "quit":
                            game_over = True
                            action_phase_over = True
                            print("Quitting game.")
                            break
                        elif action == "end_turn":
                            action_phase_over = True
                            break
                        elif action == "save":
                            game_instance.save_game()
                            # Continue turn after saving
                            input("Game saved. Press Enter to continue turn.")
                            game_instance.display_game_state(selected_unit) # Refresh display
                            continue
                        elif action == "load":
                            load_requested = True # Signal outer loop to reload
                            break # Exit inner loops
                        elif action == "select":
                            selected_unit = data
                            game_instance.display_game_state(selected_unit) # Show highlights
                            # Loop back to handle selected unit's action next
                        elif action == "build_success":
                            game_instance.display_game_state(selected_unit) # Redraw state after building
                            # Continue allowing actions
                        else: # Help or error, loop back
                            continue

                    # If a unit IS selected, get its action
                    if selected_unit:
                         # Double check the selected unit can still act
                         if not selected_unit.can_act():
                              print(f"{selected_unit.type} (ID: {selected_unit.id}) cannot act anymore this turn.")
                              selected_unit = None # Deselect
                              game_instance.display_game_state(selected_unit) # Update display (remove highlights)
                              continue # Go back to global commands/selection

                         action, data = game_instance.get_unit_action(selected_unit)

                         if action == "cancel":
                              selected_unit = None
                              game_instance.display_game_state(selected_unit) # Update display (remove highlights)
                              continue # Go back to global commands/selection

                         # Perform the action
                         action_successful = game_instance.handle_action(selected_unit, action, data)

                         # After action, check win condition
                         winner = game_instance.check_win_condition()
                         if winner:
                              game_over = True
                              action_phase_over = True
                              break # Exit inner and outer loops

                         if action_successful:
                              # If unit finished its turn, deselect it
                              if not selected_unit.can_act():
                                   selected_unit = None
                              # Redisplay needed AFTER potential deselection to update highlights
                              game_instance.display_game_state(selected_unit) # Show result of action
                         else:
                              # Action failed, keep unit selected and let player try again
                              print("Action failed or was invalid. Try again.")
                              # Redisplay state even on failure? Maybe not needed.
                              game_instance.display_game_state(selected_unit)


                 # --- End of Human Action Phase ---
                if game_over or load_requested: break

                # --- End of Human Player's Turn ---
                print(f"\n--- {current_player.name} ends their turn. ---")
                game_instance.next_turn() # Advances to AI, calls start_turn_updates for AI
                selected_unit = None # Clear selection

            # --- End of Inner Game Loop ---
            if winner:
                clear_screen()
                game_instance.map.display(game_instance.player1) # Show final board state
                print(f"\n===== GAME OVER =====")
                print(f"===== {winner.name} wins! =====")
                break # Exit outer loop

            if game_over and not load_requested: # Game quit manually
                print("\n--- GAME EXITED ---")
                break # Exit outer loop

            if load_requested:
                print("\nRequesting game load...")
                loaded_instance = Game.load_game() # Attempt to load
                if loaded_instance:
                    game_instance = loaded_instance # Replace current game with loaded one
                    print("Game loaded. Restarting game loop with loaded state.")
                    # Let the loop restart naturally, it will pick up the new game_instance
                else:
                    print("Load failed. Continuing current game.")
                    # Reset flag and continue the current game instance
                    load_requested = False
                    input("Press Enter to continue.")
                    # Need to potentially redisplay? The loop should handle it.

            # If loop finishes without break/load, should not happen unless error


# --- Main Execution ---
if __name__ == "__main__":
    # Define the map layout (W=Width, H=Height)
    # P=Plains, M=Mtn, F=Forest, G=Gold Mine, B=Base (Base locations set in Game init)
    map_layout_default = [
    #    0 1 2 3 4 5 6 7 8 9 0 1
        "PPPPPPPPPPPP", # 0
        "PBFPPPPPPPFP", # 1 P1 Base near (1,1)
        "PPFFPGPPPGPP", # 2
        "PPMMPPPPMMGP", # 3
        "PPPPMPMPPPPP", # 4
        "PPPPMPMPPPPP", # 5
        "PGMMGPPPPMMP", # 6
        "PPGPPPGFFGPP", # 7
        "PFPPPPPPPGBP", # 8 P2 Base near (10,8)
        "PPPPPPPPPPPP", # 9
    ]
    if len(map_layout_default) != MAP_HEIGHT or any(len(r) != MAP_WIDTH for r in map_layout_default):
        raise ValueError("Default map layout dimensions mismatch constants!")

    # --- Option to load game at start ---
    game_to_run = None
    while game_to_run is None:
        load_choice = input("Start a (N)ew game or (L)oad last save? ").upper().strip()
        if load_choice == 'L':
            game_to_run = Game.load_game()
            if not game_to_run:
                 print("Load failed. Starting new game...")
                 game_to_run = Game(map_layout_default)
        elif load_choice == 'N':
             game_to_run = Game(map_layout_default)
        else:
            print("Invalid choice. Please enter N or L.")

    # Start the game loop with the chosen game instance
    if game_to_run:
        game_to_run.run()
