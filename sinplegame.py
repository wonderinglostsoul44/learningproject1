import random
import math
import os
import heapq # For A* priority queue
import time # For AI turn delay (optional)

# --- Constants ---
MAP_WIDTH = 12
MAP_HEIGHT = 10
INITIAL_GOLD = 100
GOLD_PER_TURN = 25
BASE_STARTING_HP = 100 # Bases can be attacked

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
UNIT_STATS = {
    "Warrior": {"hp": 25, "attack": 6, "defense": 3, "attack_range": 1, "move_range": 3, "vision_range": 2, "cost": 50, "symbol": "W", "xp_value": 10, "ability": "Shield Wall", "ability_cooldown": 5, "ability_duration": 1},
    "Archer": {"hp": 15, "attack": 4, "defense": 1, "attack_range": 4, "move_range": 2, "vision_range": 4, "cost": 60, "symbol": "A", "xp_value": 12, "ability": "Long Shot", "ability_cooldown": 4},
    "Cavalry": {"hp": 30, "attack": 7, "defense": 2, "attack_range": 1, "move_range": 5, "vision_range": 3, "cost": 80, "symbol": "C", "xp_value": 15, "ability": "Charge", "ability_cooldown": 5},
    "Mage": {"hp": 12, "attack": 5, "defense": 0, "attack_range": 3, "move_range": 2, "vision_range": 3, "cost": 70, "symbol": "M", "xp_value": 15, "ability": "Fireball", "ability_cooldown": 6}, # Fireball needs specific implementation
    "Healer": {"hp": 15, "attack": 1, "defense": 1, "attack_range": 1, "move_range": 3, "vision_range": 3, "cost": 75, "symbol": "H", "xp_value": 8, "ability": "Heal", "ability_cooldown": 3}, # Heal needs specific implementation
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

    def display(self, player_perspective=True):
        """How the tile should be displayed"""
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
        self.ability_duration = base_stats.get("ability_duration", 0) # For temp effects like Shield Wall

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
        effective_defense = self.defense
        if self.ability_name == "Shield Wall" and self.ability_active_timer > 0:
             effective_defense += 3 # Shield Wall bonus defense
             print(f"  (Shield Wall active! Defense: {effective_defense})")

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
        else:
            # Retaliation? (Optional complexity)
            pass

    def can_attack(self, target_unit, game_map):
        if not target_unit or not target_unit.is_alive or target_unit.player == self.player:
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
                  if abs(dx) > abs(dy): check_x -= int(math.copysign(1, dx))
                  elif abs(dy) > abs(dx): check_y -= int(math.copysign(1, dy))
                  else: # Diagonal - check both tiles adjacent along diagonal path? simplified: check one
                       check_x -= int(math.copysign(1, dx))
                       # check_y -= int(math.copysign(1, dy)) # Can check both ways

                  block_pos = (check_x, check_y)
                  if game_map.is_valid_coordinate(block_pos):
                       blocking_tile = game_map.get_tile(block_pos)
                       # Mountains and Forests block LoS
                       if blocking_tile.terrain_key in ["M", "F"]:
                            # But allow shooting over adjacent blocking terrain
                            if distance(self.position, block_pos) > 1:
                                print(" (Line of sight blocked!)")
                                return False
        return True

    def get_valid_moves(self, game_map):
        """Use BFS to find all reachable tiles within move_range."""
        q = [(self.position, 0)] # (position, cost)
        visited = {self.position: 0} # pos: cost
        reachable_tiles = {self.position} # Include starting position

        move_range = self.move_range
        # Apply Charge ability effect for Cavalry
        if self.ability_name == "Charge" and self.ability_active_timer > 0:
            move_range += 2 # Temp move bonus


        while q:
            curr_pos, curr_cost = q.pop(0)

            if curr_cost >= move_range:
                continue

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
                     # Cannot move into occupied tiles
                     if next_tile.unit and next_tile.unit.is_alive:
                         continue
                     # Check if already visited with a lower or equal cost
                     if next_pos in visited and visited[next_pos] <= new_cost:
                         continue

                     visited[next_pos] = new_cost
                     reachable_tiles.add(next_pos)
                     q.append((next_pos, new_cost))

        return reachable_tiles

    def can_use_ability(self):
         return self.ability_name and self.ability_cooldown_timer <= 0

    def use_ability(self, target=None):
         """ Target can be position or unit depending on ability """
         if not self.can_use_ability():
             print("Ability not ready!")
             return False

         print(f"{self.player.name}'s {self.type} (ID: {self.id}) uses {self.ability_name}!")
         self.ability_cooldown_timer = self.max_ability_cooldown

         # --- Implement Ability Effects ---
         if self.ability_name == "Shield Wall": # Warrior
             self.ability_active_timer = self.ability_duration + 1 # +1 because it ticks down at turn start
             print("  Defense temporarily increased!")
         elif self.ability_name == "Long Shot": # Archer
             # This ability might be passive or require activation. Let's make it active.
             # Effect applied in can_attack. Need target validation here.
             if isinstance(target, Unit) and self.can_attack(target, self.player.game.map):
                  # Maybe add bonus damage or effect? For now, just range boost checked elsewhere
                  print("  Firing with increased range!")
                  # If it modifies attack directly:
                  # self.attack += 2 # Temporary bonus
                  # Need mechanism to remove bonus later
             else:
                  print("  Invalid target for Long Shot.")
                  self.ability_cooldown_timer = 0 # Refund cooldown
                  return False
         elif self.ability_name == "Charge": # Cavalry
             # Effect applied in get_valid_moves. Allows move then attack?
              self.ability_active_timer = self.ability_duration + 1 # Grants bonus move this turn
              print("  Preparing to charge!")
              # Maybe allows attacking after full move? Needs game rule adjustment.
         elif self.ability_name == "Fireball": # Mage - Area Effect
             if isinstance(target, tuple) and self.player.game.map.is_valid_coordinate(target):
                 aoe_radius = 1 # Tiles around target
                 print(f"  Casting Fireball at {target}!")
                 affected_units = []
                 for x in range(target[0] - aoe_radius, target[0] + aoe_radius + 1):
                      for y in range(target[1] - aoe_radius, target[1] + aoe_radius + 1):
                           pos = (x,y)
                           if distance(target, pos) <= aoe_radius and self.player.game.map.is_valid_coordinate(pos):
                                unit_on_tile = self.player.game.map.get_unit_at(pos)
                                if unit_on_tile and unit_on_tile.is_alive:
                                    # AoE often hits friendlies too! Be careful.
                                    # if unit_on_tile.player != self.player: # Uncomment to avoid friendly fire
                                    affected_units.append(unit_on_tile)

                 if not affected_units: print("  ...but hit nothing.")
                 for hit_unit in affected_units:
                      print(f"  Hit {hit_unit.player.name}'s {hit_unit.type}!")
                      fireball_damage = self.attack + 2 # Fireball deals slightly more damage
                      hit_unit.take_damage(fireball_damage, attacker=self) # Pass self for XP gain
             else:
                 print("  Invalid target position for Fireball.")
                 self.ability_cooldown_timer = 0 # Refund cooldown
                 return False
         elif self.ability_name == "Heal": # Healer
             if isinstance(target, Unit) and target.is_alive and target.player == self.player:
                 heal_amount = 10 + self.level # Healing scales slightly with level
                 target.hp = min(target.max_hp, target.hp + heal_amount)
                 print(f"  Healed {target.type} (ID: {target.id}) for {heal_amount} HP. (Current: {target.hp}/{target.max_hp})")
             else:
                 print("  Invalid target for Heal (must be living friendly unit).")
                 self.ability_cooldown_timer = 0 # Refund cooldown
                 return False
         else:
             print("  Ability effect not implemented.")
             self.ability_cooldown_timer = 0 # Refund cooldown
             return False

         self.has_used_ability = True
         # Using ability often counts as attack action
         self.has_attacked = True # Assume ability takes the 'attack' action slot
         return True

    def tick_cooldowns(self):
         """Called at the start of the player's turn."""
         if self.ability_cooldown_timer > 0:
             self.ability_cooldown_timer -= 1
         if self.ability_active_timer > 0:
              self.ability_active_timer -= 1
              if self.ability_active_timer == 0:
                   print(f"{self.type} (ID:{self.id})'s {self.ability_name} effect wore off.")
                   # Reset any temporary stat changes here if needed


    def reset_turn(self):
        self.has_moved = False
        self.has_attacked = False
        self.has_used_ability = False
        # Do NOT reset cooldowns here, handled by tick_cooldowns

    def can_act(self):
        return not (self.has_moved and self.has_attacked and self.has_used_ability)

    def get_possible_targets(self, players, game_map):
         """Find all enemy units this unit could potentially attack."""
         targets = []
         opponent = players[0] if self.player == players[1] else players[1]
         for enemy_unit in opponent.get_alive_units():
              if self.can_attack(enemy_unit, game_map):
                   targets.append(enemy_unit)
         return targets

    def __str__(self):
        status = []
        if not self.has_moved: status.append("Move")
        if not self.has_attacked: status.append("Attack")
        if self.can_use_ability() and not self.has_used_ability: status.append(f"Ability({self.ability_name})")

        level_str = f" L{self.level}" if self.level > 1 else ""
        cooldown_str = f" CD:{self.ability_cooldown_timer}" if self.ability_name and self.max_ability_cooldown > 0 else ""
        active_str = f" ACT:{self.ability_active_timer}" if self.ability_active_timer > 0 else ""

        return (f"{self.type}{level_str} (ID:{self.id} HP:{self.hp}/{self.max_hp} "
                f"Atk:{self.attack} Def:{self.defense}{active_str}{cooldown_str}) "
                f"Actions: {', '.join(status) if status else 'None'}")


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

        # 2. Calculate new visibility using BFS from each unit
        for unit in self.get_alive_units():
            q = [(unit.position, 0)] # (position, vision_cost_spent)
            visited_for_unit = {unit.position} # Avoid cycles for this unit's vision

            # Mark unit's own tile as visible
            self.visibility_map[unit.position[1]][unit.position[0]] = 2
            self.game.map.tiles[unit.position[1]][unit.position[0]].is_visible = (self.id == 0) # P1 perspective
            self.game.map.tiles[unit.position[1]][unit.position[0]].is_discovered = (self.id == 0)

            while q:
                curr_pos, cost_spent = q.pop(0)

                if cost_spent >= unit.vision_range:
                    continue

                # Explore neighbors
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1,1), (1,-1), (-1,1), (-1,-1)]: # Check diagonals too
                    next_x, next_y = curr_pos[0] + dx, curr_pos[1] + dy
                    next_pos = (next_x, next_y)

                    if not self.game.map.is_valid_coordinate(next_pos):
                        continue

                    if next_pos in visited_for_unit:
                         continue

                    next_tile = self.game.map.get_tile(next_pos)
                    vision_cost = next_tile.vision_cost # Terrain affects vision cost
                    new_cost = cost_spent + vision_cost

                    if new_cost <= unit.vision_range:
                        visited_for_unit.add(next_pos)
                        # Mark tile as visible (2) and discovered
                        self.visibility_map[next_y][next_x] = 2
                        # Update tile directly FOR PLAYER 1 VIEW ONLY
                        if self.id == 0:
                             self.game.map.tiles[next_y][next_x].is_visible = True
                             self.game.map.tiles[next_y][next_x].is_discovered = True

                        # Only continue spreading vision if terrain doesn't completely block (e.g., high mountains)
                        # For simplicity, we allow spreading vision past all terrains if within range/cost
                        q.append((next_pos, new_cost))


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
        for unit in self.get_alive_units():
             unit.tick_cooldowns() # Update ability cooldowns and durations
             unit.reset_turn() # Reset action flags
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

    def display(self, player1_pov): # Pass the player whose POV we are showing
        """Displays the map from the perspective of Player 1 (Human)"""
        print("   " + " ".join(f"{i:<2}" for i in range(self.width))) # Column numbers
        print("  +" + "--" * self.width + "-+")
        for y in range(self.height):
            row_str = f"{y:<2}|" # Row number
            for x in range(self.width):
                # Use the tile's display method which checks visibility
                tile = self.tiles[y][x]
                # If it's Player 1's turn, we use their visibility.
                # If AI's turn, we STILL show from P1's perspective for the human player
                display_char = tile.display(player_perspective=True)
                row_str += " " + display_char
            row_str += " |"
            print(row_str)
        print("  +" + "--" * self.width + "-+")
        print("Legend: . = Plains, ^ = Mtn, # = Forest, G = Mine, B = Base")
        print("        (Lower case = discovered but not visible)")
        print("Units: Player 1 (UPPERCASE), Player 2 AI (lowercase)")


    def is_valid_coordinate(self, pos):
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def get_tile(self, pos):
        if self.is_valid_coordinate(pos):
            return self.tiles[pos[1]][pos[0]]
        return None

    def get_unit_at(self, pos):
         tile = self.get_tile(pos)
         # Important: Check visibility from PLAYER 1's perspective before returning unit info
         if tile: # and tile.is_visible: # Only return unit if visible to human player
              if tile and self.tiles[pos[1]][pos[0]].is_visible: # Check P1 visibility
                  return tile.unit if tile.unit and tile.unit.is_alive else None
         return None # Tile not visible or no unit

    def get_terrain_key(self, pos):
         tile = self.get_tile(pos)
         return tile.terrain_key if tile else None


    def place_unit(self, unit, pos):
        if self.is_valid_coordinate(pos):
            tile = self.get_tile(pos)
            if not tile.unit or not tile.unit.is_alive:
                tile.unit = unit
                unit.position = pos
            else:
                print(f"Error: Cannot place unit at {pos}, already occupied by {tile.unit.type}")

    def remove_unit(self, unit):
         if self.is_valid_coordinate(unit.position):
             tile = self.get_tile(unit.position)
             if tile.unit == unit:
                 tile.unit = None

    def move_unit(self, unit, new_pos):
         if self.is_valid_coordinate(unit.position):
             self.get_tile(unit.position).unit = None # Clear old tile
         self.place_unit(unit, new_pos) # Place on new tile


# --- Game Class ---
class Game:
    def __init__(self, map_layout):
        self.map = GameMap(MAP_WIDTH, MAP_HEIGHT, map_layout)
        # Player IDs: 0 = Human, 1 = AI
        self.player1 = Player(0, "Player 1", self, is_ai=False, base_position=(1, 1))
        self.player2 = Player(1, "Player 2 (AI)", self, is_ai=True, base_position=(MAP_WIDTH - 2, MAP_HEIGHT - 2))
        self.players = [self.player1, self.player2]
        self.current_player_index = 0
        self.turn_number = 1
        self._setup_initial_units()
        self.player1.update_visibility() # Initial FoW calculation


    def _setup_initial_units(self):
        # Bases are created by Player init
        # Add some starting units
        self.player1.add_unit("Warrior", (1, 2))
        self.player1.add_unit("Archer", (2, 1))
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


    def display_game_state(self):
        clear_screen()
        current_player = self.get_current_player()
        print(f"===== Turn {self.turn_number} - {current_player.name}'s Turn =====")
        print(f"Gold: {self.player1.gold}") # Show human player's gold

        self.map.display(self.player1) # Always display from Player 1's POV

        print("\n--- Your Units (Player 1) ---")
        for unit in self.player1.get_alive_units():
            # Only show info the player should know
            tile = self.map.get_tile(unit.position)
            if tile: # Should always exist if unit is alive
                 print(f"  {unit}") # Unit __str__ includes actions/cooldowns

        # Optionally show visible enemy units
        print("\n--- Visible Enemy Units ---")
        visible_enemies = 0
        for unit in self.player2.get_alive_units():
             tile = self.map.get_tile(unit.position)
             # Check Player 1's visibility map
             if tile and self.player1.visibility_map[unit.position[1]][unit.position[0]] == 2:
                  # Show basic info, maybe not full stats unless scanned?
                  print(f"  {unit.type} (ID:{unit.id}) at {unit.position} HP: ?/?") # Hide exact HP/stats
                  visible_enemies +=1
        if visible_enemies == 0: print("  None")


    def get_player_input(self):
        player = self.get_current_player()
        while True:
            print("\n" + "="*20)
            print(f"{player.name}'s Action Phase")
            print(f"Gold: {player.gold}")
            prompt = "Enter command (select [id], build [type], wait, help, quit): "
            command = input(prompt).lower().strip()

            if command == "quit":
                return ("quit", None)
            elif command == "wait":
                 # End the player's action phase for this turn
                 return ("end_turn", None)
            elif command == "help":
                self.show_help()
                continue

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
                              print(f"{selected_unit.type} (ID: {unit_id_str}) has no actions left.")
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
        print("Turn Structure: Income -> Build -> Actions -> End Turn")
        print("Commands (Action Phase):")
        print("  select [unit_id]   - Choose a unit to command (e.g., select 0-1).")
        print("  build [unit_type]  - Build a unit at your Base (e.g., build Warrior). Costs gold.")
        print("  wait               - End your action phase for this turn.")
        print("  quit               - Exit the game.")
        print("\nUnit Actions (after selecting):")
        print("  move [x] [y]       - Move selected unit to (x,y) using A* pathfinding.")
        print("  attack [target_id] - Attack an enemy unit (e.g., attack 1-1).")
        print("  ability [target]   - Use special ability. Target depends on ability:")
        print("                     - Heal/Targeted: ability [friendly_unit_id]")
        print("                     - Fireball (AoE): ability [x] [y]")
        print("                     - Self/Passive activate: ability")
        print("  info               - Show detailed info about the selected unit.")
        print("  wait               - End selected unit's turn.")
        print("  cancel             - Deselect the unit.")
        print("------------")


    def get_unit_action(self, unit):
         player = self.get_current_player()
         while True:
            print(f"\nSelected: {unit}")
            options = []
            if not unit.has_moved: options.append("move [x] [y]")
            if not unit.has_attacked: options.append("attack [target_id]")
            if unit.can_use_ability() and not unit.has_used_ability:
                 options.append(f"ability ({unit.ability_name})")
            options.append("info")
            options.append("wait")
            options.append("cancel")
            print(f"Unit actions: {', '.join(options)}")

            command = input(f"Enter action for {unit.type} (ID: {unit.id}): ").lower().strip()

            if command == "cancel":
                return ("cancel", None)
            elif command == "wait":
                 unit.has_moved = True # Waiting uses up all actions for the turn
                 unit.has_attacked = True
                 unit.has_used_ability = True
                 return ("wait", None)
            elif command == "info":
                 self.show_unit_info(unit)
                 continue # Show info and re-prompt

            # --- MOVE ---
            elif command.startswith("move") and not unit.has_moved:
                parts = command.split()
                if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                    x, y = int(parts[1]), int(parts[2])
                    target_pos = (x, y)

                    # Use A* pathfinding
                    def unit_move_cost_func(terrain_key):
                        return TERRAIN_TYPES.get(terrain_key, {}).get("move_cost", None)

                    path = a_star_pathfinding(self.map, unit.position, target_pos, unit_move_cost_func)

                    if path:
                        # Calculate actual path cost
                        path_cost = 0
                        for i in range(len(path) - 1):
                            pos = path[i+1]
                            tile = self.map.get_tile(pos)
                            path_cost += tile.move_cost if tile else 999

                        move_range = unit.move_range
                        if unit.ability_name == "Charge" and unit.ability_active_timer > 0:
                             move_range += 2

                        if path_cost <= move_range:
                             # Check if destination is occupied *just before* moving
                             dest_tile = self.map.get_tile(target_pos)
                             if dest_tile.unit and dest_tile.unit.is_alive:
                                 print(f"Cannot move to {target_pos}. Destination occupied.")
                             else:
                                 return ("move", target_pos) # Return target position
                        else:
                            print(f"Cannot move to {target_pos}. Path found, but cost ({path_cost}) exceeds move range ({move_range}).")
                    else:
                        print(f"Cannot move to {target_pos}. No valid path found or destination invalid/occupied.")
                else:
                    print("Invalid move command. Use: move [x] [y]")

            # --- ATTACK ---
            elif command.startswith("attack") and not unit.has_attacked:
                 parts = command.split()
                 if len(parts) == 2:
                     target_id_str = parts[1]
                     target_unit = None
                     opponent = self.get_opponent()
                     for u in opponent.get_alive_units():
                         if u.id == target_id_str:
                             target_unit = u
                             break

                     # Important: Check visibility for attack command
                     if target_unit:
                          target_tile = self.map.get_tile(target_unit.position)
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
            elif command.startswith("ability") and unit.can_use_ability() and not unit.has_used_ability:
                 parts = command.split()
                 target_data = None # For ability target (unit ID, position, or None)

                 # Determine required target type based on ability
                 ability_needs_target = unit.ability_name in ["Heal", "Long Shot", "Fireball"] # Add other targeted abilities
                 ability_needs_pos = unit.ability_name in ["Fireball"]
                 ability_needs_unit = unit.ability_name in ["Heal", "Long Shot"]

                 if ability_needs_target:
                      if len(parts) < 2:
                           print(f"Ability '{unit.ability_name}' requires a target. Use: ability [target_id/x y]")
                           continue

                      if ability_needs_pos and len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                           target_data = (int(parts[1]), int(parts[2])) # Position tuple
                      elif ability_needs_unit and len(parts) == 2:
                           target_id_str = parts[1]
                           # Find unit (can be friendly for Heal, enemy for Long Shot)
                           found_target = None
                           for p in self.players:
                                for u in p.get_alive_units():
                                     if u.id == target_id_str:
                                          found_target = u
                                          break
                                if found_target: break

                           if found_target:
                                # Visibility check if targeting enemy
                                if found_target.player != player:
                                    target_tile = self.map.get_tile(found_target.position)
                                    if not self.player1.visibility_map[found_target.position[1]][found_target.position[0]] == 2:
                                         print(f"Cannot target unit {target_id_str}. Not currently visible.")
                                         continue
                                target_data = found_target # Unit object
                           else:
                                print(f"Invalid target ID for ability: {target_id_str}")
                                continue
                      else:
                           print(f"Invalid target format for '{unit.ability_name}'. Use ID or X Y as needed.")
                           continue
                 else: # Self-cast or passive activation
                      if len(parts) > 1:
                           print(f"Ability '{unit.ability_name}' does not take parameters.")
                           continue
                      target_data = None # E.g., for Shield Wall, Charge

                 # If target format is valid (or not needed)
                 return ("ability", target_data)

            # --- Handle invalid state actions ---
            elif command.startswith("move") and unit.has_moved:
                 print(f"{unit.type} (ID: {unit.id}) has already moved this turn.")
            elif command.startswith("attack") and unit.has_attacked:
                 print(f"{unit.type} (ID: {unit.id}) has already attacked this turn.")
            elif command.startswith("ability") and unit.has_used_ability:
                 print(f"{unit.type} (ID: {unit.id}) has already used an ability this turn.")
            elif command.startswith("ability") and not unit.can_use_ability():
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
             if unit.ability_active_timer > 0:
                 print(f"    Effect Active: {unit.ability_active_timer -1} more turns")
        print(f"  Position: {unit.position}")
        tile = self.map.get_tile(unit.position)
        if tile:
             print(f"  Terrain: {tile.name} (Move Cost: {tile.move_cost}, Def Bonus: {tile.defense_bonus})")
        actions = []
        if not unit.has_moved: actions.append("Move")
        if not unit.has_attacked: actions.append("Attack")
        if unit.can_use_ability() and not unit.has_used_ability: actions.append("Ability")
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
            # Moving reveals new area
            self.get_current_player().update_visibility()

        elif action == "attack":
            target_unit = data
            attack_power = unit.attack
             # Apply ability modifiers if active (e.g., Long Shot might reduce power?)
            print(f"{unit.player.name}'s {unit.type} (ID: {unit.id}) attacks {target_unit.player.name}'s {target_unit.type} (ID: {target_unit.id})!")
            target_unit.take_damage(attack_power, attacker=unit) # Pass attacker for XP
            unit.has_attacked = True
            action_taken = True
            if not target_unit.is_alive:
                # Don't remove Base unit, just mark as dead/destroyed
                if target_unit.type != "Base":
                     self.map.remove_unit(target_unit)
                else:
                     print(f"*** {target_unit.player.name}'s Base has been destroyed! ***")
                # Re-calculate visibility as losing a unit can reduce vision
                self.get_opponent().update_visibility()


        elif action == "ability":
            if unit.use_ability(data): # Target data passed here
                 unit.has_used_ability = True
                 # Most abilities count as the 'attack' action
                 unit.has_attacked = True # Assume this for now
                 action_taken = True
                 # Update visibility if ability reveals area or affects units
                 self.get_current_player().update_visibility()
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
        if not unit.can_act():
             print(f"{unit.type} (ID: {unit.id}) has finished its actions.")

        return action_taken


    def perform_ai_turn(self):
        """Enhanced AI Logic"""
        player = self.get_current_player()
        opponent = self.get_opponent()
        print(f"\n--- {player.name}'s Turn ---")
        time.sleep(0.5) # Small delay

        # --- AI Decision Making ---

        # 1. Resource Management / Building
        build_priority = ["Warrior", "Archer", "Healer", "Mage", "Cavalry"] # Simple build order
        if player.gold >= UNIT_STATS["Warrior"]["cost"]: # Minimum cost check
            # Simple build condition: if fewer units than opponent or below a threshold
            num_my_units = len(player.get_alive_units()) -1 # Exclude base
            num_opp_units = len(opponent.get_alive_units()) -1
            if num_my_units < num_opp_units or num_my_units < 4: # Example threshold
                 for unit_type in build_priority:
                      cost = UNIT_STATS[unit_type]["cost"]
                      if player.gold >= cost:
                           print(f"AI: Considering building {unit_type} (Cost: {cost}, Gold: {player.gold})")
                           time.sleep(0.5)
                           if player.build_unit(unit_type):
                                print(f"AI: Built {unit_type}.")
                                time.sleep(0.5)
                                # Building might end turn or allow other actions. Assume allows others.
                                break # Build one unit per turn for simplicity

        # 2. Unit Actions (Iterate through units)
        ai_units = player.get_alive_units()
        random.shuffle(ai_units) # Prevent units always acting in the same order

        for unit in ai_units:
             if unit.type == "Base" or not unit.can_act():
                 continue # Skip Base and units that already acted

             print(f"\nAI: Considering action for {unit.type} (ID: {unit.id}) at {unit.position}")
             time.sleep(0.5)

             acted = False # Flag if unit took any action this cycle

             # --- AI Ability Usage (Prioritize certain abilities) ---
             if not unit.has_used_ability and unit.can_use_ability():
                 # Healer AI: Heal nearby damaged allies
                 if unit.type == "Healer":
                      heal_target = None
                      best_heal_score = 0 # Prioritize lower HP %
                      for friendly in player.get_alive_units():
                           if friendly.is_alive and friendly != unit and friendly.hp < friendly.max_hp:
                                if distance(unit.position, friendly.position) <= unit.attack_range: # Heal range
                                     hp_percent = friendly.hp / friendly.max_hp
                                     if hp_percent < 0.7 and (1 - hp_percent) > best_heal_score: # Heal if below 70%
                                          best_heal_score = (1- hp_percent)
                                          heal_target = friendly
                      if heal_target:
                           print(f"AI: {unit.type} using Heal on {heal_target.type}")
                           time.sleep(0.5)
                           self.handle_action(unit, "ability", heal_target)
                           acted = True

                 # Mage AI: Use Fireball on clustered enemies? (Complex to calculate well)
                 elif unit.type == "Mage" and unit.ability_name == "Fireball":
                      # Find best spot for AoE (simplified: target first visible enemy)
                      visible_enemies = [e for e in opponent.get_alive_units() if player.visibility_map[e.position[1]][e.position[0]] == 2]
                      if visible_enemies:
                          target_pos = visible_enemies[0].position # Simplistic targeting
                          # Check range to target position
                          if distance(unit.position, target_pos) <= unit.attack_range:
                              print(f"AI: {unit.type} using Fireball near {target_pos}")
                              time.sleep(0.5)
                              self.handle_action(unit, "ability", target_pos)
                              acted = True

                 # Warrior AI: Use Shield Wall if engaging or low HP?
                 elif unit.type == "Warrior" and unit.ability_name == "Shield Wall":
                       enemies_in_range = [e for e in opponent.get_alive_units() if distance(unit.position, e.position) <= unit.attack_range + 1] # Check if enemy nearby
                       if enemies_in_range and unit.hp < unit.max_hp * 0.6: # Use if likely to be attacked and hurt
                           print(f"AI: {unit.type} using Shield Wall.")
                           time.sleep(0.5)
                           self.handle_action(unit, "ability", None)
                           acted = True

                 # Add logic for other abilities (Charge, Long Shot) if desired

             if acted and not unit.can_act(): continue # Skip other actions if ability used turn

             # --- AI Attack Logic ---
             if not unit.has_attacked:
                 potential_targets = []
                 for enemy in opponent.get_alive_units():
                     # Check visibility for AI too
                     if player.visibility_map[enemy.position[1]][enemy.position[0]] == 2:
                          if unit.can_attack(enemy, self.map):
                              potential_targets.append(enemy)

                 if potential_targets:
                     # Targeting priority:
                     # 1. Lowest HP absolute value
                     # 2. Base > Healer > Mage > Archer > Cavalry > Warrior (simple priority list)
                     potential_targets.sort(key=lambda t: (
                         t.hp,
                         {"Base":0, "Healer":1, "Mage":2, "Archer":3, "Cavalry":4, "Warrior":5}.get(t.type, 99)
                     ))
                     target = potential_targets[0]
                     print(f"AI: {unit.type} attacking {target.type} (HP: {target.hp})")
                     time.sleep(0.5)
                     self.handle_action(unit, "attack", target)
                     acted = True
                     # Check win condition immediately after attack
                     winner = self.check_win_condition()
                     if winner: return True # End AI turn early

             if acted and not unit.can_act(): continue # Skip move if attack used turn

             # --- AI Move Logic ---
             if not unit.has_moved:
                 # Find nearest visible enemy
                 target_enemy = None
                 min_dist = float('inf')
                 visible_enemies = [e for e in opponent.get_alive_units() if player.visibility_map[e.position[1]][e.position[0]] == 2]

                 if visible_enemies:
                      for enemy_unit in visible_enemies:
                         d = distance(unit.position, enemy_unit.position)
                         if d < min_dist:
                             min_dist = d
                             target_enemy = enemy_unit
                 else: # No visible enemies, move towards opponent's base?
                      if opponent.base_unit and opponent.base_unit.is_alive:
                           target_enemy = opponent.base_unit # Move towards base if no units seen
                           min_dist = distance(unit.position, target_enemy.position)

                 if target_enemy:
                     # Use A* to find path
                     def unit_move_cost_func(terrain_key):
                         return TERRAIN_TYPES.get(terrain_key, {}).get("move_cost", None)

                     # Find the best tile to move to: closest to target using A* path cost
                     best_move_pos = unit.position
                     shortest_path_to_target_from_move = min_dist # Current distance

                     possible_moves = unit.get_valid_moves(self.map)

                     for move_pos in possible_moves:
                         if move_pos == unit.position: continue

                         # Calculate distance from the potential move spot to the target
                         dist_from_move = distance(move_pos, target_enemy.position)

                         # Simple heuristic: prefer tile that gets closest Manhattan distance
                         if dist_from_move < shortest_path_to_target_from_move:
                              shortest_path_to_target_from_move = dist_from_move
                              best_move_pos = move_pos
                         # TODO: Could enhance heuristic: prefer tiles where unit can attack after moving

                     if best_move_pos != unit.position:
                         print(f"AI: {unit.type} moving from {unit.position} to {best_move_pos} towards {target_enemy.type}")
                         time.sleep(0.5)
                         self.handle_action(unit, "move", best_move_pos)
                         acted = True
                     else:
                          print(f"AI: {unit.type} at {unit.position} cannot find a better position or is blocked. Waiting.")
                          # If couldn't move closer and didn't attack/ability, mark turn as done
                          if not acted:
                               self.handle_action(unit,"wait",None) # Use up actions by waiting
                               acted = True
                 else:
                      # No enemies visible and base destroyed/unreachable? AI waits.
                      print(f"AI: {unit.type} sees no targets and cannot path to base. Waiting.")
                      if not acted:
                           self.handle_action(unit,"wait",None)
                           acted = True

             # Small delay after each unit acts
             # time.sleep(0.2)
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

    def run(self):
        game_over = False
        winner = None
        selected_unit = None

        # Initial state display
        self.player1.start_turn_updates() # Initial income/cooldowns/FoW for P1

        while not game_over:
            current_player = self.get_current_player()

            if current_player.is_ai:
                self.display_game_state() # Show state before AI moves
                input("Press Enter to begin AI turn...") # Pause before AI acts
                game_over = self.perform_ai_turn()
                winner = self.check_win_condition()
                if winner: break
                self.next_turn() # Advances to player 1, calls start_turn_updates
                selected_unit = None
                continue

            # --- Human Player Turn ---
            self.display_game_state()
            action_phase_over = False
            selected_unit = None # Reset selection at start of action phase

            while not action_phase_over:
                # Check if player can still act (either globally or with selected unit)
                 units_can_act = current_player.units_can_act()
                 if not units_can_act:
                      print("\nAll your units have finished their actions for this turn.")
                      action_phase_over = True
                      break

                 # If no unit selected, get global command (select, build, wait)
                 if not selected_unit:
                     action, data = self.get_player_input()

                     if action == "quit":
                         game_over = True
                         action_phase_over = True
                         print("Quitting game.")
                         break
                     elif action == "end_turn":
                          action_phase_over = True
                          break
                     elif action == "select":
                         selected_unit = data
                         # Don't break, loop back to handle selected unit's action next
                     elif action == "build_success":
                          self.display_game_state() # Redraw state after building
                          # Continue allowing actions
                     else: # Help or error, loop back
                          continue

                 # If a unit IS selected, get its action
                 if selected_unit:
                      # Double check the selected unit can still act
                      if not selected_unit.can_act():
                           print(f"{selected_unit.type} (ID: {selected_unit.id}) cannot act anymore this turn.")
                           selected_unit = None # Deselect
                           continue # Go back to global commands/selection

                      action, data = self.get_unit_action(selected_unit)

                      if action == "cancel":
                          selected_unit = None
                          continue # Go back to global commands/selection

                      # Perform the action
                      action_successful = self.handle_action(selected_unit, action, data)

                      # After action, check win condition
                      winner = self.check_win_condition()
                      if winner:
                          game_over = True
                          action_phase_over = True
                          break # Exit inner and outer loops

                      if action_successful:
                          self.display_game_state() # Show result of action
                          # If unit finished its turn, deselect it
                          if not selected_unit.can_act():
                               selected_unit = None
                          # Otherwise, keep unit selected (e.g., move then maybe attack)
                      else:
                          # Action failed, keep unit selected and let player try again
                           print("Action failed or was invalid. Try again.")

                      # Loop back to either get next action for same unit, or global command if deselected


            # --- End of Human Player's Turn ---
            if game_over: break

            print(f"\n--- {current_player.name} ends their turn. ---")
            self.next_turn() # Advances to AI, calls start_turn_updates for AI
            selected_unit = None # Clear selection
            # No need for input pause here, AI turn starts immediately after


        # --- Game Over ---
        clear_screen()
        self.map.display(self.player1) # Show final board state from P1 POV
        if winner:
            print(f"\n===== GAME OVER =====")
            print(f"===== {winner.name} wins! =====")
        elif game_over: # Game quit manually
            print("\n--- GAME EXITED ---")
        else: # Should not happen
             print("\n--- GAME ENDED UNEXPECTEDLY ---")


# --- Main Execution ---
if __name__ == "__main__":
    # Define the map layout (W=Width, H=Height)
    # P=Plains, M=Mtn, F=Forest, G=Gold Mine
    map_layout = [
    #   0 1 2 3 4 5 6 7 8 9 0 1
        "PPPPPPPPPPPP", # 0
        "PBFPPPPPPPFP", # 1 P1 Base at (1,1)
        "PPFFPGPPPGPP", # 2
        "PPMMPPPPMMGP", # 3
        "PPPPMPMPPPPP", # 4
        "PPPPMPMPPPPP", # 5
        "PGMMGPPPPMMP", # 6
        "PPGPPPGFFGPP", # 7
        "PFPPPPPPPGBP", # 8 P2 Base at (10,8)
        "PPPPPPPPPPPP", # 9
    ]
    if len(map_layout) != MAP_HEIGHT or any(len(r) != MAP_WIDTH for r in map_layout):
        raise ValueError("Map layout dimensions mismatch constants!")

    game = Game(map_layout)
    game.run()
