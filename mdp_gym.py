import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class CastleEscapeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CastleEscapeEnv, self).__init__()
        # Define a 7x7 grid (numbered from (0,0) to (6,6))
        self.grid_size = 7
        self.rooms = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        self.goal_room = (6, 6)  # Define the goal room
        
        # Episode control
        self.max_steps = 1000  # Truncate episode after this many steps
        self.steps = 0

        # Define health states
        self.health_states = ['Full', 'Injured', 'Critical']
        self.health_state_to_int = {'Full': 2, 'Injured': 1, 'Critical': 0}
        self.int_to_health_state = {2: 'Full', 1: 'Injured', 0: 'Critical'}

        # Define the guards with their strengths (affects combat) and keenness (affects hiding)
        self.guards = {
            'G1': {'strength': 0.8, 'keenness': 0.1},  # Guard 1
            'G2': {'strength': 0.6, 'keenness': 0.6},  # Guard 2
            'G3': {'strength': 0.3, 'keenness': 0.3},  # Guard 3
            'G4': {'strength': 0.2, 'keenness': 0.9},  # Guard 4
        }
        self.guard_names = list(self.guards.keys())

        # Rewards
        self.rewards = {
            'goal': 10000,
            'combat_win': 100,
            'combat_loss': -10,
            'defeat': -1000,
            'invalid_action': -5,
            'heal': 50,  # Reward for successfully healing (health increases at heal tile)
            'oob': -5   # Small penalty for attempting to move out of bounds
        }

        # Actions
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'FIGHT', 'HIDE', 'HEAL', 'WAIT']
        self.action_space = spaces.Discrete(len(self.actions))

        # Observation space
        obs_space_dict = {
            'player_position': spaces.Tuple((spaces.Discrete(self.grid_size), spaces.Discrete(self.grid_size))),
            'player_health': spaces.Discrete(len(self.health_states)),
            'guard_positions': spaces.Dict({
                guard: spaces.Tuple((spaces.Discrete(self.grid_size), spaces.Discrete(self.grid_size)))
                for guard in self.guards
            })
        }
        self.observation_space = spaces.Dict(obs_space_dict)

        # Set initial state
        self.reset()

    def reset(self):
        """Resets the game to the initial state"""
        #print(self.rooms[1:-1])

        # Reset step counter
        self.steps = 0
        rnd_indices = np.random.choice(range(1,len(self.rooms)-1), size=len(self.guards), replace=False)
        guard_pos = [self.rooms[i] for i in rnd_indices]
        
        # Initialize trap and heal positions (not at start, goal, or guard positions)
        available_positions = [pos for pos in self.rooms[1:-1] if pos not in guard_pos]
        # Choose two special positions (trap and heal) from available positions
        special_positions = random.sample(available_positions, 2)

        # Choose a random starting position for the player that is not the goal, not occupied by a guard, and not one of the special positions
        guard_set = set(guard_pos)
        special_set = set(special_positions)
        possible_starts = [pos for pos in self.rooms if pos != self.goal_room and pos not in guard_set and pos not in special_set]
        if not possible_starts:
            player_start = (0, 0)
        else:
            player_start = random.choice(possible_starts)

        self.current_state = {
            'player_position': player_start,
            'player_health': 'Full',
            'guard_positions': {
                guard: pos for guard, pos in zip(self.guard_names, guard_pos)
            },  # Guards in random rooms (not the goal or the starting state)
            'trap_position': special_positions[0],
            'heal_position': special_positions[1]
        }
        return self.get_observation(), 0, False, {}

    def get_observation(self):
        """Get observation with 3x3 window centered on agent"""
        player_position = self.current_state['player_position']
        px, py = player_position
        
        # Initialize 3x3 grid for observations (-1, 0, 1 offsets from player)
        # Each cell contains: guard_present (bool), is_trap (bool), is_heal (bool), is_goal (bool)
        window = {}
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                cell_x, cell_y = px + dx, py + dy
                
                # Check if cell is within bounds
                if 0 <= cell_x < self.grid_size and 0 <= cell_y < self.grid_size:
                    cell_pos = (cell_x, cell_y)
                    
                    # Check for guards in this cell
                    guards_here = []
                    for guard, gpos in self.current_state['guard_positions'].items():
                        if gpos == cell_pos:
                            guards_here.append(guard)
                    
                    window[(dx, dy)] = {
                        'guards': guards_here if guards_here else None,
                        'is_trap': cell_pos == self.current_state['trap_position'],
                        'is_heal': cell_pos == self.current_state['heal_position'],
                        'is_goal': cell_pos == self.goal_room,
                        'in_bounds': True
                    }
                else:
                    window[(dx, dy)] = {
                        'guards': None,
                        'is_trap': False,
                        'is_heal': False,
                        'is_goal': False,
                        'in_bounds': False
                    }
        
        # Determine guard in current cell
        guard_in_cell = None
        if window[(0, 0)]['guards']:
            guard_in_cell = window[(0, 0)]['guards'][0]
        
        obs = {
            'player_position': player_position,
            'player_health': self.health_state_to_int[self.current_state['player_health']],
            'guard_in_cell': guard_in_cell,
            'window': window,  # 3x3 observation window
            'at_trap': window[(0, 0)]['is_trap'],
            'at_heal': window[(0, 0)]['is_heal'],
            'at_goal': window[(0, 0)]['is_goal']
        }
        return obs

    def is_terminal(self):
        """Check if the game has reached a terminal state"""
        if self.current_state['player_position'] == self.goal_room:  # Reaching the goal means victory
            return 'goal'
        if self.current_state['player_health'] == 'Critical':  # Losing health 3 times results in defeat
            return 'defeat'
        if self.steps >= self.max_steps:
            return 'truncated'
        return False

    def move_player(self, action):
        """Move player based on the action"""
        x, y = self.current_state['player_position']
        directions = {
            'UP': (x - 1, y),
            'DOWN': (x + 1, y),
            'LEFT': (x, y - 1),
            'RIGHT': (x, y + 1)
        }

        # Calculate the intended move
        new_position = directions.get(action, self.current_state['player_position'])

        # Ensure new position is within bounds
        if 0 <= new_position[0] < self.grid_size and 0 <= new_position[1] < self.grid_size:
            # 99% chance to move as intended
            if random.random() <= 0.99:
                self.current_state['player_position'] = new_position
            else:
                # 1% chance to move to a random adjacent cell
                adjacent_positions = [
                    directions[act] for act in directions if act != action
                ]
                adjacent_positions = [
                    pos for pos in adjacent_positions
                    if 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size
                ]
                if adjacent_positions:
                    self.current_state['player_position'] = random.choice(adjacent_positions)
            return f"Moved to {self.current_state['player_position']}", 0
        else:
            return "Out of bounds!", self.rewards.get('oob', -5)

    def move_player_to_random_adjacent(self):
        """Move player to a random adjacent cell without going out of bounds"""
        x, y = self.current_state['player_position']
        directions = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

        # Filter out-of-bounds positions
        adjacent_positions = [
            pos for pos in directions
            if 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size
        ]

        # Move player to a random adjacent position
        if adjacent_positions:
            self.current_state['player_position'] = random.choice(adjacent_positions)
    
    def move_guards_random(self):
        """Move each guard randomly to an adjacent cell, ensuring no more than 2 characters share a cell"""
        for guard in self.guard_names:
            # If this guard is in the same cell as the player, don't move it
            if self.current_state['guard_positions'][guard] == self.current_state['player_position']:
                continue  # Skip this guard, it stays in place
            
            x, y = self.current_state['guard_positions'][guard]
            directions = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1), (x, y)]  # Include staying in place
            
            # Filter out-of-bounds positions
            adjacent_positions = [
                pos for pos in directions
                if 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size
            ]
            
            # Try to find a valid position (max 2 characters per cell)
            random.shuffle(adjacent_positions)
            for new_pos in adjacent_positions:
                # Count characters at the new position
                char_count = 0
                if self.current_state['player_position'] == new_pos:
                    char_count += 1
                for other_guard in self.guard_names:
                    if self.current_state['guard_positions'][other_guard] == new_pos:
                        char_count += 1
                
                # If position has less than 2 characters, move guard there
                if char_count < 2:
                    self.current_state['guard_positions'][guard] = new_pos
                    break
            # If no valid position found, guard stays in place

    def try_fight(self):
        """Player chooses to fight the guard"""
        current_position = self.current_state['player_position']
        guards_in_room = [
            guard for guard in self.guards
            if self.current_state['guard_positions'][guard] == current_position
        ]

        if guards_in_room:
            guard = guards_in_room[0]  # Choose one guard to fight
            strength = self.guards[guard]['strength']

            # Player tries to fight the guard
            if random.random() > strength:  # Successful fight
                old_pos = self.current_state['player_position']
                self.move_player_to_random_adjacent()  # Move player to a random adjacent cell after victory
                # Ensure guard doesn't end up in same cell as player after random movement
                if self.current_state['player_position'] == old_pos:
                    # Player didn't move, so guard will move away during guard movement phase
                    pass
                return f"Fought {guard} and won!", self.rewards['combat_win']
            else:  # Player loses the fight
                if self.current_state['player_health'] == 'Full':
                    self.current_state['player_health'] = 'Injured'
                elif self.current_state['player_health'] == 'Injured':
                    self.current_state['player_health'] = 'Critical'
                self.move_player_to_random_adjacent()  # Move player to a random adjacent cell after defeat
                return f"Fought {guard} and lost!", self.rewards['combat_loss']
        return "No guard to fight!", self.rewards['invalid_action']

    def try_hide(self):
        """Player attempts to hide from the guard"""
        current_position = self.current_state['player_position']
        guards_in_room = [
            guard for guard in self.guards
            if self.current_state['guard_positions'][guard] == current_position
        ]

        if guards_in_room:
            guard = guards_in_room[0]  # Choose one guard to hide from
            keenness = self.guards[guard]['keenness']

            # Player tries to hide
            if random.random() > keenness:  # Successful hide
                old_pos = self.current_state['player_position']
                self.move_player_to_random_adjacent()  # Move player to a random adjacent cell after successfully hiding
                return f"Successfully hid from {guard}!", 0
            else:
                return self.try_fight()  # Hide failed, must fight
        return "No guard to hide from!", self.rewards['invalid_action']
    
    def try_heal(self):
        """Player attempts to heal at a heal position"""
        if self.current_state['player_position'] == self.current_state['heal_position']:
            if self.current_state['player_health'] == 'Critical':
                self.current_state['player_health'] = 'Injured'
                return "Healed! Health restored to Injured.", self.rewards['heal']
            elif self.current_state['player_health'] == 'Injured':
                self.current_state['player_health'] = 'Full'
                return "Healed! Health restored to Full.", self.rewards['heal']
            else:
                # Already full health
                return "Already at full health.", 0
        return "No heal available here.", self.rewards['invalid_action']

    def play_turn(self, action):
        """Execute a single player action (string) and return (result, reward)."""
        # Normalize action name
        if isinstance(action, int):
            action = self.actions[action]

        current_position = self.current_state['player_position']
        guards_in_room = [
            guard for guard in self.guards
            if self.current_state['guard_positions'][guard] == current_position
        ]

        # If guard is present, only allow FIGHT or HIDE
        if guards_in_room and action not in ['FIGHT', 'HIDE']:
            return f"Guard {guards_in_room[0]} is in the room! You must fight or hide.", self.rewards['invalid_action']

        if action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
            return self.move_player(action)
        elif action == 'FIGHT':
            return self.try_fight()
        elif action == 'HIDE':
            return self.try_hide()
        elif action == 'HEAL':
            return self.try_heal()
        elif action == 'WAIT':
            return "Waiting...", 0
        else:
            return "Invalid action!", self.rewards['invalid_action']

    def step(self, action):
        """Performs one step in the environment"""

        ## Thisis a fix for gym environment. 
        if (isinstance(action, str)):
            action = self.actions.index(action)

        # Store previous position to detect trap entry
        previous_position = self.current_state['player_position']
        
        # Increment step counter
        self.steps += 1
        
        action_name = self.actions[action]
        result, reward = self.play_turn(action_name)
        
        # Check if player entered trap (not just standing on it)
        current_position = self.current_state['player_position']
        if (current_position == self.current_state['trap_position'] and 
            previous_position != self.current_state['trap_position']):
            # Only apply damage if entering trap, not if already on it
            if self.current_state['player_health'] == 'Full':
                self.current_state['player_health'] = 'Injured'
                result += " Stepped on trap! Lost health."
                reward -= 50
            elif self.current_state['player_health'] == 'Injured':
                self.current_state['player_health'] = 'Critical'
                result += " Stepped on trap! Lost health."
                reward -= 50
        
        # Move guards randomly after player action (but not if game is over)
        terminal_state = self.is_terminal()
        if not terminal_state:
            self.move_guards_random()
            # After guard movement, check if guard ended up with player after successful fight/hide
            # This is prevented by the move_guards_random function which limits to 2 characters per cell

        done = False
        if terminal_state == 'goal':
            done = True
            reward += self.rewards['goal']
            result += f" You've reached the goal! {self.rewards['goal']} points!"
        elif terminal_state == 'defeat':
            done = True
            reward += self.rewards['defeat']
            result += f" You've been caught! {self.rewards['defeat']} points!"

        # Truncate episode if max steps reached and not already done
        truncated = False
        if not done and self.steps >= self.max_steps:
            done = True
            truncated = True
            result += " Episode truncated: max steps reached."

        observation = self.get_observation()
        info = {'result': result, 'action': action_name, 'truncated': truncated}

        return observation, reward, done, info

    def render(self, mode='human'):
        print(f"Current state: {self.current_state}")

