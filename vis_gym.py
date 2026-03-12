import pygame
import sys
import time
import random
from mdp_gym import CastleEscapeEnv  # Import the CastleEscapeMDP class


GRID_SIZE = 7
CELL_SIZE = 100  # Each cell is 100x100 pixels
GRID_WIDTH = CELL_SIZE * GRID_SIZE  # 700 pixels for grid
CONSOLE_WIDTH = 400  # Console on the right
WIDTH = GRID_WIDTH + CONSOLE_WIDTH  # Total width: 1100
HEIGHT = GRID_WIDTH  # Height matches grid: 700

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)
YELLOW = (255, 255, 0)  # Color for the goal room

## Please add the file paths and use them to render some cool looking stuff.
IMGFILEPATH = {

}

# Setup display
screen=None

game_ended = False
action_results = []

fps = 60
sleeptime = 0.1
clock = None

# Initialize MDP game
game = CastleEscapeEnv()

def format_action_result(action, obs, reward, info=None):
    """Format action result to show only non-empty cells in window"""
    result_parts = [action]
    
    # Add environment result message if available
    if info and 'result' in info:
        env_result = info['result']
        # Shorten common messages
        if "Moved to" in env_result:
            pass  # Skip redundant movement messages
        elif "Guard" in env_result and "is in the room" in env_result:
            pass  # Skip, we show guard in nearby_info
        else:
            # Show important messages (fight results, etc.)
            result_parts.append(f"({env_result})")
    
    # Add reward if significant
    if reward != 0:
        result_parts.append(f"R:{reward}")
    
    # Check window for nearby threats/items
    window = obs.get('window', {})
    nearby_info = []
    
    # Check current cell first
    if obs.get('guard_in_cell'):
        nearby_info.append(f"Guard:{obs['guard_in_cell']}")
    if obs.get('at_trap'):
        nearby_info.append("OnTrap")
    if obs.get('at_heal'):
        nearby_info.append("OnHeal")
    if obs.get('at_goal'):
        nearby_info.append("AtGoal!")
    
    # Check adjacent cells for guards and special tiles
    for (dx, dy), cell_info in window.items():
        if dx == 0 and dy == 0:
            continue  # Skip current cell, already checked
        
        if not cell_info.get('in_bounds', False):
            continue
        
        cell_items = []
        if cell_info.get('guards'):
            guard = cell_info['guards'][0]
            cell_items.append(guard)
        if cell_info.get('is_trap'):
            cell_items.append('T')
        if cell_info.get('is_heal'):
            cell_items.append('H')
        if cell_info.get('is_goal'):
            cell_items.append('G')
        
        if cell_items:
            # Convert dx,dy to direction
            dirs = {(-1,0):'N', (1,0):'S', (0,-1):'W', (0,1):'E',
                    (-1,-1):'NW', (-1,1):'NE', (1,-1):'SW', (1,1):'SE'}
            direction = dirs.get((dx, dy), f"{dx},{dy}")
            nearby_info.append(f"{direction}:{','.join(cell_items)}")
    
    if nearby_info:
        result_parts.append(f"[{' '.join(nearby_info)}]")
    
    return " ".join(result_parts)

# Initialize Pygame
def setup(GUI=True):
    global screen
    if GUI:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Castle Escape MDP Visualization")
        # Constants

# Map room to grid cell positions
def position_to_grid(position):
    row, col = position
    return col * CELL_SIZE, row * CELL_SIZE

# Draw the grid for the rooms and shade console area
def draw_grid():
    for x in range(0, GRID_WIDTH, CELL_SIZE):
        for y in range(0, HEIGHT, CELL_SIZE):
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BLACK, rect, 1)

    # Shade console area on the right
    rect = pygame.Rect(GRID_WIDTH, 0, CONSOLE_WIDTH, HEIGHT)
    pygame.draw.rect(screen, GRAY, rect)

# Draw fog of war (3x3 visible window around player)
def draw_fog_of_war(player_position):
    px, py = player_position
    fog_surface = pygame.Surface((GRID_WIDTH, HEIGHT))
    fog_surface.set_alpha(180)  # Translucent (0=transparent, 255=opaque)
    fog_surface.fill((50, 50, 50))  # Dark gray fog
    
    # Clear the 3x3 area around the player
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            cell_x, cell_y = px + dx, py + dy
            if 0 <= cell_x < GRID_SIZE and 0 <= cell_y < GRID_SIZE:
                grid_x, grid_y = position_to_grid((cell_x, cell_y))
                # Create a transparent rectangle for visible area
                clear_rect = pygame.Rect(grid_x, grid_y, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(fog_surface, (0, 0, 0, 0), clear_rect)
                fog_surface.set_colorkey((0, 0, 0))  # Make black transparent
    
    # Instead, let's use a different approach with per-pixel alpha
    fog_surface = pygame.Surface((GRID_WIDTH, HEIGHT), pygame.SRCALPHA)
    
    # Draw fog over all cells
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            # Check if this cell is in the 3x3 window
            if abs(i - px) <= 1 and abs(j - py) <= 1:
                continue  # Skip visible cells
            
            grid_x, grid_y = position_to_grid((i, j))
            fog_rect = pygame.Rect(grid_x + 1, grid_y + 1, CELL_SIZE - 2, CELL_SIZE - 2)
            pygame.draw.rect(fog_surface, (50, 50, 50, 180), fog_rect)
    
    screen.blit(fog_surface, (0, 0))

# Draw the goal room in yellow
def draw_goal_room():
    x, y = position_to_grid(game.goal_room)
    rect = pygame.Rect(x, y, CELL_SIZE-2, CELL_SIZE-2)
    pygame.draw.rect(screen, YELLOW, rect)
    font = pygame.font.Font(None, 28)
    label = font.render('Goal', True, BLACK)
    screen.blit(label, (x + CELL_SIZE // 4 +1, y + CELL_SIZE // 3 +1))

# Draw trap position
def draw_trap(position):
    x, y = position_to_grid(position)
    # Draw orange triangle for trap
    ORANGE = (255, 165, 0)
    center_x = x + CELL_SIZE // 2
    # Triangle points: top, bottom-left, bottom-right
    points = [
        (center_x, y + 15),  # Top
        (x + 15, y + CELL_SIZE - 15),  # Bottom-left
        (x + CELL_SIZE - 15, y + CELL_SIZE - 15)  # Bottom-right
    ]
    pygame.draw.polygon(screen, ORANGE, points)
    pygame.draw.polygon(screen, BLACK, points, 3)  # Black outline

# Draw heal position
def draw_heal(position):
    x, y = position_to_grid(position)
    # Draw dark green plus for heal
    DARK_GREEN = (0, 128, 0)
    center_x = x + CELL_SIZE // 2
    center_y = y + CELL_SIZE // 2
    # Draw thicker plus sign
    pygame.draw.line(screen, DARK_GREEN, (center_x, y + 20), (center_x, y + CELL_SIZE - 20), 8)
    pygame.draw.line(screen, DARK_GREEN, (x + 20, center_y), (x + CELL_SIZE - 20, center_y), 8)

# Draw player at a given position
def draw_player(position):
    x, y = position_to_grid(position)
    center_x = x + CELL_SIZE // 2
    center_y = y + CELL_SIZE // 2
    DARK_TURQUOISE = (0, 150, 150)  # Darker turquoise for better visibility
    pygame.draw.circle(screen, DARK_TURQUOISE, (center_x, center_y), CELL_SIZE // 4)

# Draw guards at their positions
def draw_guards(guard_positions):
    BURGUNDY = (128, 0, 32)  # Deep burgundy color
    for guard, position in guard_positions.items():
        x, y = position_to_grid(position)
        rect = pygame.Rect(x + CELL_SIZE // 4, y + CELL_SIZE // 4, CELL_SIZE // 2, CELL_SIZE // 2)
        pygame.draw.rect(screen, BURGUNDY, rect)
        # Display guard number prominently
        guard_number = guard[-1]  # Extract number from "G1", "G2", etc.
        font = pygame.font.Font(None, 48)  # Larger font for prominence
        label = font.render(guard_number, True, WHITE)
        label_rect = label.get_rect(center=(x + CELL_SIZE // 2, y + CELL_SIZE // 2))
        screen.blit(label, label_rect)

# Draw player and guard together if they are in the same room
def draw_player_and_guard_together(position, guard_positions):
    DARK_TURQUOISE = (0, 150, 150)  # Darker turquoise for better visibility
    BURGUNDY = (128, 0, 32)  # Deep burgundy color
    guards_in_room = [guard for guard, pos in guard_positions.items() if pos == position]
    guards_not_in_room = [guard for guard in guard_positions if guard not in guards_in_room]
    if guards_in_room:
        x, y = position_to_grid(position)
        # Draw the player
        player_x = x + CELL_SIZE // 4
        player_y = y + CELL_SIZE // 2
        pygame.draw.circle(screen, DARK_TURQUOISE, (player_x, player_y), CELL_SIZE // 6)
        
        # Draw the guard
        guard_x = x + 3 * CELL_SIZE // 4
        guard_y = y + CELL_SIZE // 2
        pygame.draw.rect(screen, BURGUNDY, (guard_x - CELL_SIZE // 8, guard_y - CELL_SIZE // 8, CELL_SIZE // 4, CELL_SIZE // 4))
        
        # Display guard number prominently
        guard_number = guards_in_room[0][-1]  # Extract number from "G1", "G2", etc.
        font = pygame.font.Font(None, 32)
        label = font.render(guard_number, True, WHITE)
        label_rect = label.get_rect(center=(guard_x, guard_y))
        screen.blit(label, label_rect)

    for guard in guards_not_in_room:
        draw_guards({guard: guard_positions[guard]})

# Draw player health status
def draw_health(health):
    font = pygame.font.Font(None, 36)
    health_text = f"Health: {health}"
    health_surface = font.render(health_text, True, BLUE)
    screen.blit(health_surface, (10, HEIGHT - 40))


def draw_legend(console_x, y_offset):
    """Draw a small legend in the console area for trap/heal/guard/player."""
    font_info = pygame.font.Font(None, 20)
    # Trap icon (small triangle drawn inside the console area)
    ORANGE = (255, 165, 0)
    tri_x = console_x + 10
    tri_y = y_offset + 6
    tri_points = [
        (tri_x + 8, tri_y),
        (tri_x, tri_y + 18),
        (tri_x + 16, tri_y + 18)
    ]
    pygame.draw.polygon(screen, ORANGE, tri_points)
    pygame.draw.polygon(screen, BLACK, tri_points, 1)
    trap_label = font_info.render("Trap: Orange triangle", True, BLACK)
    screen.blit(trap_label, (console_x + 36, y_offset))
    y_offset += 26

    # Heal icon (small plus)
    DARK_GREEN = (0, 128, 0)
    h_x = console_x + 10
    h_y = y_offset + 6
    pygame.draw.line(screen, DARK_GREEN, (h_x + 8, h_y), (h_x + 8, h_y + 18), 3)
    pygame.draw.line(screen, DARK_GREEN, (h_x, h_y + 9), (h_x + 16, h_y + 9), 3)
    heal_label = font_info.render("Heal: Dark green +", True, BLACK)
    screen.blit(heal_label, (console_x + 36, y_offset))
    y_offset += 26

    # Guard icon (small burgundy square)
    BURGUNDY = (128, 0, 32)
    g_x = console_x + 10
    g_y = y_offset + 6
    pygame.draw.rect(screen, BURGUNDY, (g_x, g_y, 16, 16))
    guard_label = font_info.render("Guard: Burgundy square", True, BLACK)
    screen.blit(guard_label, (console_x + 36, y_offset))
    y_offset += 26

    # Player icon (small turquoise circle)
    DARK_TURQUOISE = (0, 150, 150)
    p_x = console_x + 10
    p_y = y_offset + 10
    pygame.draw.circle(screen, DARK_TURQUOISE, (p_x + 8, p_y), 8)
    player_label = font_info.render("Player: Turquoise circle", True, BLACK)
    screen.blit(player_label, (console_x + 36, y_offset))
    y_offset += 26

    return y_offset

# Display victory or defeat message
def display_end_message(message):
    font = pygame.font.Font(None, 100)
    text_surface = font.render(message, True, DARK_GRAY)
    text_rect = text_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(text_surface, text_rect)

# Main loop
def main():
    global game_ended, action_results
    clock = pygame.time.Clock()
    running = True
    end_message = ""

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                # Restart key: allow resetting the environment after terminal
                if event.key == pygame.K_r:
                    obs, reward, done, info = game.reset()
                    action_results.clear()
                    game_ended = False
                    end_message = ""
                    continue
                if event.key == pygame.K_w and not game_ended:
                    action = "UP"
                    obs, reward, done, info = game.step(action)
                    action_results.append(format_action_result(action, obs, reward, info))
                    if len(action_results) > 10:
                        action_results.pop(0)
                if event.key == pygame.K_s and not game_ended:
                    action = "DOWN"
                    obs, reward, done, info = game.step(action)
                    action_results.append(format_action_result(action, obs, reward, info))
                    if len(action_results) > 10:
                        action_results.pop(0)
                if event.key == pygame.K_a and not game_ended:
                    action = "LEFT"
                    obs, reward, done, info = game.step(action)
                    action_results.append(format_action_result(action, obs, reward, info))
                    if len(action_results) > 10:
                        action_results.pop(0)
                if event.key == pygame.K_d and not game_ended:
                    action = "RIGHT"
                    obs, reward, done, info = game.step(action)
                    action_results.append(format_action_result(action, obs, reward, info))
                    if len(action_results) > 10:
                        action_results.pop(0)
                if event.key == pygame.K_f and not game_ended:
                    action = "FIGHT"
                    obs, reward, done, info = game.step(action)
                    action_results.append(format_action_result(action, obs, reward, info))
                    if len(action_results) > 10:
                        action_results.pop(0)
                if event.key == pygame.K_h and not game_ended:
                    action = "HIDE"
                    obs, reward, done, info = game.step(action)
                    action_results.append(format_action_result(action, obs, reward, info))
                    if len(action_results) > 10:
                        action_results.pop(0)
                if event.key == pygame.K_e and not game_ended:
                    action = "HEAL"
                    obs, reward, done, info = game.step(action)
                    action_results.append(format_action_result(action, obs, reward, info))
                    if len(action_results) > 10:
                        action_results.pop(0)
                if event.key == pygame.K_SPACE and not game_ended:
                    action = "WAIT"
                    obs, reward, done, info = game.step(action)
                    action_results.append(format_action_result(action, obs, reward, info))
                    if len(action_results) > 10:
                        action_results.pop(0)
        screen.fill(WHITE)
        draw_grid()

        # Draw the goal room
        draw_goal_room()
        
        # Draw trap and heal positions
        if hasattr(game.current_state, '__getitem__'):
            if 'trap_position' in game.current_state:
                draw_trap(game.current_state['trap_position'])
            if 'heal_position' in game.current_state:
                draw_heal(game.current_state['heal_position'])

        # Check if player and a guard are in the same room and draw them together
        if game.current_state['player_position'] in game.current_state['guard_positions'].values():
            draw_player_and_guard_together(game.current_state['player_position'], game.current_state['guard_positions'])
        else:
            # Draw the player and guards in separate positions
            draw_player(game.current_state['player_position'])
            draw_guards(game.current_state['guard_positions'])
        
        # Draw fog of war (3x3 visible window)
        draw_fog_of_war(game.current_state['player_position'])

        # Check for terminal state; if terminal, show message and wait for restart (R)
        term = game.is_terminal()
        if term == 'goal':
            game_ended = True
            end_message = "Victory! Press R to restart."
        elif term == 'defeat':
            game_ended = True
            end_message = "Defeat! Press R to restart."
        elif term == 'truncated':
            game_ended = True
            end_message = "Truncated! Press R to restart."

        if game_ended:
            display_end_message(end_message)

        # Display console on the right side
        console_x = GRID_WIDTH + 10
        font = pygame.font.Font(None, 32)
        console_surface = font.render("Game State", True, BLUE)
        screen.blit(console_surface, (console_x, 10))
        
        # Get current observation
        current_obs = game.get_observation()
        
        # Display game info as bullet list
        font_info = pygame.font.Font(None, 22)
        y_offset = 50

        guard_in_cell = current_obs.get('guard_in_cell', None)
        guard_text = guard_in_cell if guard_in_cell else "None"

        truncated_flag = 'Yes' if game.steps >= game.max_steps else 'No'
        info_lines = [
            f"• Health: {game.current_state['player_health']}",
            f"• Position: {game.current_state['player_position']}",
            f"• Start: {game.current_state.get('player_position', 'Unknown')}",
            f"• Steps: {game.steps}/{game.max_steps}",
            f"• Truncated: {truncated_flag}",
            f"• Guard Here: {guard_text}",
            f"• At Trap: {'Yes' if current_obs.get('at_trap', False) else 'No'}",
            f"• At Heal: {'Yes' if current_obs.get('at_heal', False) else 'No'}",
        ]
        
        for line in info_lines:
            info_surface = font_info.render(line, True, BLACK)
            screen.blit(info_surface, (console_x, y_offset))
            y_offset += 28
        
        # Section divider
        y_offset += 10
        pygame.draw.line(screen, BLACK, (console_x, y_offset), (console_x + CONSOLE_WIDTH - 20, y_offset), 2)
        y_offset += 15

        # Draw legend under the info block
        y_offset = draw_legend(console_x, y_offset)
        
        # Recent actions header
        font_header = pygame.font.Font(None, 28)
        actions_header = font_header.render("Recent Actions:", True, BLUE)
        screen.blit(actions_header, (console_x, y_offset))
        y_offset += 35
        
        # Print the latest 5 actions with word wrapping
        font_action = pygame.font.Font(None, 18)
        for result in action_results:
            if result:
                # Word wrap for long text
                words = result.split(' ')
                line = ""
                for word in words:
                    test_line = line + word + " "
                    # Check if line width exceeds console width
                    if font_action.size(test_line)[0] > CONSOLE_WIDTH - 25:
                        # Render current line
                        result_surface = font_action.render(line.strip(), True, BLACK)
                        screen.blit(result_surface, (console_x, y_offset))
                        y_offset += 22
                        line = word + " "
                    else:
                        line = test_line
                
                # Render remaining text
                if line.strip():
                    result_surface = font_action.render(line.strip(), True, BLACK)
                    screen.blit(result_surface, (console_x, y_offset))
                    y_offset += 22
                
                # Add spacing between actions
                y_offset += 5

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()

def refresh(obs, reward, done, info, delay=0.1):
    global fps, sleeptime, game_ended, clock, action_results, game
    
    try:
        action = info['action']
    except:
        action = "None"

    # Use the same formatting function
    result = format_action_result(action, obs, reward, info)

    # Add to action results and keep only last 10
    action_results.append(result)
    if len(action_results) > 10:
        action_results.pop(0)

    fps = 60
    clock = pygame.time.Clock()
    screen.fill(WHITE)
    draw_grid()

    # Draw the goal room
    draw_goal_room()
    
    # Draw trap and heal positions (from game state, not obs)
    if hasattr(game.current_state, '__getitem__'):
        if 'trap_position' in game.current_state:
            draw_trap(game.current_state['trap_position'])
        if 'heal_position' in game.current_state:
            draw_heal(game.current_state['heal_position'])

    # Check if player and a guard are in the same room and draw them together
    if game.current_state['player_position'] in game.current_state['guard_positions'].values():
        draw_player_and_guard_together(game.current_state['player_position'], game.current_state['guard_positions'])
    else:
        # Draw the player and guards in separate positions
        draw_player(game.current_state['player_position'])
        draw_guards(game.current_state['guard_positions'])
    
    # Draw fog of war (3x3 visible window)
    draw_fog_of_war(game.current_state['player_position'])

    if game.is_terminal() == 'goal':
        game_ended = True
        end_message = "Victory!"
    elif game.is_terminal() == 'defeat':
        game_ended = True
        end_message = "Defeat!"

    if game_ended:
        display_end_message(end_message)
        game_ended = False

    # Display console on the right side
    console_x = GRID_WIDTH + 10
    font = pygame.font.Font(None, 32)
    console_surface = font.render("Game State", True, BLUE)
    screen.blit(console_surface, (console_x, 10))
    
    # Display game info as bullet list
    font_info = pygame.font.Font(None, 22)
    y_offset = 50
    guard_in_cell = obs.get('guard_in_cell', None)
    guard_text = guard_in_cell if guard_in_cell else "None"
    info_lines = [
        f"• Health: {game.current_state['player_health']}",
        f"• Position: {game.current_state['player_position']}",
        f"• Guard Here: {guard_text}",
        f"• At Trap: {'Yes' if obs.get('at_trap', False) else 'No'}",
        f"• At Heal: {'Yes' if obs.get('at_heal', False) else 'No'}",
    ]
    for line in info_lines:
        info_surface = font_info.render(line, True, BLACK)
        screen.blit(info_surface, (console_x, y_offset))
        y_offset += 28
    
    # Section divider
    y_offset += 10
    pygame.draw.line(screen, BLACK, (console_x, y_offset), (console_x + CONSOLE_WIDTH - 20, y_offset), 2)
    y_offset += 15
    
    # Recent actions header
    font_header = pygame.font.Font(None, 28)
    actions_header = font_header.render("Recent Actions:", True, BLUE)
    screen.blit(actions_header, (console_x, y_offset))
    y_offset += 35
    
    # Print the latest actions with word wrapping
    font_action = pygame.font.Font(None, 18)
    for result in action_results:
        if result:
            # Word wrap for long text
            words = result.split(' ')
            line = ""
            for word in words:
                test_line = line + word + " "
                # Check if line width exceeds console width
                if font_action.size(test_line)[0] > CONSOLE_WIDTH - 25:
                    # Render current line
                    result_surface = font_action.render(line.strip(), True, BLACK)
                    screen.blit(result_surface, (console_x, y_offset))
                    y_offset += 22
                    line = word + " "
                else:
                    line = test_line
            
            # Render remaining text
            if line.strip():
                result_surface = font_action.render(line.strip(), True, BLACK)
                screen.blit(result_surface, (console_x, y_offset))
                y_offset += 22
            
            # Add spacing between actions
            y_offset += 5

    # Check for terminal state
    pygame.display.flip()
    clock.tick(fps)
    time.sleep(delay)  # Use the delay parameter


if __name__ == "__main__":
    setup()
    main()