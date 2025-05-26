import pygame
import random
import sys

# Initialize Pygame
pygame.init()

# Game constants
BOARD_WIDTH = 16
BOARD_HEIGHT = 16
MINE_COUNT = 25
CELL_SIZE = 30
WINDOW_WIDTH = BOARD_WIDTH * CELL_SIZE
WINDOW_HEIGHT = BOARD_HEIGHT * CELL_SIZE + 60  # Extra space for UI

# Colors
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
LIGHT_GRAY = (192, 192, 192)
DARK_GRAY = (64, 64, 64)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Number colors
NUMBER_COLORS = {
    1: (0, 0, 255),      # Blue
    2: (0, 128, 0),      # Green
    3: (255, 0, 0),      # Red
    4: (128, 0, 128),    # Purple
    5: (128, 0, 0),      # Maroon
    6: (64, 224, 208),   # Turquoise
    7: (0, 0, 0),        # Black
    8: (128, 128, 128)   # Gray
}

class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.is_mine = False
        self.is_revealed = False
        self.is_flagged = False
        self.neighbor_mines = 0
        
class Minesweeper:
    def __init__(self):
        self.board = [[Cell(x, y) for x in range(BOARD_WIDTH)] for y in range(BOARD_HEIGHT)]
        self.game_over = False
        self.game_won = False
        self.first_click = True
        self.mines_left = MINE_COUNT
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 20)
        
    def place_mines(self, first_click_x, first_click_y):
        """Place mines randomly, avoiding the first clicked cell"""
        mines_placed = 0
        while mines_placed < MINE_COUNT:
            x = random.randint(0, BOARD_WIDTH - 1)
            y = random.randint(0, BOARD_HEIGHT - 1)
            
            # Don't place mine on first click or if already has mine
            if (x == first_click_x and y == first_click_y) or self.board[y][x].is_mine:
                continue
                
            self.board[y][x].is_mine = True
            mines_placed += 1
            
        self.calculate_numbers()
        
    def calculate_numbers(self):
        """Calculate number of neighboring mines for each cell"""
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                if not self.board[y][x].is_mine:
                    count = 0
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < BOARD_HEIGHT and 0 <= nx < BOARD_WIDTH:
                                if self.board[ny][nx].is_mine:
                                    count += 1
                    self.board[y][x].neighbor_mines = count
                    
    def reveal_cell(self, x, y):
        """Reveal a cell and handle game logic"""
        if self.game_over or self.game_won:
            return
            
        cell = self.board[y][x]
        
        # Can't reveal flagged cells
        if cell.is_flagged:
            return
            
        # First click - place mines
        if self.first_click:
            self.place_mines(x, y)
            self.first_click = False
            
        # Already revealed
        if cell.is_revealed:
            return
            
        cell.is_revealed = True
        
        # Hit a mine
        if cell.is_mine:
            self.game_over = True
            self.reveal_all_mines()
            return
            
        # If cell has no neighboring mines, reveal all neighbors
        if cell.neighbor_mines == 0:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < BOARD_HEIGHT and 0 <= nx < BOARD_WIDTH:
                        if not self.board[ny][nx].is_revealed:
                            self.reveal_cell(nx, ny)
                            
        self.check_win()
        
    def toggle_flag(self, x, y):
        """Toggle flag on a cell"""
        if self.game_over or self.game_won:
            return
            
        cell = self.board[y][x]
        
        if cell.is_revealed:
            return
            
        cell.is_flagged = not cell.is_flagged
        self.mines_left += 1 if not cell.is_flagged else -1
        
    def reveal_all_mines(self):
        """Reveal all mines when game is over"""
        for row in self.board:
            for cell in row:
                if cell.is_mine:
                    cell.is_revealed = True
                    
    def check_win(self):
        """Check if player has won"""
        for row in self.board:
            for cell in row:
                if not cell.is_mine and not cell.is_revealed:
                    return
        self.game_won = True
        
    def reset_game(self):
        """Reset the game"""
        self.__init__()
        
    def draw(self, screen):
        """Draw the game board"""
        # Draw header
        header_rect = pygame.Rect(0, 0, WINDOW_WIDTH, 60)
        pygame.draw.rect(screen, LIGHT_GRAY, header_rect)
        
        # Draw mines counter
        mines_text = self.font.render(f"Mines: {self.mines_left}", True, BLACK)
        screen.blit(mines_text, (10, 20))
        
        # Draw status
        if self.game_over:
            status_text = self.font.render("GAME OVER! Press R to restart", True, RED)
        elif self.game_won:
            status_text = self.font.render("YOU WIN! Press R to restart", True, GREEN)
        else:
            status_text = self.font.render("Left click: reveal, Right click: flag", True, BLACK)
        
        text_rect = status_text.get_rect()
        text_rect.centerx = WINDOW_WIDTH // 2
        text_rect.y = 35
        screen.blit(status_text, text_rect)
        
        # Draw board
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                cell = self.board[y][x]
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE + 60, CELL_SIZE, CELL_SIZE)
                
                # Draw cell background
                if cell.is_revealed:
                    if cell.is_mine:
                        pygame.draw.rect(screen, RED, rect)
                    else:
                        pygame.draw.rect(screen, WHITE, rect)
                else:
                    pygame.draw.rect(screen, GRAY, rect)
                
                # Draw cell border
                pygame.draw.rect(screen, BLACK, rect, 1)
                
                # Draw cell content
                if cell.is_flagged and not cell.is_revealed:
                    # Draw flag
                    flag_text = self.small_font.render("F", True, RED)
                    text_rect = flag_text.get_rect()
                    text_rect.center = rect.center
                    screen.blit(flag_text, text_rect)
                    
                elif cell.is_revealed:
                    if cell.is_mine:
                        # Draw mine
                        mine_text = self.small_font.render("*", True, BLACK)
                        text_rect = mine_text.get_rect()
                        text_rect.center = rect.center
                        screen.blit(mine_text, text_rect)
                    elif cell.neighbor_mines > 0:
                        # Draw number
                        color = NUMBER_COLORS.get(cell.neighbor_mines, BLACK)
                        num_text = self.small_font.render(str(cell.neighbor_mines), True, color)
                        text_rect = num_text.get_rect()
                        text_rect.center = rect.center
                        screen.blit(num_text, text_rect)
                        
    def handle_click(self, pos, button):
        """Handle mouse clicks"""
        x = pos[0] // CELL_SIZE
        y = (pos[1] - 60) // CELL_SIZE
        
        if 0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT:
            if button == 1:  # Left click
                self.reveal_cell(x, y)
            elif button == 3:  # Right click
                self.toggle_flag(x, y)

def main():
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption('Minesweeper')
    clock = pygame.time.Clock()
    
    game = Minesweeper()
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                game.handle_click(event.pos, event.button)
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game.reset_game()
        
        screen.fill(WHITE)
        game.draw(screen)
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()