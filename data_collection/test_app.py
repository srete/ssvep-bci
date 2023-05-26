import pygame
import time

# Define screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Define rectangle dimensions and colors
RECT_WIDTH = 200
RECT_HEIGHT = 200
RECT_COLOR_ON = pygame.Color(255, 0, 0)   # Red
RECT_COLOR_OFF = pygame.Color(0, 0, 0)    # Black

def main():
    # Initialize Pygame
    pygame.init()
    
    # Create the screen
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("SSVEP Stimulation")
    
    # Set initial rectangle position
    rect_x = (SCREEN_WIDTH - RECT_WIDTH) // 2
    rect_y = (SCREEN_HEIGHT - RECT_HEIGHT) // 2
    
    # Set initial blinking frequency and duration
    blinking_frequency = 30  # Hz
    stimulation_duration = 10  # seconds
    
    # Calculate blinking interval in milliseconds
    blinking_interval = int(1000 / (2*blinking_frequency))
    
    # Calculate the total number of blinks
    total_blinks = int(stimulation_duration * blinking_frequency)
    
    # Start the stimulation loop
    start_time = time.time()
    current_blink = 0
    is_rectangle_on = True
    
    while current_blink < total_blinks:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        # Calculate the elapsed time in milliseconds
        elapsed_time = int((time.time() - start_time) * 1000)
        
        # Check if it's time to toggle the rectangle
        if elapsed_time >= blinking_interval:
            is_rectangle_on = not is_rectangle_on
            start_time = time.time()
            current_blink += 1
        
        # Draw the rectangle
        screen.fill(pygame.Color(255, 255, 255))  # White background
        if is_rectangle_on:
            pygame.draw.rect(screen, RECT_COLOR_ON, (rect_x, rect_y, RECT_WIDTH, RECT_HEIGHT))
        else:
            pygame.draw.rect(screen, RECT_COLOR_OFF, (rect_x, rect_y, RECT_WIDTH, RECT_HEIGHT))
        pygame.display.flip()
    
    # Stimulation finished, quit the application
    pygame.quit()

if __name__ == "__main__":
    main()
