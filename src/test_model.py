import pygame
import neat
import pickle
from src.car import Car  # Import your existing Car class

# Constants
WIDTH = 1500
HEIGHT = 700
FPS = 60


def load_model(config_path, model_path):
    """
    Load the trained model (best genome) and return the corresponding neural network.
    """
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    # Load the saved genome
    with open(model_path, "rb") as f:
        best_genome = pickle.load(f)

    # Create and return the neural network from the best genome
    net = neat.nn.FeedForwardNetwork.create(best_genome, config)
    return net


def test_model_on_map(config_path, model_path, map_path):
    """
    Test the trained model on the map using the existing Car class.
    """
    # Initialize PyGame and the display
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))  # Remove FULLSCREEN for debugging
    pygame.display.set_caption("Model Test Run")

    # Clock Settings
    clock = pygame.time.Clock()

    # Load the saved model
    net = load_model(config_path, model_path)

    # Load map
    game_map = pygame.image.load(map_path).convert()
    game_map = pygame.transform.scale(game_map, (WIDTH, HEIGHT))

    # Initialize the car
    car = Car()  # Use the existing Car class

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get the inputs for the neural network from the car's sensors
        inputs = car.get_data()  # Get radar distances from car

        # Get the outputs from the neural network
        outputs = net.activate(inputs)

        # Control the car using the neural network outputs
        # Assuming the network has two outputs: steering and acceleration
        car.angle += outputs[0] * 10  # Steering control (modify as necessary)
        car.speed += outputs[1] * 2  # Speed control (modify as necessary)

        # Update the car's position
        car.update(game_map)

        # Check if the car is still alive (collision detection)
        if not car.is_alive():
            print("Collision detected! Ending test.")
            running = False

        # Draw the map and the car
        screen.blit(game_map, (0, 0))
        car.draw(screen)

        # Update the display
        pygame.display.flip()
        clock.tick(FPS)

    # pygame.quit()


if __name__ == "__main__":
    # Paths to configuration, model, and map
    config_path = "../config/config.txt"
    model_path = "best_genome.pkl"
    map_path = "../data/Maps/map.png"

    test_model_on_map(config_path, model_path, map_path)
