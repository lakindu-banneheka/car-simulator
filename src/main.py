import pickle
import sys
import neat
import pygame

from src.car import Car

# Constants
WIDTH = 1500
HEIGHT = 700

CAR_SIZE_X = 30
CAR_SIZE_Y = 30

BORDER_COLOR = (255, 255, 255, 255)

current_generation = 0


def run_simulation(genomes, config):
    # Empty Collections For Nets and Cars
    nets = []
    cars = []

    # Initialize PyGame and the display
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))  # Remove FULLSCREEN for debugging
    pygame.display.set_caption("F1 Simulator")

    # For All Genomes Passed Create A New Neural Network
    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0

        cars.append(Car())

    # Clock Settings
    clock = pygame.time.Clock()

    # Font Settings
    generation_font = pygame.font.SysFont("Arial", 30)
    alive_font = pygame.font.SysFont("Arial", 20)

    # Load map
    game_map = pygame.image.load('../data/Maps/map.png').convert()
    game_map = pygame.transform.scale(game_map, (WIDTH, HEIGHT))

    global current_generation
    current_generation += 1

    # Simple Counter To Roughly Limit Time (Not Good Practice)
    counter = 0

    # Draw Map And All Cars That Are Alive
    screen.blit(game_map, (0, 0))

    while True:
        # Exit On Quit Event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        # For Each Car Get The Acton It Takes
        for i, car in enumerate(cars):
            output = nets[i].activate(car.get_data())
            choice = output.index(max(output))
            if choice == 0:
                car.angle += 10  # Left
            elif choice == 1:
                car.angle -= 10  # Right
            elif choice == 2:
                if (car.speed - 2 >= 12):
                    car.speed -= 2  # Slow Down
            else:
                car.speed += 2  # Speed Up

        # Check If Car Is Still Alive
        # Increase Fitness If Yes And Break Loop If Not
        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                car.update(game_map)
                genomes[i][1].fitness += car.get_reward()

        if still_alive == 0:
            break

        counter += 1
        if counter == 30 * 40:  # Stop After About 20 Seconds
            break

        # Draw Map And All Cars That Are Alive
        screen.blit(game_map, (0, 0))
        for car in cars:
            if car.is_alive():
                car.draw(screen)

        # Display Info
        text = generation_font.render("Generation: " + str(current_generation), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (900, 450)
        screen.blit(text, text_rect)

        text = alive_font.render("Still Alive: " + str(still_alive), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (900, 490)
        screen.blit(text, text_rect)

        pygame.display.flip()
        clock.tick(60)  # 60 FPS



if __name__ == "__main__":
    # Load Config
    config_path = "../config/config.txt"
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_path)

    # Create Population And Add Reporters
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Run Simulation For A Maximum of 10 Generations
    winner = population.run(run_simulation, 10)

    # Save the best genome (winner) to a file
    with open("best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)

    print("Best genome saved as 'best_genome.pkl'")

# if __name__ == "__main__":
#     # Load Config
#     config_path = "../config/config.txt"
#     config = neat.config.Config(neat.DefaultGenome,
#                                 neat.DefaultReproduction,
#                                 neat.DefaultSpeciesSet,
#                                 neat.DefaultStagnation,
#                                 config_path)
#
#     # Create Population And Add Reporters
#     population = neat.Population(config)
#     population.add_reporter(neat.StdOutReporter(True))
#     stats = neat.StatisticsReporter()
#     population.add_reporter(stats)
#
#     # Run Simulation
#     population.run(run_simulation, 10)
#     winner = population.best_genome
#
#     # Save the best genome (winner) to a file
#     with open("best_genome_map1.pkl", "wb") as f:
#         pickle.dump(winner, f)
#
#     print("Best genome saved as 'best_genome.pkl'")

# if __name__ == "__main__":
#     # Load Config
#     config_path = "../config/config.txt"
#     config = neat.config.Config(neat.DefaultGenome,
#                                 neat.DefaultReproduction,
#                                 neat.DefaultSpeciesSet,
#                                 neat.DefaultStagnation,
#                                 config_path)
#
#     # Load the previously saved best genome (from first map)
#     with open("best_genome_map1.pkl", "rb") as f:
#         winner = pickle.load(f)
#
#     # Create a new population with the saved best genome
#     population = neat.Population(config)
#
#     # Replace the first genome with the best genome from the previous map
#     # population.population[0] = winner
#
#     # Add reporters
#     population.add_reporter(neat.StdOutReporter(True))
#     stats = neat.StatisticsReporter()
#     population.add_reporter(stats)
#
#     # Run simulation on the new map
#     population.run(run_simulation, 10)
#
#     # Save the new best genome from training on the new map
#     winner = population.best_genome
#     with open("best_genome_map1.pkl", "wb") as f:
#         pickle.dump(winner, f)
#
#     print("Best genome saved as 'best_genome_map2.pkl'")