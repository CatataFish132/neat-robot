import pygame
import sys
import random
import configparser
import neat
from viewgenome import draw_net
import gzip
from robot import Robot
import math
import os
import signal
import sys

try:
    import cPickle as pickle
except ImportError:
    import pickle


class Game:
    def __init__(self, level, population=50, max_gen=50, filename="neat",
                 save=True, test=False, neatfile=None, train=False, make=False, interactive=False):
        self.game_config = {"gen": max_gen, "filename": filename, "save": save, "level": level, "test": test,
                            "neatfile": neatfile, "train": train}
        signal.signal(signal.SIGINT, self.signal_handler)
        pygame.init()
        pygame.font.init()
        self.average_fitness_list = []
        self.config = configparser.ConfigParser()
        self.config.read("config.ini")
        self.random_spawn = True
        self.random_checkpoint = True
        self.spawn_pos = (0, 0)
        self.amount = int(self.config["TRAINING"]["robot_amount"])
        self.size = int(self.config["GAME"]["screen_x"]), int(self.config["GAME"]["screen_y"])
        self.black = 0, 0, 0
        if not train:
            self.screen = pygame.display.set_mode(self.size)
        self.obstacles = []
        if not make:
            try:
                self.level_builder(level)
            except FileNotFoundError:
                exit("that level doesn't exist")
        # self.default()
        # spawn robots
        self.robots = []
        self.clock = pygame.time.Clock()
        self.frames = 0
        self.max_frames = int(self.config["TRAINING"]["round_time"]) * 60
        self.generation = 0
        self.mutation_rate = int(self.config["TRAINING"]["starting_mutating_rate"])
        self.font = pygame.font.SysFont('Comic Sans MS', 30)
        self.checkpoint_font = pygame.font.SysFont("Arial", 20)
        self.text_surface = self.font.render('Generation 0', False, (255, 255, 255))
        #
        # self.test_loop()

        config_path = r"config.txt"
        self.nets = []
        self.neat_config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                              neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
        self.neat_config.pop_size = population
        try:
            self.p = neat.Population(self.neat_config)
        except ZeroDivisionError:
            exit("Population size has to be bigger than 1")
        self.p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        self.p.add_reporter(stats)
        self.generation = 0
        if test:
            genome, generation, config = self.open("generated/" + neatfile)
            self.run_test(genome, config, interactive=interactive)
        elif make:
            self.map_maker_loop()
        else:
            self.p.run(self.run_robots, max_gen)
            if save:
                self.save(filename)

    def test_loop(self):
        self.robots.append(Robot(self, x=200, y=200, angle=0))
        while True:
            self.screen.fill(self.black)
            for obstacle in self.obstacles:
                obstacle.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
            keys = pygame.key.get_pressed()
            turn = 0
            forward = 0
            if keys[pygame.K_RIGHT]:
                turn += 1
            if keys[pygame.K_LEFT]:
                turn -= 1
            if keys[pygame.K_UP]:
                forward += 1
            if keys[pygame.K_DOWN]:
                forward -= 1
            if keys[pygame.K_t]:
                print(self.robots[0].get_data())
            self.robots[0].turn(turn)
            self.robots[0].go_forward(forward)
            self.robots[0].update()
            # render text
            self.screen.blit(self.text_surface, (0, 0))
            # render destination
            pygame.display.flip()
            # print(self.destination.get_direction(self.robots[0]))
            self.clock.tick(240)

    def loop(self, genomes):
        if not self.game_config["train"]:
            self.text_surface = self.font.render(f'Generation {self.generation}', False, (255, 255, 255))
            self.generation += 1
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
            if not self.game_config["train"]:
                self.screen.fill(self.black)
            alive_counter = self.robot_input()
            if not self.game_config["train"]:
                for obstacle in self.obstacles:
                    obstacle.update()
                for checkpoint in self.checkpoints:
                    checkpoint.update()
                # render text
                self.screen.blit(self.text_surface, (0, 0))
                pygame.draw.circle(self.screen, (255, 0, 255), self.spawn_pos, 20)
                pygame.display.flip()
            self.frames += 1
            if self.frames == self.max_frames or alive_counter == 0:
                for i, robot in enumerate(self.robots):
                    genomes[i][1].fitness += robot.calculate_fitness()
                # if alive_counter == 0:
                self.frames = 0
                best = 0
                best_gnome = None
                for id, g in genomes:
                    if g.fitness > best:
                        best = g.fitness
                        best_gnome = g
                draw_net(self.neat_config, best_gnome)
                break
            # self.clock.tick(60)

    def get_random_spawn(self):
        while True:
            valid = True
            spawn_point = (random.randint(100, 700), random.randint(100, 700))
            for obs in self.obstacles:
                border = 50
                center = obs.rect.center
                pos_x, pos_y, width, height = obs.rect
                image = pygame.Surface((width + border, height + border))
                rect = image.get_rect()
                rect.center = center
                x, y = spawn_point
                if rect.collidepoint(x, y):
                    valid = False
                    break
            if valid:
                return spawn_point

    def get_random_destination_spawn(self):
        while True:
            valid = True
            robot_x, robot_y = self.spawn_pos
            spawn_point = (random.randint(100, 700), random.randint(100, 700))
            x, y = spawn_point
            if pygame.Vector2(x - robot_x, y - robot_y).length() < 300:
                continue
            for obs in self.obstacles:
                border = 50
                center = obs.rect.center
                pos_x, pos_y, width, height = obs.rect
                image = pygame.Surface((width + border, height + border))
                rect = image.get_rect()
                rect.center = center
                if rect.collidepoint(x, y):
                    valid = False
                    break
            if valid:
                return spawn_point

    def run_robots(self, genomes, config):
        x, y = self.spawn_pos
        angle = 0
        self.robots = []
        self.nets = []
        for id, g in genomes:
            net = neat.nn.FeedForwardNetwork.create(g, config)
            self.nets.append(net)
            g.fitness = 0
            self.robots.append(Robot(self, x=x, y=y, angle=angle))
        self.loop(genomes)

    def robot_input(self):
        alive_counter = 0
        for i, robot in enumerate(self.robots):
            if robot.alive and robot.active:
                alive_counter += 1
                output = self.nets[i].activate(robot.get_data())
                robot.turn(output[0])
                robot.go_forward(abs(output[1]))
                robot.memory = [output[2], output[3], output[4]]
                #if output[2] > 0.8:
                #    robot.memory = 30
                #elif output[2] < 0:
                #    robot.memory += output[2]
                #    if robot.memory < 0:
                #        robot.memory = 0
                robot.update()
        return alive_counter

    def run_test(self, genome, config, net=None, interactive=False):
        self.robots = []
        self.nets = []
        x, y = self.spawn_pos
        if interactive:
            self.checkpoints = []
            self.checkpoints.append(CheckPoint(self, x, y, 0))
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.nets.append(net)
        self.robots.append(Robot(self, x=x, y=y, angle=0))

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN and interactive:
                    pos = pygame.mouse.get_pos()
                    x, y = pos
                    self.checkpoints = []
                    self.checkpoints.append(CheckPoint(self, x, y, 0))
                    for robot in self.robots:
                        robot.current_checkpoint_index = 0
                        robot.active = True
            self.screen.fill((0, 0, 0))
            alive_counter = self.robot_input()
            for i, robot in enumerate(self.robots):
                if not robot.alive:
                    x, y = self.spawn_pos
                    robot.x = x
                    robot.y = y
                    robot.angle = 0
                    robot.current_checkpoint_index = 0
                    robot.active = True
                    robot.alive = True
                if robot.alive and not robot.active:
                    robot.render()
            for obstacle in self.obstacles:
                obstacle.update()
            for checkpoint in self.checkpoints:
                checkpoint.update()
            self.screen.blit(self.text_surface, (0, 0))
            pygame.draw.circle(self.screen, (255, 0, 255), self.spawn_pos, 20)
            pygame.display.flip()
            self.clock.tick(60)

    def level_builder(self, filename):
        self.obstacles = []
        filename = "levels/" + filename
        with gzip.open(filename) as f:
            obs_list, checkpoint_list, spawn_pos = pickle.load(f)
        self.checkpoints = [None] * len(checkpoint_list)
        for obs in obs_list:
            self.obstacles.append(Obstacle(self, obs["x"], obs["y"], obs["width"], obs["height"]))
        for checkpoint in checkpoint_list:
            self.checkpoints[checkpoint["index"]] = (
                CheckPoint(self, checkpoint["x"], checkpoint["y"], checkpoint["index"]))
        self.spawn_pos = spawn_pos

    def map_maker_loop(self):
        """"map maker loop"""

        def get_obs_pos(pos_obs, mouse_pos):
            x, y = mouse_pos
            obs_x, obs_y = pos_obs
            return (obs_x * (obs_x < x) + x * (x < obs_x),
                    obs_y * (obs_y < y) + y * (y < obs_y),
                    obs_x * (obs_x > x) + x * (x > obs_x) - obs_x * (obs_x < x) - x * (x < obs_x),
                    obs_y * (obs_y > y) + y * (y > obs_y) - obs_y * (obs_y < y) - y * (y < obs_y))

        # exit game when cross is clicked
        self.obstacles = []
        self.checkpoints = []
        # make spawn not visible outside screen
        self.spawn_pos = (-50, -50)
        selected = "obstacle"
        stage = 0
        modes = ["checkpoint", "obstacle", "spawn", "remove_obstacle"]
        current_mode_number = 1
        obs_pos = (0, 0)
        angle = 0
        self.text_surface = self.font.render(selected, False, (255, 255, 255))
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("saving...")
                    obs_list = []
                    checkpoint_list = []
                    for obs in self.obstacles:
                        x, y = obs.rect.topleft
                        obs_list.append({"x": x, "y": y, "width": obs.image.get_width(),
                                         "height": obs.image.get_height()})
                    for checkpoint in self.checkpoints:
                        checkpoint_list.append({"x": checkpoint.x, "y": checkpoint.y, "index": checkpoint.index})

                    filename = self.game_config["level"]
                    if not os.path.exists('levels'):
                        os.makedirs('levels')
                    filename = "levels/" + filename
                    with gzip.open(filename, 'w', compresslevel=5) as f:
                        data = (obs_list, checkpoint_list, self.spawn_pos)
                        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    if event.button == 1:
                        x, y = pygame.mouse.get_pos()
                        if selected == "checkpoint":
                            self.checkpoints.append(CheckPoint(self, x, y, len(self.checkpoints)))
                        elif selected == "spawn":
                            if stage == 0:
                                self.spawn_pos = (x, y)
                                stage = 1
                            elif stage == 1:
                                stage = 0
                                print(angle)
                        elif selected == "obstacle":
                            if stage == 0:
                                obs_pos = (x, y)
                                stage = 1
                            elif stage == 1:
                                self.obstacles.append(Obstacle(self, *get_obs_pos(obs_pos, (x, y))))
                                stage = 0
                        elif selected == "remove_obstacle":
                            for index, obs in enumerate(self.obstacles):
                                if obs.rect.collidepoint(x, y):
                                    self.obstacles.pop(index)
                                    break
                    elif event.button == 3:
                        if selected == "checkpoint":
                            if len(self.checkpoints) != 0:
                                self.checkpoints.pop()
                        elif selected == "obstacles" and stage == 1:
                            stage = 0
                    elif event.button == 4:
                        stage = 0
                        current_mode_number += 1
                        current_mode_number %= len(modes)
                        selected = modes[current_mode_number]
                        self.text_surface = self.font.render(selected, False, (255, 255, 255))
                    elif event.button == 5:
                        stage = 0
                        current_mode_number -= 1
                        current_mode_number %= len(modes)
                        selected = modes[current_mode_number]
                        self.text_surface = self.font.render(selected, False, (255, 255, 255))
            self.screen.fill((0, 0, 0))
            if stage == 1 and selected == "obstacle":
                Obstacle(self, *get_obs_pos(obs_pos, pygame.mouse.get_pos())).update()
            elif stage == 1 and selected == "spawn":
                spawn_x, spawn_y = self.spawn_pos
                x, y = pygame.mouse.get_pos()
                end_pos_x, end_pos_y = self.spawn_pos
                angle = (math.atan2(y - spawn_y, x - spawn_x) * 180 / math.pi)
                end_pos_y += math.sin(angle * (math.pi / 180)) * int(100)
                end_pos_x += math.cos(angle * (math.pi / 180)) * int(100)
                pygame.draw.line(self.screen, (255, 255, 255), self.spawn_pos, (end_pos_x, end_pos_y))
            for obstacle in self.obstacles:
                obstacle.update()
            for checkpoint in self.checkpoints:
                checkpoint.update()
            self.screen.blit(self.text_surface, (0, 0))
            pygame.draw.circle(self.screen, (255, 0, 255), self.spawn_pos, 20)
            pygame.display.flip()
            self.clock.tick(60)

    def save(self, filename):
        if not os.path.exists('generated'):
            os.makedirs('generated')
        filename = "generated/" + filename
        with gzip.open(filename, 'w', compresslevel=5) as f:
            data = (self.p.best_genome, self.p.generation, self.p.config)
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def open(self, filename):
        with gzip.open(filename) as f:
            best_genome, generation, config = pickle.load(f)
            return best_genome, generation, config

    def signal_handler(self, sig, frame):
        print(f'You pressed Ctrl+C!\nsaving to {self.game_config["filename"]}')
        self.save(self.game_config["filename"])
        sys.exit(0)

class CheckPoint:
    def __init__(self, game, x, y, index=0):
        self.index = index
        self.game = game
        self.x = x
        self.y = y
        self.pos = (x, y)
        self.image = pygame.Surface((20, 20))
        self.rect = self.image.get_rect()
        self.rect.center = self.pos

    def get_pos(self):
        return self.pos

    # test if this works well
    def get_direction(self, robot):
        robot_pos = robot.get_pos()
        angle = robot.angle
        x, y = robot_pos
        robot_pos = pygame.Vector2(x, y)
        x, y = self.pos
        pos = pygame.Vector2(x, y)
        if pos == robot_pos:
            return 0
        angle = math.atan2(pos.y - robot_pos.y, pos.x - robot_pos.x) * 180 / math.pi
        # 0 while facing direction 1 or -1 when not
        correctness = (angle - robot.angle + 180) % 360
        # 1 -1 when facing direction 0 when not
        # correctness = (angle - robot.angle) % 360

        correctness = (correctness - 180) / 180
        return correctness

    def get_distance(self, robot):
        robot_pos = robot.get_pos()
        x, y = robot_pos
        robot_pos = pygame.Vector2(x, y)
        x, y = self.pos
        pos = pygame.Vector2(x, y)
        return (pos - robot_pos).length()

    def update(self):
        pygame.draw.circle(self.image, (0, 128, 128), center=(20 // 2, 20 // 2), radius=20 // 2)
        self.text_surface = self.game.checkpoint_font.render(str(self.index), False, (255, 255, 255))
        self.game.screen.blit(self.image, self.pos)
        self.game.screen.blit(self.text_surface, (self.x + 6, self.y - 3))


class Obstacle:
    def __init__(self, game, x, y, width, height):
        self.game = game
        self.image = pygame.Surface((width, height))
        self.rect = self.image.get_rect()
        self.rect.topleft = (x, y)
        self.image.fill((0, 255, 0))

    def update(self):
        self.game.screen.blit(self.image, self.rect)
