import pygame
import math


class Robot:

    def __init__(self, game, x=100, y=100, size=50, angle=180):
        self.memory = [0, 0, 0]
        self.game = game
        self.current_checkpoint_index = 0
        self.fitness = None
        self.alive = True
        self.active = True
        self.reached_end = False
        self.image = pygame.Surface((size, size))
        self.rect = self.image.get_rect()
        self.speed = float(self.game.config["GAME"]["speed"])
        self.turn_speed = int(self.game.config["GAME"]["turn_speed"])
        # change this to random
        self.rect.center = (x, y)
        self.size = size
        self.colour = (255, 255, 255)
        self.distance_sensors = []
        self.angle = angle
        for angle, x, y in [(-45, math.sin(0.25 * math.pi) * self.size // 2, -math.sin(0.25 * math.pi) * self.size // 2),
                            (0, self.size//2, 0),
                            (45, math.sin(0.25 * math.pi) * self.size // 2, math.sin(0.25 * math.pi) * self.size // 2)]:
            print(x, y)
            self.distance_sensors.append(Robot.DistanceSensor(self, initial_angle=angle, rel_x=x, rel_y=y))
        # make random on spawn
        self.x = self.rect.centerx
        self.y = self.rect.centery
        self.collision_points = [(0, self.size // 2), (0, -self.size // 2), (self.size // 2, 0), (-self.size // 2, 0),
                                 (math.sin(0.25 * math.pi) * self.size // 2,) * 2,
                                 (-math.sin(0.25 * math.pi) * self.size // 2,) * 2,
                                 (
                                 math.sin(0.25 * math.pi) * self.size // 2, -math.sin(0.25 * math.pi) * self.size // 2),
                                 (
                                 -math.sin(0.25 * math.pi) * self.size // 2, math.sin(0.25 * math.pi) * self.size // 2)]
        self.update()

    def move(self, value):  # choose value between 1 and -1
        if value > 0:
            if value > 1:
                value = 1
            self.go_forward(value)
        elif value < 0:
            if value < -1:
                value = -1
            self.go_backward(value)

    def go_forward(self, speed):
        y = math.sin(self.angle * (math.pi / 180)) * self.speed * speed
        x = math.cos(self.angle * (math.pi / 180)) * self.speed * speed
        self.x += x
        self.y += y

    def go_backward(self, speed):
        y = math.sin(self.angle * (math.pi / 180)) * self.speed * speed
        x = math.cos(self.angle * (math.pi / 180)) * self.speed * speed
        self.x += x
        self.y += y

    def turn(self, degrees):
        self.angle += degrees * self.turn_speed
        self.angle = self.angle % 360

    def render(self):
        pygame.draw.circle(self.image, self.colour, center=(self.size // 2, self.size // 2),
                           radius=self.size // 2)
        self.game.screen.blit(self.image, self.rect)
        for point in self.get_colission_points():
            pygame.draw.circle(self.game.screen, (255, 0, 255), center=point, radius=3)
        for sensor in self.distance_sensors:
            sensor.update()

    def get_colission_points(self):
        return_list = []
        for point in self.collision_points:
            x, y = point
            x += self.x
            y += self.y
            return_list.append((x, y))
        return return_list

    def update(self):
        if self.alive and self.active:
            self.rect.center = (self.x, self.y)
            if not self.game.game_config["train"]:
                self.render()
            # collision detection
            # might be better with collide dict of a dict with rect
            for obs in self.game.obstacles:
                result = False
                for point in self.get_colission_points():
                    result = obs.rect.collidepoint(point)
                    if result:
                        self.death()
            if self.reached_checkpoint():
                if self.current_checkpoint_index == len(self.game.checkpoints) - 1:
                    self.active = False
                    self.reached_end = True
                else:
                    self.current_checkpoint_index += 1

            if self.game.max_frames / len(self.game.checkpoints) * (
                    self.current_checkpoint_index + 1) < self.game.frames:
                self.death()

    def death(self):
        self.alive = False

    def calculate_fitness(self):
        points = 0
        if not self.reached_end:
            distance_to_destination = self.game.checkpoints[self.current_checkpoint_index].get_distance(self)
            points += (1 / (distance_to_destination * float(
                self.game.config["TRAINING"]["distance_reward_multiplier"]) ** 2)) * 100

        points += self.current_checkpoint_index * 100

        if self.alive:
            points += float(self.game.config["TRAINING"]["alive_point_reward"])

        if self.reached_end:
            points += float(self.game.config["TRAINING"]["reached_end_reward"]) * (
                        self.game.max_frames / self.game.frames) * 100
        return points

    def get_pos(self):
        return self.x, self.y

    def reached_destination(self):
        return self.game.destination.rect.colliderect(self.rect)

    def reached_checkpoint(self):
        return self.game.checkpoints[self.current_checkpoint_index].rect.colliderect(self.rect)

    def get_data(self):
        results = []
        for sensor in self.distance_sensors:
            results.append(sensor.get_distance())
        direction = self.game.checkpoints[self.current_checkpoint_index].get_direction(self)
        results.append(direction)
        # results.append(self.game.checkpoints[self.current_checkpoint_index].get_distance(self))
        direction = [direction]
        results += self.memory
        return results
        # return results

    class DistanceSensor:
        def __init__(self, robot, initial_angle=0, rel_x=0, rel_y=0):
            self.robot = robot
            self.initial_angle = initial_angle
            self.length = math.sqrt(rel_y**2+rel_x**2)
            self.rotation = math.atan2(rel_y, rel_x)
            self.update()

        def get_position(self):
            x, y = self.robot.rect.center
            print(self.rotation)
            x += math.cos(self.robot.angle*math.pi/180 + self.rotation) * self.length
            y += math.sin(self.robot.angle*math.pi/180 + self.rotation) * self.length
            return x, y

        def update(self):
            if self.robot.game.game_config["train"]:
                return
            pos = self.get_position()
            x, y = pos
            self.line = [x, y]
            angle = self.robot.angle
            angle += self.initial_angle
            y += math.sin(angle * (math.pi / 180)) * int(self.robot.game.config["GAME"]["distance_length"])
            x += math.cos(angle * (math.pi / 180)) * int(self.robot.game.config["GAME"]["distance_length"])
            pygame.draw.line(self.robot.game.screen, (255, 255, 0), pos, (x, y))
            self.line.append(x)
            self.line.append(y)

        # returns a value between 1 and 0
        # 1 if it doesnt detect anything or is at the very end and 0 if its close
        def get_distance(self):
            # this might be faster with this
            # https://www.pygame.org/wiki/IntersectingLineDetection
            # or with rect.collidepoint(point)
            x, y = self.get_position()
            length = 0
            increment = 10
            max_length = int(self.robot.game.config["GAME"]["distance_length"])
            angle = self.robot.angle
            angle += self.initial_angle
            while length < max_length:
                length += increment
                px = x + math.cos(angle * (math.pi / 180)) * length
                py = y + math.sin(angle * (math.pi / 180)) * length
                # check if point is in an object
                for obs in self.robot.game.obstacles:
                    pos_x, pos_y, width, height = obs.rect
                    # if px not between the x coordinates of the obstacle skip
                    if not (pos_x < px < pos_x + width):
                        continue
                    if not (pos_y < py < pos_y + height):
                        continue
                    return (1 - (length / max_length)) * 30
            return 0
