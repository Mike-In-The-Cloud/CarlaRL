import glob
import os
import sys
import argparse
import time
import random
import numpy as np
import math
import cv2
import pygame 
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Flatten, AveragePooling2D, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model, backend, Sequential
from tensorflow.keras.activations import relu
from collections import deque
from threading import Thread
from tqdm import tqdm
tf.compat.v1.disable_eager_execution()

from ModdedTensorboard import ModifiedTensorBoard
# ============================================= #
# Path to carla                                 #
# ============================================= #
try:
    sys.path.append(glob.glob('/home/michael/carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 30
REPLAY_MEMORY_SIZE = 4_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "Initial"

MEMORY_FRACTION = 0.4
MIN_REWARD = -200

EPISODES = 100

DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.95 ## 0.9975 99975
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 10
"""
list of waypoints
x = 117.6 y = 59.2 z = 0
x = 98.5 y = 57.7 z = 0
x = 83.5 y = 49.6 z = 0
x = 64.2 y = 57.6 z = 0
x = 74.1 y = 71.8 z = 0
x = 99.5 y = 63 z = 0
x = 114.5 y = 62.5 z = 0
x = 146 y = 62.5 z = 0
"""
# ==============================================================================
# -- Environment() -------------------------------------------------------------
# ==============================================================================
class Environment:
    # ============================================= #
    # variables used                                #
    # ============================================= #

    actor_list = [] # empty list to store the actors spawned
    distance_travelled = [] # empty list to store distance travelled
    number_of_collisions = [] # empty list of collisions
    collision_history = [] # empty list to store collistion history
    SHOW_CAM = SHOW_PREVIEW
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    SECONDS_PER_EPISODE = 45
    front_camera = None
    
    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(2.0)

        # Once we have a client we can retrieve the world that is currently
        # running.
        self.world = self.client.get_world()

        # The world contains the list blueprints that we can use for adding new
        # actors into the simulation.
        self.blueprint_library = self.world.get_blueprint_library()

        # Now let's filter all the blueprints of type 'vehicle' and choose one
        # at random.
        #print(blueprint_library.filter('vehicle'))
        self.bp_car  = self.blueprint_library.filter("bmw")[0] # filter for a car

    def reset(self):
        self.collision_history = []
        self.actor_list = []

        self.spawn_point = carla.Transform(carla.Location(x=147 , y = 59, z = 0.5), carla.Rotation(yaw=180)) # spawn point for the vehicle
        self.vehicle = self.world.spawn_actor(self.bp_car, self.spawn_point)
        self.actor_list.append(self.vehicle)

        # ============================================= #
        # Camera sensor                                 #
        # ============================================= #
        self.bp_camera = self.blueprint_library.find("sensor.camera.rgb") # camera sensor RGBA
        self.bp_camera.set_attribute("image_size_x", str(self.im_width)) # set image width
        self.bp_camera.set_attribute("image_size_y", str(self.im_height)) # set iamge height
        self.bp_camera.set_attribute("fov", "110") # field of view for camera
        spawn_point = carla.Transform(carla.Location(x = 2, z = 1.0)) # spawn camera relative to the vehicle
        self.sensor_cam = self.world.spawn_actor(self.bp_camera, spawn_point, attach_to = self.vehicle)
        self.actor_list.append(self.sensor_cam)
        self.sensor_cam.listen(lambda data: self.process_img(data))

        # apply control to the actor 
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        time.sleep(4) # sleep to get things started and to not detect a collision when the car spawns/falls from sky.

        # ============================================= #
        # collision  sensor                             #
        # ============================================= #
        bp_collision = self.world.get_blueprint_library().find('sensor.other.collision')
        self.bp_collision = self.world.spawn_actor(bp_collision, spawn_point, attach_to=self.vehicle)
        self.actor_list.append(self.bp_collision)
        self.bp_collision.listen(lambda event: self.collision_data(event))

        """
        while self.front_camera is None:
            time.sleep(0.01)
        """
        self.episode_start = time.time()

        self.vehicle.apply_control(carla.VehicleControl(brake=1.0, throttle=0.0))
        return self.front_camera

    # ============================================= #
    # collision data                                #
    # ============================================= #
    def collision_data(self, event):
        self.collision_history.append(event)

    # ============================================= #
    # image pre-processing                          #
    # ============================================= #
    def process_img(self, image):
        img = np.array(image.raw_data) # raw data is flattened array
        img2 = img.reshape((self.im_height, self.im_width, 4)) # reshape image to 640x480 with 4 colour channels RGBA
        img3 = img2[:, :, :3] # removes the alpha values from the image array
        return img3 # normalised image array    


    # ============================================= #
    # step action                                   #
    # ============================================= #
    def step(self, action):
        '''
        actions that can be taken
        0 - apply throttle
        1 - apply brake
        2 - apply throttle and steer left
        3 - apply throttle and steer right
        4 - apply steer left
        5 - apply steer right
        6 - apply brake and steer left
        7 - apply brake and steer right
        8 - no action
        '''
        apply_throttle = 1 
        apply_brake = 1
        apply_steer_left = -1
        apply_steer_right = 1
        no_action = 0

        if action == 0: # throttle only 
            self.vehicle.apply_control(carla.VehicleControl(throttle = apply_throttle))
        if action == 1: # brake only 
            self.vehicle.apply_control(carla.VehicleControl(brake = apply_brake))
        if action == 2: # throttle and left
            self.vehicle.apply_control(carla.VehicleControl(throttle = apply_throttle, steer = apply_steer_left))
        if action == 3: # throttle and right
            self.vehicle.apply_control(carla.VehicleControl(throttle = apply_throttle, steer = apply_steer_right))
        if action == 4: # steer left only 
            self.vehicle.apply_control(carla.VehicleControl(steer = apply_steer_left))
        if action == 5: # steer right only 
            self.vehicle.apply_control(carla.VehicleControl(steer = apply_steer_right))
        if action == 6: # brake and steer left
            self.vehicle.apply_control(carla.VehicleControl(brake = apply_brake, steer = apply_steer_left))
        if action == 7: # brake and steer right
            self.vehicle.apply_control(carla.VehicleControl(brake = apply_brake, steer = apply_steer_right))
        if action == 8: # no action
            self.vehicle.apply_control(carla.VehicleControl(throttle = no_action, brake = no_action, steer = no_action))

        velocity = self.vehicle.get_velocity()
        KPH = int(3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))


        """
        simple reward function
        """
        if len(self.collision_history) != 0:
            done = True
            reward = -1
        elif KPH > 50:
            done = False
            reward = -1
        else:
            done = False
            reward = 1

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return self.front_camera, reward, done, None


# ==============================================================================
# -- Agent() ---------------------------------------------------------------
# ==============================================================================
class Agent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0  # will track when it's time to update the target model
        self.graph = tf.compat.v1.get_default_graph()

        self.terminate = False  # Should we quit?
        self.last_logged_episode = 0
        self.training_initialized = False  # waiting for TF to get rolling

    def create_model(self):
        tf.compat.v1.disable_eager_execution()
        model = Sequential()

        model.add(Conv2D(64, (3, 3), input_shape=(480,640,3),activation = "relu", padding='same'))
        model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        model.add(Conv2D(64, (3, 3), activation = "relu", padding='same'))
        model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        model.add(Conv2D(64, (3, 3), activation = "relu", padding='same'))
        model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        model.add(Flatten())
        
        model.add(Dense(100, activation = "relu"))
        model.add(Dropout(0.5))

        model.add(Dense(50, activation = "relu"))
        model.add(Dropout(0.5))

        model.add(Dense(20, activation = "relu"))
        model.add(Dropout(0.5))

        model.add(Dense(8)) # eight actions
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])
        return model

    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def train(self):
        # makes sure that there is enought information in memory
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        """ 
        current state
        """
        current_states = np.array([transition[0] for transition in minibatch])/255
        with self.graph.as_default():
            current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)

        """
        future states
        """
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        with self.graph.as_default():
            future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE)

        X = []
        y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        """
        update tensorboard
        """
        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step


        with self.graph.as_default():
            self.model.fit(np.array(X)/255, 
            np.array(y), 
            batch_size=TRAINING_BATCH_SIZE, 
            verbose=0, 
            shuffle=False, 
            callbacks=[self.tensorboard] if log_this_step else None)


        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

    def train_in_loop(self):
        """
        X = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)
        y = np.random.uniform(size=(0, 8)).astype(np.float32)
        with self.graph.as_default():
            self.model.fit(X,y, verbose=False, batch_size=1)
        """       
        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            #time.sleep(0.01)

if __name__ == '__main__':

    argparser = argparse.ArgumentParser(
        description='CARLA Reinforment Learning Dissertation 2020')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='640x480',
        help='window resolution (default: 640x480)')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    FPS = 60
    # For stats
    ep_rewards = [0]

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1)

    # Memory fraction, used mostly when trai8ning multiple agents
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)))

    # Create models folder
    if not os.path.isdir('models'):
        os.makedirs('models')

    # Create agent and environment
    agent = Agent()
    env = Environment()
    
    # Start training thread and wait for training to be initialized
    trainer_thread = Thread(target = agent.train_in_loop, daemon=True)
    trainer_thread.start()
    while not agent.training_initialized:
        time.sleep(0.01)

    # Initialize predictions - forst prediction takes longer as of initialization that has to be done
    # It's better to do a first prediction then before we start iterating over episode steps
    agent.get_qs(np.ones((480, 640, 3)))

    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), unit='episodes'):
        env.collision_hist = []

        # Update tensorboard step every episode
        agent.tensorboard.step = episode

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1

        # Reset environment and get initial state
        current_state = env.reset()

        # Reset flag and start iterating until episode ends
        done = False
        episode_start = time.time()

        # Play for given number of seconds only
        while True:

            # This part stays mostly the same, the change is to query a model for Q values
            if np.random.random() > epsilon:
                # Get action from Q table
                action = np.argmax(agent.get_qs(current_state))
            else:
                # Get random action
                action = np.random.randint(low = 0, high = 8, size = 1)
                # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
                time.sleep(1/FPS)
            new_state, reward, done, _ = env.step(action)

            # Transform new continous state to new discrete state and count reward
            episode_reward += reward

            # Every step we update replay memory
            agent.update_replay_memory((current_state, action, reward, new_state, done))
            step += 1
            #current_state = new_state
            

            if done:
                break
    
        # End of episode - destroy agents
        for actor in env.actor_list:
            actor.destroy()

        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

            # Save model, but only when min reward is greater or equal a set value
            if min_reward >= MIN_REWARD:
                agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)


    # Set termination flag for training thread and wait for it to finish
    agent.terminate = True
    trainer_thread.join()
    agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
""" # ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world, args.autopilot)

        clock = pygame.time.Clock()
        while True:
            clock.tick_busy_loop(60)
            if controller.parse_events(client, world, clock):
                return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

    finally:

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()


# main funtion where the arguments are set


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Reinforment Learning Dissertation 2020')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='640x480',
        help='window resolution (default: 640x480)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
env

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == "__main__":
    main()
 """