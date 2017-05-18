#!/usr/bin/env python
import os
import sys
import pip

os.makedirs("pip-temp")
pip.main(['install', '--no-cache-dir', '-t', 'pip-temp', 'networkx'])
sys.path = ['pip-temp'] + sys.path

import networkx as nx
import numpy as np
import json
import math


class Agent(object):

    def __init__(self, grid_size, target, scene_config_dir):
        self.target = target
        self.scene_config = self.scene_configuration(scene_config_dir)
        self.graph = nx.Graph()
        self.grid_size = grid_size
        self.all_points = {}
        self.heading_angles = [0.0, 90.0, 180.0, 270.0]
        self.horizon_angles = [60.0, 30.0, 0.0, 330.0]

        for p in self.grid_points():
            self.all_points[self.key_for_point(p)] = p
            self.build_graph(p)

        self.position = self.scene_config['gridPoints'][target['agentPositionIndex']]
        self.rotation = target['startingRotation']
        self.horizon = target['startingHorizon']

    def key_for_point(self, p):
        return "{x:0.3f}|{z:0.3f}".format(**p)
    
    def grid_points(self):
        return self.scene_config['gridPoints']
    
    def plan_horizons(self):
        actions = []
        horizon_step_map = {330:3, 0:2, 30:1, 60:0}
        look_diff = horizon_step_map[int(self.horizon)] - horizon_step_map[int(self.target['targetAgentHorizon'])]
        if look_diff > 0:
            for i in range(look_diff):
                actions.append(dict(action='LookDown'))
        else:
            for i in range(abs(look_diff)):
                actions.append(dict(action='LookUp'))

        return actions

    def plan_rotations(self):
        right_diff = self.target['targetAgentRotation'] - self.rotation
        if right_diff < 0:
            right_diff += 360
        right_steps = right_diff / 90

        left_diff = self.rotation - self.target['targetAgentRotation'] 
        if left_diff < 0:
            left_diff += 360
        left_steps = left_diff / 90

        actions = []
        if right_steps < left_steps:
            for i in range(int(right_steps)):
                actions.append(dict(action='RotateRight'))
        else:
            for i in range(int(left_steps)):
                actions.append(dict(action='RotateLeft'))

        return actions

    def shortest_plan(self):
        path = nx.shortest_path(self.graph, self.key_for_point(self.position), self.key_for_point(self.target['targetPosition']))
        actions = []
        assert self.all_points[path[0]] == self.position
        current_position = self.position
        for p in path[1:]:
            inv_pms = {self.key_for_point(v): k for k, v in self.move_relative_points(current_position, self.rotation).items()}
            actions.append(dict(action=inv_pms[p]))
            current_position = self.all_points[p]
        actions += self.plan_horizons()
        actions += self.plan_rotations()
        # self.visualize_points(path)
        return actions

    def build_graph(self, point):
        for p in self.grid_points():
            dist = math.sqrt(((point['x'] - p['x']) ** 2) + ((point['z'] - p['z']) ** 2))
            if dist <= (self.grid_size + 0.01) and dist > 0:
                self.graph.add_edge(self.key_for_point(point), self.key_for_point(p))

    def step(self, action):
        allowed_actions = {'MoveAhead','MoveBack', 'MoveLeft', 'MoveRight', 'RotateLeft', 'RotateRight', 'LookUp', 'LookDown'}
        assert action['action'] in allowed_actions

        getattr(self, action['action'])()


    def roll_array(self, current_value, ar, offset, wrap=True):
        a = np.array(ar)
        roll = (-1 * np.argmin(np.abs(a - current_value))) + offset
        if wrap or (roll <= 0 and roll > -1 * len(ar)):
            return np.roll(a, roll)[0]
        else:
            return current_value


    def LookUp(self):
        self.horizon = self.roll_array(self.horizon, self.horizon_angles, -1, wrap=False)

    def LookDown(self):
        self.horizon = self.roll_array(self.horizon, self.horizon_angles, 1, wrap=False)

    def RotateRight(self):
        self.rotation = self.roll_array(self.rotation, self.heading_angles, -1)

    def RotateLeft(self):
        self.rotation = self.roll_array(self.rotation, self.heading_angles, 1)

    def move_relative_points(self, position, rotation):
        
        action_orientation = {
            0:dict(x=0, z=1, action='MoveAhead'),
            90:dict(x=1, z=0, action='MoveRight'),
            180:dict(x=0, z=-1, action='MoveBack'),
            270:dict(x=-1, z=0, action='MoveLeft')
        }


        move_points = dict()

        for n in self.graph.neighbors(self.key_for_point(position)):
            point = self.all_points[n]
            x_o = round((point['x'] - position['x'])/self.grid_size)
            z_o = round((point['z'] - position['z'])/self.grid_size)
            for target_rotation, offsets in action_orientation.items():
                delta = round(rotation + target_rotation) % 360
                ao = action_orientation[delta]
                action_name = action_orientation[target_rotation]['action']
                if x_o == ao['x'] and z_o == ao['z']:
                    move_points[action_name] = point
                    break


        return move_points
    
    def MoveRelative(self, move_action):
        
        mps = self.move_relative_points(self.position, self.rotation)
        if move_action in mps:
            self.position = mps[move_action]
            print("Moving to %s" % self.position)
            return True
        else:
            return False

    def MoveAhead(self):
        self.MoveRelative('MoveAhead')

    def MoveBack(self):
        self.MoveRelative('MoveBack')
    
    def MoveRight(self):
        self.MoveRelative('MoveRight')

    def MoveLeft(self):
        self.MoveRelative('MoveLeft')

    def target_found(self):
        position = self.position
        horizon = self.horizon
        rotation = self.rotation
        target_position= self.target['targetPosition']
        target_horizon= self.target['targetAgentHorizon']
        target_rotation= self.target['targetAgentRotation']
        return abs(target_position['x'] - position['x']) < 0.01 and \
            abs(target_position['z'] - position['z']) < 0.01 and \
            round(horizon) == round(target_horizon) and \
            round(rotation) == round(target_rotation)

    def scene_configuration(self, scene_config_dir):
        with open("%s/%s.json" % (scene_config_dir, self.target['sceneName'])) as f:
            configs = json.loads(f.read())
        return configs[self.target['sceneIndex']]

    def visualize_points(self, path):
        import cv2
        points = set()
        xs = []
        zs = []

            # Follow the file as it grows
        for point in self.grid_points():
            xs.append(point['x'])
            zs.append(point['z'])
            points.add(str(point['x']) + "," + str(point['z']))

        image_width = 470
        image_height = 530
        image = np.zeros((image_height,image_width,3), np.uint8)
        if not xs:
            return

        min_x = min(xs)  - 1
        max_x = max(xs) + 1
        min_z = min(zs)  - 1
        max_z = max(zs) + 1

        all_points = {}
        for p in self.grid_points():
            x,z = p['x'], p['z']
            circle_x = round(((x - min_x)/float(max_x - min_x)) * image_width)
            z = (max_z - z) + min_z
            circle_y = round(((z - min_z)/float(max_z - min_z)) * image_height)
            all_points[self.key_for_point(p)] = (circle_x, circle_y)
            cv2.circle(image, (circle_x, circle_y), 5, (0, 255,0), -1 )
        
        for i, k in enumerate(path):
            end_x, end_y = all_points[k]
            if i > 0:
                start_x, start_y = all_points[path[i - 1]]
                cv2.line(image, (start_x, start_y), (end_x, end_y), (0, 255,255), 2 )
            if i == 0:
                cv2.circle(image, (end_x, end_y), 5, (255, 0,0), -1 )
            elif i == len(path) - 1:
                cv2.circle(image, (end_x, end_y), 5, (0, 0,255), -1 )

        cv2.imshow('aoeu', image)
        cv2.waitKey(0)

if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    submit_dir = os.path.join(input_dir, 'res') 
    truth_dir = os.path.join(input_dir, 'ref')
    if not os.path.isdir(submit_dir):
        print("%s doesn't exist" % submit_dir)

    if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    targets = []
    with open(os.path.join(truth_dir, "thor-challenge-targets/targets-val.json")) as f:
        targets = json.loads(f.read())

    with open(os.path.join(submit_dir, "submission.json")) as f:
        submission = json.loads(f.read())
        total_steps = 0
        for t in targets:
            if t['uuid'] in submission:
                agent = Agent(0.25, t, scene_config_dir=os.path.join(truth_dir, 'scene_configurations'))
                total_steps += len(submission[t['uuid']])
                for step in submission[t['uuid']]:
                    agent.step(step)
                print("target found %s" % agent.target_found())

    output_filename = os.path.join(output_dir, 'scores.txt')              
    with open(output_filename, 'w') as of:
        of.write("Difference: %f" % total_steps)
