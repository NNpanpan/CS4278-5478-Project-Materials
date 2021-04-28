from modular.lane_detection.lane import inspect_box, detect_lane, has_red_bar, is_pit
from modular.sign_recognition.detect_stop_sign import stop_detect

import numpy as np
import os
import argparse

class Controller:
    def __init__(self, has_pit=False, has_intersection=False, has_stop_sign=False):
        self.turning = 0
        self.remaining_steps_of_slow = 0
        self.buffer = [] # Actions can be decided in advanced
        self.turn_left_incoming = False
        self.forwarding = False
        self.alt = True
        self.middle = False
        self.intersection = has_intersection
        self.stop_sign = has_stop_sign
        self.seeing_stop = False
        self.has_pit = has_pit
        self.passing_intersection = False

        self.locating_turns = 1

        self.turn_right_incoming = False
        self.turning_right = False
        self.turn_right_prep = -1
        self.turn_right_prep_angles = [0, -1, 0.5, 1]

    def load_actions(self, actions):
        for (speed, steer) in actions:
            self.buffer.append([speed, steer])

        self.buffer.reverse()
        self.middle = True

    def stop_speed(self, speed):
        if self.remaining_steps_of_slow <= 0:
            return speed
        self.remaining_steps_of_slow -= 1
        return 0.1

    def detect_stop(self, obs):
        has_stop = stop_detect(obs)
        print("Stop detection: ", has_stop)

        # not seeing stop sign on the way
        # just set
        if not self.seeing_stop:
            self.seeing_stop = has_stop

        # first time seeing the stop sign
        if has_stop:
            self.seeing_stop = True
            self.remaining_steps_of_slow = 0
            return

        if self.seeing_stop == True:
            # When the stop sign is just out of the way,
            # slow down for 200 steps
            self.seeing_stop = False
            self.remaining_steps_of_slow = 200

    def predict(self, rgb_array=None, raw_obs=None):
        print("Status ",self.turn_left_incoming, self.turn_right_incoming, self.middle)

        if rgb_array is None:
            return [0, 0]

        image = np.array(rgb_array)

        if self.buffer != []:
            action = self.buffer.pop()
            if self.stop_sign and raw_obs is not None:
                self.detect_stop(raw_obs)
            return action

        # For maps with pits
        if self.has_pit and is_pit(rgb_array, 100, 200, 699, 399) > 0.9:
            self.buffer = [[0.1, 1.0]] * 62
            self.middle = False
            return [0.1, 1.0]

        # For maps with stop signs
        if self.stop_sign and raw_obs is not None:
            self.detect_stop(raw_obs)

        lines, pos_slope_avg, neg_slope_avg = detect_lane(rgb_array=image)
        print("Slopes ", pos_slope_avg, neg_slope_avg)

        if lines is []:
            return [1.0, 1]

        if pos_slope_avg is None:
            psa = 0
        else:
            psa = pos_slope_avg
        
        if neg_slope_avg is None:
            nsa = 0
        else:
            nsa = neg_slope_avg

        steerer = abs(nsa) - psa
        if steerer < 0:
            steerer = max(steerer, -1)
        if steerer > 0:
            steerer = min(steerer, 1)
        # if nsa != 0 and psa != 0:
        #     steerer = abs(nsa) - psa
        # else:
        #     steerer = 0

        # Bot-left
        yellow_b_l, white_b_l, max_g_b_l = inspect_box(image, 0, 300, 399, 599)

        # Bot-right
        yellow_b_r, white_b_r, max_g_b_r = inspect_box(image, 400, 300, 799, 599)

        # Bottom stripe
        yellow_b_s, white_b_s, max_g_b_s = inspect_box(image, 200, 500, 599, 599)

        print("Yellows ", yellow_b_l, yellow_b_s, yellow_b_r)
        print("Whites ", white_b_l, white_b_s, white_b_r)
        print("Max green vals ", max_g_b_l, max_g_b_s, max_g_b_r)

        

        # Big assumption of intersection: Always have either left or forward
        if self.passing_intersection:
            if white_b_l > 0.01 and white_b_r > 0.01:
                # approaching the sideway, have to turn left
                if self.remaining_steps_of_slow > 0:
                    fwd_steps = [[0.1, 0]] * 50
                    buf_steps = [[0.1, 0]] * 10
                    self.remaining_steps_of_slow = 10
                else:
                    fwd_steps = [[0.44, 0]] * 16
                    buf_steps = [[0.44, 0]] * 3
                self.buffer = fwd_steps + [[0.1, 1]] * 30 + buf_steps
                self.passing_intersection = False
                return self.buffer.pop()

            # okay, can go forward
            if self.remaining_steps_of_slow > 0:
                fwd_steps = [[0.1, 0]] * 50
                self.remaining_steps_of_slow = 0
            else:
                fwd_steps = [[0.44, 0]] * 16
            self.buffer = fwd_steps
            self.passing_intersection = False

            return self.buffer.pop()

        # Find lanes
        if not self.middle:
            #     return [-self.stop_speed(0.44), 0]
            # if self.intersection:
            #     if white_b_l < 0.001:
            #         if white_b_r < 0.001:
            #             self.buffer = [[0.1, 0], [0.1, -1], [0.1, -1]] * 8
            #             return self.buffer.pop()

            #     if white_b_l > 0.01:
            #         if white_b_r < 0.001:
            #             # stuck to the left side...
            #             self.buffer = [[0.1, 0], [0.1, -1], [0.1, -1]] * 13
            #             return self.buffer.pop()

            if yellow_b_l > 0.02 and white_b_l < 0.0001 and white_b_r > 0.0001:
                # consider the middle boy
                if yellow_b_s > 0.001:
                    # oof
                    # if psa >= 1:
                    #     self.buffer = [[]]
                    if psa > 0.25:
                        self.buffer = [[0.1, 0], [0.1, -1], [0.1, -1]] * 8 + [[-self.stop_speed(0.35), 0]]
                    else:
                        self.buffer = [[0.1, 0], [0.1, -1], [0.1, -1]] * 10 + [[-self.stop_speed(0.35), 0]]
                    return self.buffer.pop()

                # between 2 lanes alr
                self.middle = True
                if steerer > 0:
                    return [self.stop_speed(0.44), 1]
                else:
                    return [self.stop_speed(0.44), -1]
            elif yellow_b_l > 0.02 and white_b_l < 0.0001 and white_b_r <= 0.0001:
                if psa > 0.5:
                    # between 2 lanes, a bit close to left lane
                    # self.middle = True
                    return [self.stop_speed(0.44), steerer]
                else:
                    # slope too small
                    if yellow_b_s < 0.1:
                        return [0.1, 1]
                    elif psa < 0.1:
                        # self.middle = True
                        self.buffer = [[0.1, 0], [0.1, -1], [0.1, -1]] * 16
                        return self.buffer.pop()
                    elif psa < 0.2:
                        # self.middle = True
                        self.buffer = [[0.1, 0], [0.1, -1], [0.1, -1]] * 8
                        return self.buffer.pop()
                    else:
                        self.buffer = [[0.1, 0], [0.1, -1], [0.1, -1]] * 4
                        return self.buffer.pop()
                        
                    # self.buffer = [[0.35, -1]] *20 + [[0.44, 0]] * 5
                    # return self.buffer.pop()
            elif yellow_b_l > 0.02 and white_b_r > 0.02 and yellow_b_r > 0.02:
                # definitely grass on the right
                return [self.stop_speed(0.44), 1]

            elif yellow_b_l <= 0.02:
                if yellow_b_r > 0.02 and white_b_r > 0.02:
                    if white_b_l < 0.02:
                        # prolly the grass
                        # turn to the left to find out more
                        return [0.1, 1]
                    else:
                        # kinda freakish
                        # let's backdown
                        return [-self.stop_speed(0.44), 0]

                if white_b_l > 0.02 and white_b_r > 0.02:
                    # cheeky
                    # in front could be the grass
                    # but it could also be facing the edge
                    # false edges may be detected, so let's back down a bit
                    return [-self.stop_speed(0.44), 0]  

                if white_b_r > 0.02:
                    return [self.stop_speed(0.44), 1]
                
                if white_b_l > 0.02 and yellow_b_r > 0.02:
                    # oh boy, we're reverse
                    # let's turn around madly
                    # welp, based on the slopes!
                    if pos_slope_avg is not None and pos_slope_avg < 0.6:
                        # positive slope is quite low
                        # let's back down once for safety and turn right
                        self.buffer = [[0.1, -1]] * 20 + [[-self.stop_speed(0.44), 0]]
                        return [-self.stop_speed(0.44), 0]
                    
                    # By default, just turn right madly, should be safe

                    self.buffer = [[0.1, -1]] * 19
                    return [0.1, -1]

                if white_b_l > 0.02 and yellow_b_r <= 0.02:
                    # it's kinda hard to tell
                    # if there's a lot of white on the left, then it shouldn't be the yellow lane
                    # just turn right to identify where the yellow lane is
                    return [0.1, -1]

                return [0.1, 1]
            else: # plenty of yellow on the left
                # If there's lots of white-gray on the left, it's the grass
                # If not, highly likely that it's the yellow lane
                # Anyhow, turn right to find more information

                return [0.1, -1]

        if self.intersection:
            red_bar, red_bar_pos = has_red_bar(image)

            if red_bar and red_bar_pos == 'top':
                self.turn_right_incoming = self.turn_left_incoming = False
                return [self.stop_speed(0.44), steerer]

            if red_bar and red_bar_pos == 'mid':
                self.turn_right_incoming = self.turn_left_incoming = False
                return [self.stop_speed(0.44), steerer]

            if red_bar and red_bar_pos == 'bot':
                self.turn_right_incoming = self.turn_left_incoming = False
                self.middle = False
                self.passing_intersection = True
                
                if self.remaining_steps_of_slow > 0:
                    self.buffer = [[0.1, 0]] * 50
                    return [0.1, 0]
                else:
                    self.buffer = [[self.stop_speed(0.44), 0]] * 16
                    return [self.stop_speed(0.44), steerer]

        # On track to turn left
        if self.turn_left_incoming:
            if yellow_b_l <= 0.02:
                return [self.stop_speed(0.35), 1]

            if (neg_slope_avg is not None and neg_slope_avg <= -0.6) \
                or pos_slope_avg is not None:
                # If we turn left, eventually the bot will find a high enuf slope on the right side
                # or we see the positive slopes!
                self.turn_left_incoming = False
                # self.middle = False

            return [self.stop_speed(0.44), 0]

        # On track to turn right
        # Only start to turn once you can't see the yellow lane
        if self.turn_right_incoming:
            if yellow_b_l < 0.02 and yellow_b_s < 0.1:
                # something's wrong
                self.turn_right_incoming = False
                return [self.stop_speed(0.35), 1]

            if yellow_b_s < 0.1 or white_b_r > 0.01:
                # still seeing lots of yellow lane to pass thru
                return [self.stop_speed(0.44), 0]

            if psa > 0 and psa < 0.25:
                self.buffer = [[0.1, 0], [0.1, -1], [0.1, -1]] * 16
            else:
                self.buffer = [[0.1, 0], [0.1, -1], [0.1, -1]] * 8

            # self.buffer = [[0.1, 0], [0.1, -1], [0.1, -1]] * 16
            self.turn_right_incoming = False
            # self.middle = False

            return self.buffer.pop()

            # if self.turning_right:
            #     if yellow_b_r < 0.01 and ((pos_slope_avg is not None and pos_slope_avg >= 0.65) \
            #         or neg_slope_avg is not None):
            #         # If we turn right, eventually the bot will find a high enuf slope on the right side
            #         # or we see the negative slopes!
            #         self.turn_right_incoming = self.turning_right = False
            #         self.turn_right_prep = -1
            #         # self.middle = False

            #     if  yellow_b_s < 0.1 and abs(yellow_b_r - yellow_b_l) < 0.01:
            #         self.buffer = [[0.1, -1]] * 20
            #         return self.buffer.pop()

            #     if self.alt:
            #         self.alt = False
            #         # return [self.stop_speed(0.35), -1]
            #         return [0.35, -1]
            #     else:
            #         self.alt = True
            #         return [0.1, 0]


            # if white_b_r > 0.01 or (yellow_b_l > 0.01 and yellow_b_r < 0.03):
            #     if self.turn_right_prep < 0:
            #         self.turn_right_prep = 3
            #     st = self.turn_right_prep_angles[self.turn_right_prep]
            #     self.turn_right_prep -= 1
            #     return [self.stop_speed(0.44), st]

            # self.turning_right = True
            # return [0.35, -1]

            # if white_b_r <= 0.01:
            #     # A bit of hack here
            #     # Turning right is easy to 'die' so alternate btw turning and step slightly forward
            #     if yellow_b_l > 0.015 and yellow_b_r > 0.015 \
            #         and max_g_b_l >= max_g_b_r:
            #         # return [-0.35, -1]
            #         return [0.1, -1]

            #     # if max_g_b_r > 177:
            #     #     return [self.stop_speed(0.44), 1]
                
            #     if self.alt:
            #         self.alt = False
            #         # return [self.stop_speed(0.35), -1]
            #         return [0.35, -1]
            #     else:
            #         self.alt = True
            #         return [0.1, 0]

            # if (pos_slope_avg is not None and pos_slope_avg >= 0.65) \
            #     or neg_slope_avg is not None:
            #     # If we turn right, eventually the bot will find a high enuf slope on the right side
            #     # or we see the negative slopes!
            #     self.turn_right_incoming = False
            
            # return [self.stop_speed(0.44), 0]

        # Duckie the bot is in the middle of 2 lanes
        # Duckie is not expecting any turns
        # Expecting standard situations
        if pos_slope_avg is not None and neg_slope_avg is not None:
            if pos_slope_avg >= 0.65 and neg_slope_avg <= -0.6:
                return [self.stop_speed(0.8), steerer]

            if pos_slope_avg < 0.65 and neg_slope_avg <= -0.6 :
                # Found a right curve here
                # Believe Duckie will turn right in the future
                self.turn_right_incoming = True

            if neg_slope_avg > -0.6:
                # Found a left curve here
                # Believe Duckie will turn left in the future
                self.turn_left_incoming = True

            if yellow_b_l < 0.02:
                return [self.stop_speed(0.8), 1]
            
            if white_b_r < 0.02:
                return [self.stop_speed(0.8), -1]
            
            return [self.stop_speed(0.8), 0]
        
        if pos_slope_avg is not None:
            if pos_slope_avg < 0.5 and neg_slope_avg is None:
                # Found a right curve here
                # Believe Duckie will turn right in the future
                self.turn_right_incoming = True

            if white_b_r > 0.02:
                return [self.stop_speed(0.8), steerer]
            else:
                return [self.stop_speed(0.44), steerer]

        if neg_slope_avg is not None:
            if neg_slope_avg > -0.6:
                # Found a left curve here
                # Believe Duckie will turn left in the future
                self.turn_left_incoming = True

            if white_b_r > 0.02:
                return [self.stop_speed(0.8), steerer]
            else:
                return [self.stop_speed(0.44), steerer]


        return [self.stop_speed(0.44), steerer]

    def predict3(self, rgb_array=None, raw_obs=None):
        print("Status ",self.turn_left_incoming, self.turn_right_incoming, self.middle)

        if rgb_array is None:
            return [0, 0]

        image = np.array(rgb_array)

        if self.buffer != []:
            action = self.buffer.pop()
            if self.stop_sign and raw_obs is not None:
                self.detect_stop(raw_obs)
            return action

        # For maps with pits
        if self.has_pit and is_pit(rgb_array, 100, 200, 699, 399) > 0.9:
            self.buffer = [[0.1, 1.0]] * 62
            self.middle = False
            return [0.1, 1.0]

        # For maps with stop signs
        if self.stop_sign and raw_obs is not None:
            self.detect_stop(raw_obs)

        lines, pos_slope_avg, neg_slope_avg = detect_lane(rgb_array=image)
        print("Slopes ", pos_slope_avg, neg_slope_avg)

        if lines is []:
            return [1.0, 1]

        if pos_slope_avg is None:
            psa = 0
        else:
            psa = pos_slope_avg
        
        if neg_slope_avg is None:
            nsa = 0
        else:
            nsa = neg_slope_avg

        steerer = abs(nsa) - psa
        if steerer < 0:
            steerer = max(steerer, -1)
        if steerer > 0:
            steerer = min(steerer, 1)
        # if nsa != 0 and psa != 0:
        #     steerer = abs(nsa) - psa
        # else:
        #     steerer = 0

        # Bot-left
        yellow_b_l, white_b_l, max_g_b_l = inspect_box(image, 0, 300, 399, 599)

        # Bot-right
        yellow_b_r, white_b_r, max_g_b_r = inspect_box(image, 400, 300, 799, 599)

        # Bottom stripe
        yellow_b_s, white_b_s, max_g_b_s = inspect_box(image, 200, 500, 599, 599)

        print("Yellows ", yellow_b_l, yellow_b_s, yellow_b_r)
        print("Whites ", white_b_l, white_b_s, white_b_r)
        print("Max green vals ", max_g_b_l, max_g_b_s, max_g_b_r)

        

        # Big assumption of intersection: Always have either left or forward
        if self.passing_intersection:
            if white_b_l > 0.01 and white_b_r > 0.01:
                # approaching the sideway, have to turn left
                if self.remaining_steps_of_slow > 0:
                    fwd_steps = [[0.1, 0]] * 50
                    buf_steps = [[0.1, 0]] * 10
                    self.remaining_steps_of_slow = 10
                else:
                    fwd_steps = [[0.44, 0]] * 16
                    buf_steps = [[0.44, 0]] * 3
                self.buffer = fwd_steps + [[0.1, 1]] * 30 + buf_steps
                self.passing_intersection = False
                return self.buffer.pop()

            # okay, can go forward
            if self.remaining_steps_of_slow > 0:
                fwd_steps = [[0.1, 0]] * 50
                self.remaining_steps_of_slow = 0
            else:
                fwd_steps = [[0.44, 0]] * 16
            self.buffer = fwd_steps
            self.passing_intersection = False

            return self.buffer.pop()

        # Find lanes
        if not self.middle:
            #     return [-self.stop_speed(0.44), 0]
            # if self.intersection:
            #     if white_b_l < 0.001:
            #         if white_b_r < 0.001:
            #             self.buffer = [[0.1, 0], [0.1, -1], [0.1, -1]] * 8
            #             return self.buffer.pop()

            #     if white_b_l > 0.01:
            #         if white_b_r < 0.001:
            #             # stuck to the left side...
            #             self.buffer = [[0.1, 0], [0.1, -1], [0.1, -1]] * 13
            #             return self.buffer.pop()

            if yellow_b_l > 0.02 and white_b_l < 0.0001 and white_b_r > 0.0001:
                # consider the middle boy
                if yellow_b_s > 0.001:
                    # oof
                    # if psa >= 1:
                    #     self.buffer = [[]]
                    if psa > 0.25:
                        self.buffer = [[0.1, 0], [0.1, -1], [0.1, -1]] * 8 + [[-self.stop_speed(0.35), 0]]
                    else:
                        self.buffer = [[0.1, 0], [0.1, -1], [0.1, -1]] * 10 + [[-self.stop_speed(0.35), 0]]
                    return self.buffer.pop()

                # between 2 lanes alr
                self.middle = True
                if steerer > 0:
                    return [self.stop_speed(0.44), 1]
                else:
                    return [self.stop_speed(0.44), -1]
            elif yellow_b_l > 0.02 and white_b_l < 0.0001 and white_b_r <= 0.0001:
                if psa > 0.5:
                    # between 2 lanes, a bit close to left lane
                    # self.middle = True
                    return [self.stop_speed(0.44), steerer]
                else:
                    # slope too small
                    if yellow_b_s < 0.1:
                        return [0.1, 1]
                    elif psa > 0 and psa < 0.1:
                        # self.middle = True
                        self.buffer = [[0.1, 0], [0.1, -1], [0.1, -1]] * 16
                        return self.buffer.pop()
                    elif psa > 0 and psa < 0.2:
                        # self.middle = True
                        self.buffer = [[0.1, 0], [0.1, -1], [0.1, -1]] * 8
                        return self.buffer.pop()
                    elif psa > 0:
                        self.buffer = [[0.1, 0], [0.1, -1], [0.1, -1]] * 4
                        return self.buffer.pop()
                        
                    return [0.1, -1]
                    # self.buffer = [[0.35, -1]] *20 + [[0.44, 0]] * 5
                    # return self.buffer.pop()
            elif yellow_b_l > 0.02 and white_b_r > 0.02 and yellow_b_r > 0.02:
                # definitely grass on the right
                return [self.stop_speed(0.44), 1]

            elif yellow_b_l <= 0.02:
                if yellow_b_r > 0.02 and white_b_r > 0.02:
                    if white_b_l < 0.02:
                        # prolly the grass
                        # turn to the left to find out more
                        return [0.1, 1]
                    else:
                        # kinda freakish
                        # let's backdown
                        return [-self.stop_speed(0.44), 0]

                if white_b_l > 0.02 and white_b_r > 0.02:
                    # cheeky
                    # in front could be the grass
                    # but it could also be facing the edge
                    # false edges may be detected, so let's back down a bit
                    return [-self.stop_speed(0.44), 0]  

                if white_b_r > 0.02:
                    return [self.stop_speed(0.44), 1]
                
                if white_b_l > 0.02 and yellow_b_r > 0.02:
                    # oh boy, we're reverse
                    # let's turn around madly
                    # welp, based on the slopes!
                    if pos_slope_avg is not None and pos_slope_avg < 0.6:
                        # positive slope is quite low
                        # let's back down once for safety and turn right
                        self.buffer = [[0.1, -1]] * 20 + [[-self.stop_speed(0.44), 0]]
                        return [-self.stop_speed(0.44), 0]
                    
                    # By default, just turn right madly, should be safe

                    self.buffer = [[0.1, -1]] * 19
                    return [0.1, -1]

                if white_b_l > 0.02 and yellow_b_r <= 0.02:
                    # it's kinda hard to tell
                    # if there's a lot of white on the left, then it shouldn't be the yellow lane
                    # just turn right to identify where the yellow lane is
                    return [0.1, -1]

                return [0.1, 1]
            else: # plenty of yellow on the left
                # If there's lots of white-gray on the left, it's the grass
                # If not, highly likely that it's the yellow lane
                # Anyhow, turn right to find more information

                return [0.1, -1]

        if self.intersection:
            red_bar, red_bar_pos = has_red_bar(image)

            if red_bar and red_bar_pos == 'top':
                self.turn_right_incoming = self.turn_left_incoming = False
                return [self.stop_speed(0.44), steerer]

            if red_bar and red_bar_pos == 'mid':
                self.turn_right_incoming = self.turn_left_incoming = False
                return [self.stop_speed(0.44), steerer]

            if red_bar and red_bar_pos == 'bot':
                self.turn_right_incoming = self.turn_left_incoming = False
                self.middle = False
                self.passing_intersection = True
                
                if self.remaining_steps_of_slow > 0:
                    self.buffer = [[0.1, 0]] * 100
                    return [0.1, 0]
                else:
                    self.buffer = [[self.stop_speed(0.44), 0]] * 16
                    return [self.stop_speed(0.44), steerer]

        # On track to turn left
        if self.turn_left_incoming:
            if yellow_b_r > 0.001:
                # sth wrong
                self.turn_left_incoming = False
                return [0.1, -1]

            if white_b_l > 0.01:
                return [0.1, 1]

            if yellow_b_l <= 0.02:
                return [self.stop_speed(0.35), 1]

            if (neg_slope_avg is not None and neg_slope_avg <= -0.6) \
                or pos_slope_avg is not None:
                # If we turn left, eventually the bot will find a high enuf slope on the right side
                # or we see the positive slopes!
                self.turn_left_incoming = False
                # self.middle = False

            return [self.stop_speed(0.44), 0]

        # On track to turn right
        # Only start to turn once you can't see the yellow lane
        if self.turn_right_incoming:
            if yellow_b_l < 0.02 and yellow_b_s < 0.1:
                # something's wrong
                self.turn_right_incoming = False
                return [self.stop_speed(0.35), 1]

            if yellow_b_s < 0.15 or white_b_r > 0.01:
                # still seeing lots of yellow lane to pass thru
                return [self.stop_speed(0.44), 0]

            if psa > 0 and psa < 0.2:
                self.buffer = [[0.1, 0], [0.1, -1], [0.1, -1]] * 16
            else:
                self.buffer = [[0.1, 0], [0.1, -1], [0.1, -1]] * 8

            # self.buffer = [[0.1, 0], [0.1, -1], [0.1, -1]] * 16
            self.turn_right_incoming = False
            # self.middle = False

            return self.buffer.pop()

            # if self.turning_right:
            #     if yellow_b_r < 0.01 and ((pos_slope_avg is not None and pos_slope_avg >= 0.65) \
            #         or neg_slope_avg is not None):
            #         # If we turn right, eventually the bot will find a high enuf slope on the right side
            #         # or we see the negative slopes!
            #         self.turn_right_incoming = self.turning_right = False
            #         self.turn_right_prep = -1
            #         # self.middle = False

            #     if  yellow_b_s < 0.1 and abs(yellow_b_r - yellow_b_l) < 0.01:
            #         self.buffer = [[0.1, -1]] * 20
            #         return self.buffer.pop()

            #     if self.alt:
            #         self.alt = False
            #         # return [self.stop_speed(0.35), -1]
            #         return [0.35, -1]
            #     else:
            #         self.alt = True
            #         return [0.1, 0]


            # if white_b_r > 0.01 or (yellow_b_l > 0.01 and yellow_b_r < 0.03):
            #     if self.turn_right_prep < 0:
            #         self.turn_right_prep = 3
            #     st = self.turn_right_prep_angles[self.turn_right_prep]
            #     self.turn_right_prep -= 1
            #     return [self.stop_speed(0.44), st]

            # self.turning_right = True
            # return [0.35, -1]

            # if white_b_r <= 0.01:
            #     # A bit of hack here
            #     # Turning right is easy to 'die' so alternate btw turning and step slightly forward
            #     if yellow_b_l > 0.015 and yellow_b_r > 0.015 \
            #         and max_g_b_l >= max_g_b_r:
            #         # return [-0.35, -1]
            #         return [0.1, -1]

            #     # if max_g_b_r > 177:
            #     #     return [self.stop_speed(0.44), 1]
                
            #     if self.alt:
            #         self.alt = False
            #         # return [self.stop_speed(0.35), -1]
            #         return [0.35, -1]
            #     else:
            #         self.alt = True
            #         return [0.1, 0]

            # if (pos_slope_avg is not None and pos_slope_avg >= 0.65) \
            #     or neg_slope_avg is not None:
            #     # If we turn right, eventually the bot will find a high enuf slope on the right side
            #     # or we see the negative slopes!
            #     self.turn_right_incoming = False
            
            # return [self.stop_speed(0.44), 0]

        # Duckie the bot is in the middle of 2 lanes
        # Duckie is not expecting any turns
        # Expecting standard situations
        if pos_slope_avg is not None and neg_slope_avg is not None:
            if pos_slope_avg >= 0.65 and neg_slope_avg <= -0.6:
                return [self.stop_speed(0.8), steerer]

            if pos_slope_avg < 0.65 and neg_slope_avg <= -0.6 :
                # Found a right curve here
                # Believe Duckie will turn right in the future
                self.turn_right_incoming = True

            if neg_slope_avg > -0.6:
                # Found a left curve here
                # Believe Duckie will turn left in the future
                self.turn_left_incoming = True

            if yellow_b_l < 0.02:
                return [self.stop_speed(0.8), 1]
            
            if white_b_r < 0.02:
                return [self.stop_speed(0.8), -1]
            
            return [self.stop_speed(0.8), 0]
        
        if pos_slope_avg is not None:
            if pos_slope_avg < 0.5 and neg_slope_avg is None:
                # Found a right curve here
                # Believe Duckie will turn right in the future
                self.turn_right_incoming = True

            if white_b_r > 0.02:
                return [self.stop_speed(0.8), steerer]
            else:
                return [self.stop_speed(0.44), steerer]

        if neg_slope_avg is not None:
            if neg_slope_avg > -0.6:
                # Found a left curve here
                # Believe Duckie will turn left in the future
                self.turn_left_incoming = True

            if white_b_r > 0.02:
                return [self.stop_speed(0.8), steerer]
            else:
                return [self.stop_speed(0.44), steerer]


        return [self.stop_speed(0.44), steerer]

    def predict5(self, rgb_array=None, raw_obs=None):
        print("Status ",self.turn_left_incoming, self.turn_right_incoming, self.middle)

        if rgb_array is None:
            return [0, 0]

        image = np.array(rgb_array)

        if self.buffer != []:
            action = self.buffer.pop()
            if self.stop_sign and raw_obs is not None:
                self.detect_stop(raw_obs)
            return action

        # For maps with pits
        if self.has_pit and is_pit(rgb_array, 100, 200, 699, 399) > 0.9:
            self.buffer = [[0.1, 1.0]] * 62
            self.middle = False
            return [0.1, 1.0]

        # For maps with stop signs
        if self.stop_sign and raw_obs is not None:
            self.detect_stop(raw_obs)

        lines, pos_slope_avg, neg_slope_avg = detect_lane(rgb_array=image)
        print("Slopes ", pos_slope_avg, neg_slope_avg)

        if lines is []:
            return [1.0, 1]

        if pos_slope_avg is None:
            psa = 0
        else:
            psa = pos_slope_avg
        
        if neg_slope_avg is None:
            nsa = 0
        else:
            nsa = neg_slope_avg

        steerer = abs(nsa) - psa
        if steerer < 0:
            steerer = max(steerer, -1)
        if steerer > 0:
            steerer = min(steerer, 1)
        # if nsa != 0 and psa != 0:
        #     steerer = abs(nsa) - psa
        # else:
        #     steerer = 0

        # Bot-left
        yellow_b_l, white_b_l, max_g_b_l = inspect_box(image, 0, 300, 399, 599)

        # Bot-right
        yellow_b_r, white_b_r, max_g_b_r = inspect_box(image, 400, 300, 799, 599)

        # Bottom stripe
        yellow_b_s, white_b_s, max_g_b_s = inspect_box(image, 200, 500, 599, 599)

        print("Yellows ", yellow_b_l, yellow_b_s, yellow_b_r)
        print("Whites ", white_b_l, white_b_s, white_b_r)
        print("Max green vals ", max_g_b_l, max_g_b_s, max_g_b_r)

        

        # Big assumption of intersection: Always have either left or forward
        if self.passing_intersection:
            if white_b_l > 0.01 and white_b_r > 0.01:
                # approaching the sideway, have to turn left
                if self.remaining_steps_of_slow > 0:
                    fwd_steps = [[0.1, 0]] * 50
                    buf_steps = [[0.1, 0]] * 10
                    self.remaining_steps_of_slow = 10
                else:
                    fwd_steps = [[0.44, 0]] * 16
                    buf_steps = [[0.44, 0]] * 3
                self.buffer = fwd_steps + [[0.1, 1]] * 30 + buf_steps
                self.passing_intersection = False
                return self.buffer.pop()

            # okay, can go forward
            if self.remaining_steps_of_slow > 0:
                fwd_steps = [[0.1, 0]] * 50
                self.remaining_steps_of_slow = 0
            else:
                fwd_steps = [[0.44, 0]] * 16
            self.buffer = fwd_steps
            self.passing_intersection = False

            return self.buffer.pop()

        # Find lanes
        if not self.middle:
            #     return [-self.stop_speed(0.44), 0]
            # if self.intersection:
            #     if white_b_l < 0.001:
            #         if white_b_r < 0.001:
            #             self.buffer = [[0.1, 0], [0.1, -1], [0.1, -1]] * 8
            #             return self.buffer.pop()

            #     if white_b_l > 0.01:
            #         if white_b_r < 0.001:
            #             # stuck to the left side...
            #             self.buffer = [[0.1, 0], [0.1, -1], [0.1, -1]] * 13
            #             return self.buffer.pop()

            if yellow_b_l > 0.02 and white_b_l < 0.0001 and white_b_r > 0.0001:
                # consider the middle boy
                if yellow_b_s > 0.001:
                    # oof
                    # if psa >= 1:
                    #     self.buffer = [[]]
                    if psa > 0.25:
                        self.buffer = [[0.1, 0], [0.1, -1], [0.1, -1]] * 8 + [[-self.stop_speed(0.35), 0]]
                    else:
                        self.buffer = [[0.1, 0], [0.1, -1], [0.1, -1]] * 10 + [[-self.stop_speed(0.35), 0]]
                    return self.buffer.pop()

                # between 2 lanes alr
                self.middle = True
                if steerer > 0:
                    return [self.stop_speed(0.44), 1]
                else:
                    return [self.stop_speed(0.44), -1]
            elif yellow_b_l > 0.02 and white_b_l < 0.0001 and white_b_r <= 0.0001:
                
                if psa > 0.5:
                    # between 2 lanes, a bit close to left lane
                    # self.middle = True
                    return [self.stop_speed(0.44), steerer]
                elif yellow_b_s < 0.1:
                    return [0.1, 0]
                elif psa < 0.1:
                    # self.middle = True
                    self.buffer = [[0.1, 0], [0.1, -1], [0.1, -1]] * 16
                    return self.buffer.pop()
                elif psa < 0.2:
                    # self.middle = True
                    self.buffer = [[0.1, 0], [0.1, -1], [0.1, -1]] * 8
                    return self.buffer.pop()
                elif nsa < 0 and nsa > -0.2:
                    self.buffer = [[0.1, 0], [0.1, 1], [0.1, 1]] * 8
                    return self.buffer.pop()
                elif nsa < 0 and nsa > -0.1:
                    self.buffer = [[0.1, 0], [0.1, 1], [0.1, 1]] * 16
                    return self.buffer.pop()
                else:
                    return [0.1, -1]
                        
                        
                # self.buffer = [[0.35, -1]] *20 + [[0.44, 0]] * 5
                # return self.buffer.pop()
            elif yellow_b_l > 0.02 and white_b_r > 0.02 and yellow_b_r > 0.02:
                # definitely grass on the right
                return [self.stop_speed(0.44), 1]

            elif yellow_b_l <= 0.02:
                # if yellow_b_r < 0.001 and yellow_b_l < 0.001:
                #     # Possibly in the right lane, just need to turn
                #     self.buffer = [[0.1, 0], [0.1, 1], [0.1, 1]] * int(20 / self.locating_turns)
                #     self.locating_turns += 1
                #     return self.buffer.pop()
                if yellow_b_r > 0.02 and white_b_r > 0.02:
                    if white_b_l < 0.02:
                        # prolly the grass
                        # turn to the left to find out more
                        return [0.1, 1]
                    else:
                        # kinda freakish
                        # let's backdown
                        return [-self.stop_speed(0.44), 0]
                
                if yellow_b_r > 0.02 and white_b_l < 0.02:
                    return [-self.stop_speed(0.44), 0]

                if white_b_l > 0.02 and white_b_r > 0.02:
                    if white_b_s < 0.01 and psa > 0 and psa < 0.25 and nsa < 0 and nsa > -0.25:
                        print(psa, nsa)
                        # in a middle of the curve, let's u-turn
                        self.buffer = [[0.1, 0], [0.1, 1], [0.1, 1]] * int(20 / self.locating_turns)
                        self.locating_turns += 1
                        return self.buffer.pop()
                    # cheeky
                    # in front could be the grass
                    # but it could also be facing the edge
                    # false edges may be detected, so let's back down a bit
                    return [-self.stop_speed(0.44), 0]  

                if white_b_r > 0.02:
                    return [self.stop_speed(0.44), 1]
                
                if white_b_l > 0.01 and yellow_b_r > 0.01:
                    # oh boy, we're reverse
                    # let's turn around madly
                    # welp, based on the slopes!
                    if pos_slope_avg is not None and pos_slope_avg < 0.1:
                        # positive slope is very low
                        # let's back down once for safety and turn right
                        self.buffer = [[0.1, -1]] * 30 + [[-self.stop_speed(0.44), 0]] * 3
                        return self.buffer.pop()
                    
                    # By default, just turn right madly, should be safe

                    self.buffer = [[0.1, -1]] * 16
                    return [0.1, -1]

                if white_b_l > 0.02 and yellow_b_r <= 0.02:
                    # it's kinda hard to tell
                    # if there's a lot of white on the left, then it shouldn't be the yellow lane
                    # just turn right to identify where the yellow lane is
                    return [0.1, -1]

                return [0.1, 1]
            else: # plenty of yellow on the left
                # If there's lots of white-gray on the left, it's the grass
                # If not, highly likely that it's the yellow lane
                # Anyhow, turn right to find more information

                return [0.1, -1]

        if self.intersection:
            red_bar, red_bar_pos = has_red_bar(image)

            if red_bar and red_bar_pos == 'top':
                self.turn_right_incoming = self.turn_left_incoming = False
                return [self.stop_speed(0.44), steerer]

            if red_bar and red_bar_pos == 'mid':
                self.turn_right_incoming = self.turn_left_incoming = False
                return [self.stop_speed(0.44), steerer]

            if red_bar and red_bar_pos == 'bot':
                self.turn_right_incoming = self.turn_left_incoming = False
                self.middle = False
                self.passing_intersection = True
                
                if self.remaining_steps_of_slow > 0:
                    self.buffer = [[0.1, 0]] * 50
                    return [0.1, 0]
                else:
                    self.buffer = [[self.stop_speed(0.44), 0]] * 16
                    return [self.stop_speed(0.44), steerer]

        # if white_b_l > 0.01 and white_b_r > 0.01 and white_b_s > 0:
        #     self.buffer = [[-self.stop_speed(0.44), 1]] * 5
        #     return [-self.stop_speed(0.44), 1]

        # On track to turn left
        if self.turn_left_incoming:
            if yellow_b_l <= 0.02:
                if yellow_b_l <= 0.001:
                    return [0.1, 1]
                return [self.stop_speed(0.35), 1]
            
            if nsa <= -0.6 or pos_slope_avg is not None:
                # If we turn left, eventually the bot will find a high enuf slope on the right side
                # or we see the positive slopes!
                self.turn_left_incoming = False
                # self.middle = False

            return [self.stop_speed(0.44), 0]

        # On track to turn right
        # Only start to turn once you can't see the yellow lane
        if self.turn_right_incoming:
            if yellow_b_l < 0.02 and yellow_b_s < 0.15:
                # something's wrong
                self.turn_right_incoming = False
                return [self.stop_speed(0.35), 1]

            if yellow_b_s < 0.15 or white_b_r > 0.01:
                # still seeing lots of yellow lane to pass thru
                return [self.stop_speed(0.44), 0]

            if psa < 0.2:
                self.buffer = [[0.1, 0], [0.1, -1], [0.1, -1]] * 16
            else:
                self.buffer = [[0.1, 0], [0.1, -1], [0.1, -1]] * 8

            # self.buffer = [[0.1, 0], [0.1, -1], [0.1, -1]] * 16
            self.turn_right_incoming = False
            # self.middle = False

            return self.buffer.pop()

        # Duckie the bot is in the middle of 2 lanes
        # Duckie is not expecting any turns
        # Expecting standard situations

        # This is to correct extreme cases
        if yellow_b_r > 0.01:
            return [0.1, -1]

        if yellow_b_l < 0.01:
            return [0.1, 1]

        if pos_slope_avg is not None and neg_slope_avg is not None:
            if pos_slope_avg >= 0.65 and neg_slope_avg <= -0.6:
                return [self.stop_speed(0.8), steerer]

            if pos_slope_avg < 0.65 and (neg_slope_avg <= -0.6 or yellow_b_l > 0.02):
                # Found a right curve here
                # Believe Duckie will turn right in the future
                self.turn_right_incoming = True

            if neg_slope_avg > -0.6 and white_b_r > 0.02:
                # Found a left curve here
                # Believe Duckie will turn left in the future
                self.turn_left_incoming = True

            if yellow_b_l < 0.02:
                return [self.stop_speed(0.44), 1]
            
            if white_b_r < 0.02:
                return [self.stop_speed(0.44), -1]
            
            return [self.stop_speed(0.44), 0]

        
        if pos_slope_avg is not None:
            if pos_slope_avg < 0.5 and neg_slope_avg is None:
                # Found a right curve here
                # Believe Duckie will turn right in the future
                self.turn_right_incoming = True

            if white_b_r > 0.02:
                return [self.stop_speed(0.8), steerer]
            else:
                return [self.stop_speed(0.44), steerer]

        if neg_slope_avg is not None:
            if neg_slope_avg > -0.6:
                # Found a left curve here
                # Believe Duckie will turn left in the future
                self.turn_left_incoming = True

            if white_b_r > 0.02:
                return [self.stop_speed(0.8), steerer]
            else:
                return [self.stop_speed(0.44), steerer]


        return [self.stop_speed(0.44), steerer]


    def predict4(self, rgb_array=None, raw_obs=None):
        print("Status ",self.turn_left_incoming, self.turn_right_incoming, self.middle)

        if rgb_array is None:
            return [0, 0]

        image = np.array(rgb_array)

        if self.buffer != []:
            action = self.buffer.pop()
            if self.stop_sign and raw_obs is not None:
                self.detect_stop(raw_obs)
            return action

        # For maps with pits
        if self.has_pit and is_pit(rgb_array, 100, 200, 699, 399) > 0.9:
            self.buffer = [[0.1, 1.0]] * 62
            self.middle = False
            return [0.1, 1.0]

        # For maps with stop signs
        if self.stop_sign and raw_obs is not None:
            self.detect_stop(raw_obs)

        lines, pos_slope_avg, neg_slope_avg = detect_lane(rgb_array=image)
        print("Slopes ", pos_slope_avg, neg_slope_avg)

        if lines is []:
            return [1.0, 1]

        if pos_slope_avg is None:
            psa = 0
        else:
            psa = pos_slope_avg
        
        if neg_slope_avg is None:
            nsa = 0
        else:
            nsa = neg_slope_avg

        steerer = abs(nsa) - psa
        if steerer < 0:
            steerer = max(steerer, -1)
        if steerer > 0:
            steerer = min(steerer, 1)
        # if nsa != 0 and psa != 0:
        #     steerer = abs(nsa) - psa
        # else:
        #     steerer = 0

        # Bot-left
        yellow_b_l, white_b_l, max_g_b_l = inspect_box(image, 0, 300, 399, 599)

        # Bot-right
        yellow_b_r, white_b_r, max_g_b_r = inspect_box(image, 400, 300, 799, 599)

        # Bottom stripe
        yellow_b_s, white_b_s, max_g_b_s = inspect_box(image, 200, 500, 599, 599)

        print("Yellows ", yellow_b_l, yellow_b_s, yellow_b_r)
        print("Whites ", white_b_l, white_b_s, white_b_r)
        print("Max green vals ", max_g_b_l, max_g_b_s, max_g_b_r)

        

        # Big assumption of intersection: Always have either left or forward
        if self.passing_intersection:
            if white_b_l > 0.01 and white_b_r > 0.01:
                # approaching the sideway, have to turn left
                if self.remaining_steps_of_slow > 0:
                    fwd_steps = [[0.1, 0]] * 50
                    buf_steps = [[0.1, 0]] * 10
                    self.remaining_steps_of_slow = 10
                else:
                    fwd_steps = [[0.44, 0]] * 16
                    buf_steps = [[0.44, 0]] * 3
                self.buffer = fwd_steps + [[0.1, 1]] * 30 + buf_steps
                self.passing_intersection = False
                return self.buffer.pop()

            # okay, can go forward
            if self.remaining_steps_of_slow > 0:
                fwd_steps = [[0.1, 0]] * 50
                self.remaining_steps_of_slow = 0
            else:
                fwd_steps = [[0.44, 0]] * 16
            self.buffer = fwd_steps
            self.passing_intersection = False

            return self.buffer.pop()

        # Find lanes
        if not self.middle:
            #     return [-self.stop_speed(0.44), 0]
            # if self.intersection:
            #     if white_b_l < 0.001:
            #         if white_b_r < 0.001:
            #             self.buffer = [[0.1, 0], [0.1, -1], [0.1, -1]] * 8
            #             return self.buffer.pop()

            #     if white_b_l > 0.01:
            #         if white_b_r < 0.001:
            #             # stuck to the left side...
            #             self.buffer = [[0.1, 0], [0.1, -1], [0.1, -1]] * 13
            #             return self.buffer.pop()

            if yellow_b_l > 0.02 and white_b_l < 0.0001 and white_b_r > 0.0001:
                # consider the middle boy
                if yellow_b_s > 0.001:
                    # oof
                    # if psa >= 1:
                    #     self.buffer = [[]]
                    if psa > 0.25:
                        self.buffer = [[0.1, 0], [0.1, -1], [0.1, -1]] * 8 + [[-self.stop_speed(0.35), 0]]
                    else:
                        self.buffer = [[0.1, 0], [0.1, -1], [0.1, -1]] * 10 + [[-self.stop_speed(0.35), 0]]
                    return self.buffer.pop()

                # between 2 lanes alr
                self.middle = True
                if steerer > 0:
                    return [self.stop_speed(0.44), 1]
                else:
                    return [self.stop_speed(0.44), -1]
            elif yellow_b_l > 0.02 and white_b_l < 0.0001 and white_b_r <= 0.0001:
                if psa > 0.5:
                    # between 2 lanes, a bit close to left lane
                    # self.middle = True
                    return [self.stop_speed(0.44), steerer]
                else:
                    # slope too small
                    # if yellow_b_s < 0.1:
                    #     return [0.1, 1]
                    if psa == 0:
                        self.buffer = [[0.1, 0], [0.1, -1], [0.1, -1]] * 4
                        return self.buffer.pop()
                    elif psa < 0.1:
                        # self.middle = True
                        self.buffer = [[0.1, 0], [0.1, -1], [0.1, -1]] * 16
                        return self.buffer.pop()
                    elif psa < 0.2:
                        # self.middle = True
                        self.buffer = [[0.1, 0], [0.1, -1], [0.1, -1]] * 8
                        return self.buffer.pop()
                    else:
                        self.buffer = [[0.1, 0], [0.1, -1], [0.1, -1]] * 4
                        return self.buffer.pop()
                        
                    # self.buffer = [[0.35, -1]] *20 + [[0.44, 0]] * 5
                    # return self.buffer.pop()
            elif yellow_b_l > 0.02 and white_b_r > 0.02 and yellow_b_r > 0.02:
                # definitely grass on the right
                return [self.stop_speed(0.44), 1]

            elif yellow_b_l <= 0.02:
                if yellow_b_r > 0.02 and white_b_r > 0.02:
                    if white_b_l < 0.02:
                        # prolly the grass
                        # turn to the left to find out more
                        return [0.1, 1]
                    else:
                        # kinda freakish
                        # let's backdown
                        return [-self.stop_speed(0.44), 0]
                
                if yellow_b_r > 0.02 and white_b_l < 0.02:
                    return [-self.stop_speed(0.44), 0]

                if white_b_l > 0.02 and white_b_r > 0.02:
                    # cheeky
                    # in front could be the grass
                    # but it could also be facing the edge
                    # false edges may be detected, so let's back down a bit
                    return [-self.stop_speed(0.44), 0]  

                if white_b_r > 0.02:
                    return [self.stop_speed(0.44), 1]
                
                if white_b_l > 0.02 and yellow_b_r > 0.02:
                    # oh boy, we're reverse
                    # let's turn around madly
                    # welp, based on the slopes!
                    if pos_slope_avg is not None and pos_slope_avg < 0.1:
                        # positive slope is very low
                        # let's back down once for safety and turn right
                        self.buffer = [[0.1, -1]] * 30 + [[-self.stop_speed(0.44), 0]] * 3
                        return self.buffer.pop()
                    
                    # By default, just turn right madly, should be safe

                    self.buffer = [[0.1, -1]] * 16
                    return [0.1, -1]

                if white_b_l > 0.02 and yellow_b_r <= 0.02:
                    # it's kinda hard to tell
                    # if there's a lot of white on the left, then it shouldn't be the yellow lane
                    # just turn right to identify where the yellow lane is
                    return [0.1, -1]

                return [0.1, 1]
            else: # plenty of yellow on the left
                # If there's lots of white-gray on the left, it's the grass
                # If not, highly likely that it's the yellow lane
                # Anyhow, turn right to find more information

                return [0.1, -1]

        if self.intersection:
            red_bar, red_bar_pos = has_red_bar(image)

            if red_bar and red_bar_pos == 'top':
                self.turn_right_incoming = self.turn_left_incoming = False
                return [self.stop_speed(0.44), steerer]

            if red_bar and red_bar_pos == 'mid':
                self.turn_right_incoming = self.turn_left_incoming = False
                return [self.stop_speed(0.44), steerer]

            if red_bar and red_bar_pos == 'bot':
                self.turn_right_incoming = self.turn_left_incoming = False
                self.middle = False
                self.passing_intersection = True
                
                if self.remaining_steps_of_slow > 0:
                    self.buffer = [[0.1, 0]] * 50
                    return [0.1, 0]
                else:
                    self.buffer = [[self.stop_speed(0.44), 0]] * 16
                    return [self.stop_speed(0.44), steerer]

        if white_b_l > 0.01 and white_b_r > 0.01 and white_b_s > 0:
            self.buffer = [[-self.stop_speed(0.44), 1]] * 5
            return [-self.stop_speed(0.44), 1]

        if white_b_l < 0.001 and white_b_r < 0.001 \
            and yellow_b_l < 0.001 and yellow_b_r < 0.001 \
            and white_b_s < 0.001 and yellow_b_s < 0.001:
            self.buffer = [[-self.stop_speed(0.44), 1]] * 5
            return [-self.stop_speed(0.44), 1]

        # On track to turn left
        if self.turn_left_incoming:
            if yellow_b_l <= 0.01 or yellow_b_r > 0.01 or white_b_r > 0.001:
                return [self.stop_speed(0.35), 1]
            
            if nsa <= -0.6 or pos_slope_avg is not None:
                # If we turn left, eventually the bot will find a high enuf slope on the right side
                # or we see the positive slopes!
                self.turn_left_incoming = False
                # self.middle = False

            return [self.stop_speed(0.44), 0]

        # On track to turn right
        # Only start to turn once you can't see the yellow lane
        if self.turn_right_incoming:
            if yellow_b_l < 0.02 and yellow_b_s < 0.15:
                # something's wrong
                self.turn_right_incoming = False
                return [self.stop_speed(0.35), 1]

            if yellow_b_s < 0.15 or white_b_r > 0.01:
                # still seeing lots of yellow lane to pass thru
                return [self.stop_speed(0.44), 0]

            if psa < 0.2:
                self.buffer = [[0.1, 0], [0.1, -1], [0.1, -1]] * 16
            else:
                self.buffer = [[0.1, 0], [0.1, -1], [0.1, -1]] * 8

            # self.buffer = [[0.1, 0], [0.1, -1], [0.1, -1]] * 16
            self.turn_right_incoming = False
            # self.middle = False

            return self.buffer.pop()

        # This is to correct extreme cases
        if yellow_b_l < 0.005:
            return [0.1, 1]

        # Duckie the bot is in the middle of 2 lanes
        # Duckie is not expecting any turns
        # Expecting standard situations
        if pos_slope_avg is not None and neg_slope_avg is not None:
            if pos_slope_avg >= 0.65 and neg_slope_avg <= -0.6:
                return [self.stop_speed(0.8), steerer]

            if pos_slope_avg < 0.65 and (neg_slope_avg <= -0.6 or yellow_b_l > 0.02):
                # Found a right curve here
                # Believe Duckie will turn right in the future
                self.turn_right_incoming = True

            if neg_slope_avg > -0.6 and white_b_r > 0.02:
                # Found a left curve here
                # Believe Duckie will turn left in the future
                self.turn_left_incoming = True

            if yellow_b_l < 0.02:
                return [self.stop_speed(0.44), 1]
            
            if white_b_r < 0.02:
                return [self.stop_speed(0.44), -1]
            
            return [self.stop_speed(0.44), 0]
        
        if pos_slope_avg is not None:
            if pos_slope_avg < 0.5 and neg_slope_avg is None:
                # Found a right curve here
                # Believe Duckie will turn right in the future
                self.turn_right_incoming = True

            if white_b_r > 0.02:
                return [self.stop_speed(0.8), 1]
            else:
                return [self.stop_speed(0.44), 0]

        if neg_slope_avg is not None:
            if neg_slope_avg > -0.6:
                # Found a left curve here
                # Believe Duckie will turn left in the future
                self.turn_left_incoming = True

            if white_b_r > 0.02:
                return [self.stop_speed(0.8), steerer]
            else:
                return [self.stop_speed(0.44), steerer]


        return [self.stop_speed(0.44), steerer]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default=None)
    arg = parser.parse_args()


