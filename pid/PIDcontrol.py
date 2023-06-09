import time, cv2
import numpy as np
from djitellopy import Tello
import time

previous_time = time.time()
current_time = 0

tello = Tello()
tello.imperial = False
x_travelled: int = 0
y_travelled: int = 0
z_travelled: int = 0
x_speed: int = 0
y_speed: int = 0
z_speed: int = 0
x_accl: int = 0
y_accl: int = 0
z_accl: int = 0

counter = 0
x_accl_preset: int = 0
y_accl_preset: int = 0
z_accl_preset: int = 0

'''
# position of human:
human_x = 100
human_y = 100
human_z = 30
'''

class PID:
    def __init__(self):
        self.PID = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]  # row = direction, column = PID


PID = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])


def start_drone():
    #tello.takeoff()

    if tello.land():
        flying = False


def get_position_travelled():
    current_time = time.time()
    x_speed = tello.get_speed_x()
    y_speed = tello.get_speed_y()
    z_speed = tello.get_speed_z()

    x_travelled += x_speed * (current_time - previous_time)
    y_travelled += y_speed * (current_time - previous_time)
    z_travelled += z_speed * (current_time - previous_time)
    previous_time = current_time

'''
human_x = 100
human_y = 100
human_z = 30
def human_detection():
    while True:
        length = tello.get_distance_tof()
        print("length : {}".format(length))
        #time.sleep(5)
        if length < 60:
            return True
        else:
            return False
'''

def PID_Controll(Kp, Ki, Kd):
    global x_accl_preset, y_accl_preset, z_accl_preset, counter
    while flying:
        #만약 배터리가 일정 수준 이하라면 break 하도록
        if drone.get_battery()<15:
            break
        '''
        if human_detect:
            print("human detected")
            while abs(tello.get_pitch()) > 30 and abs(tello.get_roll()) > 30:
                temp = 1
            tello.takeoff()
            tello.move_up(50)
            tello.move_right(200)
            break
        '''

        x_accl_act = tello.get_acceleration_x()
        y_accl_act = tello.get_acceleration_y()
        z_accl_act = tello.get_acceleration_z()

        x_accl = x_accl_act - x_accl_preset
        y_accl = y_accl_act - y_accl_preset
        z_accl = z_accl_act - z_accl_preset

        x_prev_speed = tello.get_speed_x()
        y_prev_speed = tello.get_speed_y()
        z_prev_speed = tello.get_speed_z()

        print("x_speed = {}".format(x_prev_speed))
        print("y_speed = {}".format(y_prev_speed))
        print("z_speed = {}".format(z_prev_speed))
        print("x_accl_act = {}".format(x_accl))
        print("y_accl_act = {}".format(y_accl))
        print("z_accl_act = {}".format(z_accl))
        #print("flight cm = {}".format(tello.get_distance_tof()))
        #print("pitch = {}".format(tello.get_pitch()))
        #print("roll = {}".format(tello.get_roll()))
        #print("yaw = {}".format(tello.get_yaw()))
        print()

        sum_accel = x_accl ** 2 + y_accl ** 2 + z_accl ** 2
        # print("sum_accl = {}".format(int(sum_accel)))
        # Thrown = 0
        # 400000 with motors on and turn off z_accl
        # 900000 with motors off
        #if sum_accel >= 400000:  # 100000000:
        if sum_accel >= 900000: #100000000:
            print("Accel > 948")
            #Thrown = 1
            while abs(tello.get_pitch()) > 30 and abs(tello.get_roll()) > 30:
                temp = 1
            tello.takeoff()
            tello.move_up(50)
            tello.move_right(100)
            break
        else:
            x_accl_preset = x_accl_act * 0.4 + x_accl_preset * 0.6
            y_accl_preset = y_accl_act * 0.4 + y_accl_preset * 0.6
            z_accl_preset = z_accl_act * 0.4 + z_accl_preset * 0.6

        # PID based on speed
        # while Thrown:
        #     x_speed = tello.get_speed_x()
        #
        #     y_speed = tello.get_speed_y()
        #
        #     z_speed = tello.get_speed_z()
        #
        #     PID[0][0] = x_speed
        #     PID[1][0] = y_speed
        #     PID[2][0] = z_speed
        #
        #     PID[0][1] = (x_speed - x_prev_speed)
        #     PID[1][1] = (y_speed - y_prev_speed)
        #     PID[2][1] = (z_speed - z_prev_speed)
        #
        #     PID[0][2] += x_speed
        #     PID[1][2] += y_speed
        #     PID[2][2] += z_speed
        #
        #     set_x = Kp * PID[0][0] + Kd * PID[0][1] + Ki * PID[0][2]
        #
        #     set_y = Kp * PID[1][0] + Kd * PID[1][1] + Ki * PID[1][2]
        #
        #     set_z = Kp * PID[2][0] + Kd * PID[2][1] + Ki * PID[2][2]
        #
        #     sum_pid = set_x ** 2 + set_y ** 2 + set_z ** 2
        #     #tello.set_speed(int(sum_pid))
        #     if sum_pid < 1:
        #         break


        # PID based on position
            # if set_x >= 500:
            #     set_x = 500
            # elif set_x < -500:
            #     set_x = -500
            #
            # if set_y >= 500:
            #     set_y = 500
            # elif set_y < -500:
            #     set_y = -500
            #
            # if set_z >= 500:
            #     set_z = 500
            # elif set_z < -500:
            #     set_z = -500
            #
            # if set_x < 0:
            #     tello.move_forward(set_x)
            # else:
            #     tello.move_back(set_x)
            #
            #
            # if set_y < 0:
            #     tello.move_right(set_y)
            # else:
            #     tello.move_left(set_y)
            #
            #
            # if set_z < 0:
            #     tello.move_up(set_z)
            # else:
            #     tello.move_down(set_z)


def videoRecorder():
    frame_read = tello.get_frame_read()
    height, width, _ = frame_read.frame.shape
    video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))
    counter = 0
    while counter < 1000:
        video.write(frame_read.frame)
        time.sleep(1 / 30)
        counter += 1

    video.release()


def stream():
    frame_read = tello.get_frame_read()
    height, width, _ = frame_read.frame.shape
    counter = 0
    tello.takeoff()
    while counter < 100000:
        frame_read = tello.get_frame_read()
        cv.imshow('frame', frame_read.frame)
        counter += 1
    time.sleep(3)  # requires sleep for the processing of cv2.write
    cv2.imwrite("picture.png", frame_read.frame)
    print("Done")



#Needed to calibrate as the detector can be off
def caliberateAcceleration():
    global x_accl_preset, y_accl_preset, z_accl_preset, counter

    while counter < 10000:
        x_accl_preset += tello.get_acceleration_x()
        y_accl_preset += tello.get_acceleration_y()
        z_accl_preset += tello.get_acceleration_z()
        counter += 1
        
    x_accl_preset = x_accl_preset/counter
    y_accl_preset = y_accl_preset/counter
    z_accl_preset = z_accl_preset/counter
    print("x_accl avg = {}".format(x_accl_preset))
    print("y_accl avg = {}".format(y_accl_preset))
    print("z_accl avg = {}".format(z_accl_preset))

if __name__ == "__main__":
    tello.connect()
    print("Battery: {}".format(tello.get_battery()))
    caliberateAcceleration()
    PID_Controll(0.1, 0.2, 0.15)
    tello.land()
    tello.end()