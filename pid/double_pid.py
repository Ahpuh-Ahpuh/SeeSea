#이중 pid 프로그램 (c(아두이노)-> python(tello)

#각도를 저장하기 위한 변수
#int AcX,AcY,Acz,Tmp,GyX,GyY,Gyz
#float accel_angle_x,accel_angle_y,accel_angel_z

#msg 출력을 위한 변수 ; 변수가 어딘가에 있다는 것을 sw에 알릴 때 사용
'''
roll_output,pitch_output,yaw_output
'''
#보정형 변수
#float baseAcX,baseAcY,baseAcz
#float baseGyX, baseGyY, baseGyZ;

#시간 관련
#float dt, t_now,t_prev

#pid에서 제어할 변수들 (roll,pitch,yaw)

from djitellopy import Tello
import cv2
import time
import threading
import os
import PID

def initDT():
    t_prev=micros()

def readAccelYPR():
    current_time = time.time()
    x_speed = tello.get_speed_x()
    y_speed = tello.get_speed_y()
    z_speed = tello.get_speed_z()

    x_travelled += x_speed * (current_time - previous_time)
    y_travelled += y_speed * (current_time - previous_time)
    z_travelled += z_speed * (current_time - previous_time)
    previous_time = current_time

count=0
#10000번 동안 반복
#gyro센서 값 받는 것 -> 자이로 센서 대신
def readAccelGyro():
    global x_accl_preset, y_accl_preset, z_accl_preset, counter

    while counter < 10000:
        x_accl_preset += tello.get_acceleration_x()
        y_accl_preset += tello.get_acceleration_y()
        z_accl_preset += tello.get_acceleration_z()
        counter += 1

    x_accl_preset = x_accl_preset / counter
    y_accl_preset = y_accl_preset / counter
    z_accl_preset = z_accl_preset / counter

def initYPT():
    #초기 호버링의 각도를 잡아주기 위해
    for i in range(10):
        reedAccelGyro()
        calcDT()
        calcAccelYPR()
        calcFilteredYPR()

        base_roll_target_angle+=filtered_angle_y
        base_pitch_target_angle+=filtered_angel_x
        base_yaw_target_angle+=filtered_angle_z

        sleep(100)
    base_roll_target_angle/=10
    base_pitch_target_angle/=10
    base_yaw_target_angle/=10

    roll_target_angle=base_roll_target_angle
    pitch_target_angle=base_pitch_target_angle
    yaw_target_angle=base_yaw_target_angle

def calibAccelGyro():
    sumAcX=0.0
    sumAcY=0.0
    sumAcZ=0.0
    sumGyX=0.0
    sumGyY=0.0
    sumGyZ=0.0

    readAccelGryo()

    for i in range(10):
        readAccelGryo()
        sumAcX+=AcX; sumAcY+=AcZ; sumAcZ+=AcZ
        sumGyX+=GyX; sumGyY+=GyY; sumGyZ+=GyZ

        sleep(100)

    baseAcX=sumAcX/10
    baseAcY=sumAcY/10
    baseAcZ=sumAcZ/10
    baseGyX=sumGyX/10
    baseGyY=sumGyY/10
    baseGyZ=sumGyZ/10


def dualPID(target_angle,angle_in,rate_in,stab_kp,stab_ki,rate_kp,rate_ki,stab_iterm,rate_iterm,output):
    #이중 PI 알고리즘
    angle_error=target_angle-angle_in

    stab_pterm=stab_iterm*angle_error
    stab_iterm+=stab_ki*angele_error*dt

    desired_rate=stab_pterm

    rate_error=desired_rate-rate_in

    rate_pterm=rate_kp*rate_error
    rate_iterm+=rate_ki*rate_error*dt

    ##각속도 비례항 + 각속도 적분항 + 안정화 적분항
    output=rate_pterm+rate_iterm+stab_iterm

def calcYPRtoDualPID():
    roll_angle_in=filtered_angle_y
    roll_rate_in=gyro_y

    dualPID(roll_target_angle,roll_angle_in,roll_rate_in,
            roll_stab_kp,roll_stab_ki,
            roll_rate_kp,roll_rate_ki,roll_stab_iterm,
            roll_rate_iterm,roll_output)

    pitch_angle_in=filtered_angle_X
    pitch_rate_in=gyro_x

    dualPID(pitch_target_in,pitch_angle_in,pitch_rate_in,
            pitch_stab_kp,pitch_stab_ki,
            pitch_rate_kp,pitch_rate_ki,
            pitch_stab_iterm,pitch_rate_iterm,pitch_output)

    yaw_angle_in=filtered_angle_z
    yaw_rate_in=gyro_x

    dualPID(yaw_target_in,yaw_angle_in,yaw_rate_in,
            yaw_stab_kp,yaw_stab_ki,
            yaw_rate_kp,yaw_rate_ki,
            yaw_stab_iterm,yaw_rate_iterm,yaw_output)



def setup():
    initYPT() #초기각도 값 설정 (호버링을 위한 목표 각도로 사용)
    calibAccelGyro()
    initDT()


def loop():
    readAccelGryo()

    calcDt()

    calcAccelYPR()
    calcGyroYPR()
    calcFilteredYPR()

    #이중PID 구현
    calcYPRtoDualPID()
    calcMotorSpeed()



