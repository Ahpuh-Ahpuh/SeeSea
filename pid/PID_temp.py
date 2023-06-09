import numpy as np
from PathMaker.PathMaker import PathMaker
from controller import TelloController
import time
import matplotlib.pyplot as plt

class TelloPID:
    def __init__(self,xref,kp,ki,kd,dt):

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.xref = xref
        self.erro_lis = np.zeros((2,len(self.xref[0])))
        self.dt = dt
        self.drone_manager = TelloController()

        self.vxhis = []
        self.vyhis = []

    def main(self):
        self.drone_manager.takeoff()
        time.sleep(5)
        self.drone_manager.set_rc(0,0,0,0)
        time.sleep(1)
        for i in range(len(self.xref[0])):
            xnow = np.array((self.drone_manager.vgx_now,self.drone_manager.vgy_now))
            self.vxhis.append(xnow[0])
            self.vyhis.append(xnow[1])
            xrefi = self.xref[:,i]
            if i == 0:
                epre = np.array((0,0))
            else:
                epre = self.erro_lis[:,i-1]
            enow = xrefi -xnow
            u = self.controller(enow,epre)
            a = self.process_u(u[1])
            b = self.process_u(u[0])
            self.drone_manager.set_rc(a,b,0,0)
            time.sleep(self.dt)

        self.drone_manager.land()
        self.drone_manager.stop()


    def controller(self,enow,epre):
        up = self.kp * enow
        ui = self.ki * np.sum(self.erro_lis,axis=1) * self.dt
        ud = self.kd * (enow - epre)/self.dt

        u = up + ui + ud
        return u

    def process_u(self,u):
        if u > 30:
            u = 30
        elif u < -30:
            u = -30

        return round(u,0)



if __name__ == '__main__':
    print('start')
    n = 100
    path = PathMaker(n)
    x, y = path.circle_path(0.5)
    vx, vy = path.calc_path_v(x, y)
    ax, ay = path.clac_path_a(vx, vy)
    xref = np.vstack((ax,ay))
    dt = 0.05
    kp = 10
    ki = 10
    kd = 1.5
    tellopid = TelloPID(xref,kp,ki,kd,dt)
    tellopid.main()
    plt.plot(range(len(ax)),ax)
    plt.plot(range(len(ax)),tellopid.vxhis)
    plt.show()
    plt.plot(range(len(ay)), ay)
    plt.plot(range(len(ay)), tellopid.vyhis)
    plt.show()

    plt.plot(tellopid.drone_manager.px,tellopid.drone_manager.py)
    plt.show()