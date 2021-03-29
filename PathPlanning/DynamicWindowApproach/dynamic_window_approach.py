"""

Mobile robot motion planning sample with Dynamic Window Approach

author: Atsushi Sakai (@Atsushi_twi), Göktuğ Karakaşlı

"""

import math
from enum import Enum

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

show_animation = True
fig, ax = plt.subplots(2, 1)


class RobotType(Enum):
    circle = 0
    rectangle = 1


class Config:
    """
    simulation parameter class
    """

    def __init__(self):
        # robot parameter
        self.max_speed = 1.0  # [m/s]
        self.min_speed = -0.5  # [m/s]
        self.max_yaw_rate = 40.0 * math.pi / 180.0  # [rad/s]
        self.max_accel = 0.2  # [m/ss]
        self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  # [rad/ss]
        self.v_resolution = 0.01  # [m/s] 速度の解像度？→速度の分解能=0.01[m/s]単位で計算したい？
        self.yaw_rate_resolution = 0.1 * math.pi / 180.0  # [rad/s]
        self.dt = 0.1  # [s] Time tick for motion prediction モーション予測のための時間刻み
        self.predict_time = 3.0  # [s] # 予測期間
        self.to_goal_cost_gain = 0.15  # 評価関数で使用する係数その1
        self.speed_cost_gain = 1.0  # 評価関数で使用する係数その2
        self.obstacle_cost_gain = 1.0  # 評価関数で使用する係数その3
        self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked
        self.robot_type = RobotType.circle

        # if robot_type == RobotType.circle
        # Also used to check if goal is reached in both types
        self.robot_radius = 1.0  # [m] for collision check

        # if robot_type == RobotType.rectangle
        self.robot_width = 0.5  # [m] for collision check
        self.robot_length = 1.2  # [m] for collision check
        # obstacles [x(m) y(m), ....]
        """
        self.ob = np.array(
            [
                [-1, -1],
                [0, 2],
                [4.0, 2.0],
                [5.0, 4.0],
                [5.0, 5.0],
                [5.0, 6.0],
                [5.0, 9.0],
                [8.0, 9.0],
                [7.0, 9.0],
                [8.0, 10.0],
                [9.0, 11.0],
                [12.0, 13.0],
                [12.0, 12.0],
                [15.0, 15.0],
                [13.0, 13.0],
            ]
        )
        """
        self.ob = np.array(
            [
                [1, 0],
                [1, 0.5],
                [1, 1],
                [1, 1.5],
                [1, 2],
                [1, 3],
                [1, 4],
                [1, 5],
                [1, 6],
                [1, 7],
                [1, 8],
                [1, 9],
            ]
        )

    @property
    def robot_type(self):
        """getter

        Returns:
            [type]: [description]
        """
        return self._robot_type

    @robot_type.setter
    def robot_type(self, value):
        """setter

        Args:
            value ([type]): [description]

        Raises:
            TypeError: [description]
        """
        if not isinstance(value, RobotType):
            raise TypeError("robot_type must be an instance of RobotType")
        self._robot_type = value


config = Config()  # Configインスタンスを生成


def dwa_control(x, config, goal, ob):
    """Dynamic Window Approach control

    Args:
        x (list): 制御入力値
        config (object): 設定値
        goal (ndarray): 目的地(x,y)[m]
        ob (ndarray): 障害物を格納した配列

    Returns:
        [type]: [description]
    """
    dw = calc_dynamic_window(x, config)

    u, trajectory, trajec_list = calc_control_and_trajectory(x, dw, config, goal, ob)

    return u, trajectory, trajec_list


def motion(x, u, dt):
    """motion model

    Args:
        x (list): [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
        u (list): 速度, 旋回速度
        dt (float): 予測期間の間隔

    Returns:
        list: 予測した結果
    """
    x[2] += u[1] * dt  # 旋回速度の予測
    x[0] += u[0] * math.cos(x[2]) * dt  # x座標の予測
    x[1] += u[0] * math.sin(x[2]) * dt  # y座標の予測
    x[3] = u[0]  # 速度はそのままｍ
    x[4] = u[1]  # 旋回速度はそのまま

    return x


def calc_dynamic_window(x, config):
    """
    calculation dynamic window based on current state x
    x=[x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    [位置x座標, 位置y座標, 向きyaw, 速度v, 旋回速度omega]
    """

    # Dynamic window from robot specification
    # ロボットが取りうる制御入力の最大と最小の範囲
    Vs = [config.min_speed, config.max_speed, -config.max_yaw_rate, config.max_yaw_rate]
    a = patches.Rectangle(
        xy=(Vs[2], Vs[0]),
        width=Vs[3] - Vs[2],
        height=Vs[1] - Vs[0],
        ec="g",
        fill=False,
    )

    # Dynamic window from motion model
    # 現在の速度とスペック上での最大加減速を元に計算された
    # 次の時刻までに取りうる最大、最小の制御入力
    Vd = [
        x[3] - config.max_accel * config.dt,  # 速度の最小制御入力 V1(予測速度)=V0(現在速度) - A*T
        x[3] + config.max_accel * config.dt,  # 速度の最大制御入力
        x[4] - config.max_delta_yaw_rate * config.dt,  # 旋回速度の最小制御入力
        x[4] + config.max_delta_yaw_rate * config.dt,  # 旋回速度の最大制御入力
    ]
    b = patches.Rectangle(
        xy=(Vd[2], Vd[0]),
        width=Vd[3] - Vd[2],
        height=Vd[1] - Vd[0],
        ec="r",
        fill=False,
    )
    #  [v_min, v_max, yaw_rate_min, yaw_rate_max]
    # ロボットが取りうる最大最小範囲と現在の速度等のandを取りダイナミックウィンドウを生成
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]), max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]
    c = patches.Rectangle(
        xy=(dw[2], dw[0]), width=dw[3] - dw[2], height=dw[1] - dw[0], ec="y", fill=False
    )
    ax[1].cla()
    ax[1].set_ylim(config.min_speed, config.max_speed)  # y方向＝並進速度
    ax[1].set_xlim(-config.max_yaw_rate, config.max_yaw_rate)  # x方向=旋回速度
    ax[1].set_ylabel("theta vel(rad/s)")
    ax[1].set_xlabel("trans vel(m/s)")
    ax[1].spines["bottom"].set_position(("data", 0))
    ax[1].spines["left"].set_position(("data", 0))
    ax[1].add_patch(a)
    ax[1].add_patch(b)
    ax[1].add_patch(c)
    ax[1].set_aspect("equal")
    fig.show()
    return dw


def predict_trajectory(x_init, v, y, config):
    """predict trajectory with an input

    Args:
        x_init (list): 制御入力. [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
        v (float)): 並進速度[m/s]
        y (float): 旋回速度[rad/s]
        config (object): 設定値

    Returns:
        nd.array: 予測経路.arrayの先頭からn[s]単位で予測した経路が順番に格納される。[x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]の値がスタックされる。
    """

    x = np.array(x_init)
    trajectory = np.array(x)
    time = 0
    while time <= config.predict_time:
        # 予測期間内であれば経路予測を行う

        # 経路予測を実施
        x = motion(x, [v, y], config.dt)
        # 予測値をスタックに格納(vstackで縦方向に結合)
        trajectory = np.vstack((trajectory, x))
        # 予測時間更新
        time += config.dt

    return trajectory


def calc_control_and_trajectory(x, dw, config, goal, ob):
    """calculation final input with dynamic window

    Args:
        x (list): 制御入力. [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
        dw (list): ダイナミックウィンドウ. [vel_min, vel_max, yaw_min, yaw_max]
        config (list): [description]
        goal (list): 目的地
        ob ([type]): [description]

    Returns:
        [type]: [description]
    """
    # 制御入力を格納
    x_init = x[:]
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])
    trajec_list = []

    # evaluate all trajectory with sampled input in dynamic window
    # ダイナミックウィンドウでサンプリングされた入力のすべての軌道を評価する
    for v in np.arange(dw[0], dw[1], config.v_resolution):  # 速度
        for y in np.arange(dw[2], dw[3], config.yaw_rate_resolution):  # 旋回速度

            # 予測経路を生成
            trajectory = predict_trajectory(x_init, v, y, config)
            # calc cost
            # 制御入力のときのロボットの方位とゴール方向の差の角度を180度から引いた値
            # ゴールに向かってまっすぐ向かっている場合は評価値が大きくなる。
            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(
                trajectory, goal
            )
            # 現在の制御入力の速度をそのまま使う
            # 速度が早い制御入力の評価値が大きくなる。
            speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])
            # 現在の制御入力のときの最近棒の障害物までの距離を表す.
            # 障害物から遠い制御入力の評価値が大きくなる。
            ob_cost = config.obstacle_cost_gain * calc_obstacle_cost(
                trajectory, ob, config
            )

            final_cost = to_goal_cost + speed_cost + ob_cost
            trajec_list.append(trajectory)
            # search minimum trajectory
            if min_cost >= final_cost:
                min_cost = final_cost
                best_u = [v, y]
                best_trajectory = trajectory
                if (
                    abs(best_u[0]) < config.robot_stuck_flag_cons
                    and abs(x[3]) < config.robot_stuck_flag_cons
                ):
                    # to ensure the robot do not get stuck in
                    # best v=0 m/s (in front of an obstacle) and
                    # best omega=0 rad/s (heading to the goal with
                    # angle difference of 0)
                    best_u[1] = -config.max_delta_yaw_rate
    return best_u, best_trajectory, trajec_list


def calc_obstacle_cost(trajectory, ob, config):
    """calc obstacle cost inf: collision

    Args:
        trajectory (nd.array): 予測経路.arrayの先頭からn[s]単位で予測した経路が順番に格納される。[x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]の値がスタックされる。
        ob (list): 障害物のリスト
        config (object): 設定値

    Returns:
        float: 予測値が障害物と接触しない場合、
    """
    ox = ob[:, 0]  # obstacleのx座標値
    oy = ob[:, 1]  # obstacleのy座標値
    dx = trajectory[:, 0] - ox[:, None]  # 障害物と予測値のx座標の差分
    dy = trajectory[:, 1] - oy[:, None]  # 障害物と予測値のy座標の差分
    r = np.hypot(dx, dy)  # 障害物と予測値の距離(斜辺)を算出

    if config.robot_type == RobotType.rectangle:  # ロボットタイプが長方形の場合

        # @note 以下はあまり理解できていないがエラー判定を行う処理。おそらく予測値を元にロボットが障害物と接触するかどうかを判定している。

        yaw = trajectory[:, 2]  # 予測値の向いている方向
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        rot = np.transpose(rot, [2, 0, 1])  # rotを転置。[2,0,1]は転置する際の軸の順番.
        local_ob = ob[:, None] - trajectory[:, 0:2]  # 障害物と予測値の差分を取る。向きは予測値の値を-にしたものを返す
        # 1次元は-1で形状変形を自動判定
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        # @演算子。local_ob × x
        local_ob = np.array([local_ob @ x for x in rot])
        # 1次元は-1で形状変形を自動判定
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        # local_obがロボットの形状に接触するかどうかを判定
        upper_check = local_ob[:, 0] <= config.robot_length / 2
        right_check = local_ob[:, 1] <= config.robot_width / 2
        bottom_check = local_ob[:, 0] >= -config.robot_length / 2
        left_check = local_ob[:, 1] >= -config.robot_width / 2
        if (
            np.logical_and(
                np.logical_and(upper_check, right_check),
                np.logical_and(bottom_check, left_check),
            )
        ).any():
            return float("Inf")
    elif config.robot_type == RobotType.circle:
        if np.array(r <= config.robot_radius).any():
            return float("Inf")

    min_r = np.min(r)
    return 1.0 / min_r  # OK


def calc_to_goal_cost(trajectory, goal):
    """
    calc to goal cost with angle difference
    """

    # ゴールと予測のx座標の差
    dx = goal[0] - trajectory[-1, 0]
    # ゴールと予測のy座標の差
    dy = goal[1] - trajectory[-1, 1]
    # 予測から見たゴールの向きを算出
    error_angle = math.atan2(dy, dx)
    # 予測の向きと予測から見たゴールの方向の差をコスト角度として算出
    cost_angle = error_angle - trajectory[-1, 2]
    cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

    return cost


def plot_arrow(x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
    ax[0].arrow(
        x,
        y,
        length * math.cos(yaw),
        length * math.sin(yaw),
        head_length=width,
        head_width=width,
    )
    ax[0].plot(x, y)


def plot_robot(x, y, yaw, config):  # pragma: no cover
    if config.robot_type == RobotType.rectangle:
        outline = np.array(
            [
                [
                    -config.robot_length / 2,
                    config.robot_length / 2,
                    (config.robot_length / 2),
                    -config.robot_length / 2,
                    -config.robot_length / 2,
                ],
                [
                    config.robot_width / 2,
                    config.robot_width / 2,
                    -config.robot_width / 2,
                    -config.robot_width / 2,
                    config.robot_width / 2,
                ],
            ]
        )
        Rot1 = np.array(
            [[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]]
        )
        outline = (outline.T.dot(Rot1)).T
        outline[0, :] += x
        outline[1, :] += y
        ax[0].plot(
            np.array(outline[0, :]).flatten(), np.array(outline[1, :]).flatten(), "-k"
        )
    elif config.robot_type == RobotType.circle:
        circle = ax[0].Circle((x, y), config.robot_radius, color="b")
        plt.gcf().gca().add_artist(circle)
        out_x, out_y = (
            np.array([x, y])
            + np.array([np.cos(yaw), np.sin(yaw)]) * config.robot_radius
        )
        ax[0].plot([x, out_x], [y, out_y], "-k")


def main(gx=10.0, gy=10.0, robot_type=RobotType.circle):
    print(__file__ + " start!!")
    # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    x = np.array([0.0, 0.0, 95 * (math.pi / 180), 0.0, 0.0])
    # goal position [x(m), y(m)]
    goal = np.array([gx, gy])

    # input [forward speed, yaw_rate]

    config.robot_type = robot_type
    trajectory = np.array(x)
    ob = config.ob
    while True:
        #
        u, predicted_trajectory, trajec_list = dwa_control(x, config, goal, ob)
        x = motion(x, u, config.dt)  # simulate robot
        trajectory = np.vstack((trajectory, x))  # store state history

        if show_animation:
            ax[0].cla()  # 現在の軸をクリア
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                "key_release_event",
                lambda event: [exit(0) if event.key == "escape" else None],
            )
            ax[0].plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")
            # 描画が重い
            for v in trajec_list:
                ax[0].plot(v[:, 0], v[:, 1], "-b")
            ax[0].plot(x[0], x[1], "xr")
            ax[0].plot(goal[0], goal[1], "xb")
            ax[0].plot(ob[:, 0], ob[:, 1], "ok")
            plot_robot(x[0], x[1], x[2], config)
            plot_arrow(x[0], x[1], x[2])
            ax[0].axis("equal")
            ax[0].grid(True)
            plt.pause(0.0001)

        # check reaching goal
        dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
        if dist_to_goal <= config.robot_radius:
            print("Goal!!")
            break

    print("Done")
    if show_animation:
        ax[0].plot(trajectory[:, 0], trajectory[:, 1], "-r")
        plt.pause(0.0001)

    fig.show()


if __name__ == "__main__":
    # main(robot_type=RobotType.rectangle)
    main(gx=2.0, gy=10.0, robot_type=RobotType.rectangle)
# main(robot_type=RobotType.circle)
