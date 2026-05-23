import control as ctrl
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# 设置画图的中文字体，保证生成的图表能直接贴进 PPT
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

# ==========================================
# 第一部分：构建受控体物理模型 (Plant)
# ==========================================
def build_plant():
    """
    根据物理参数和推导出的传递函数，构建飞行器姿态被控对象 Gp(s)
    """
    s = ctrl.tf('s')
    
    # 物理参数录入
    Ks = 1.0
    K = 181.17      # 题目中指定的前置放大器增益
    K1 = 10.0
    K2 = 0.5
    Kt = 0.0        # 测速反馈系数为0，大幅简化分母
    Ra = 5.0
    La = 0.003
    Ki = 9.0
    Kb = 0.0636
    Jm = 0.0001
    JL = 0.01
    Bm = 0.005
    BL = 1.0
    N = 110.0
    
    # 计算折算到电机轴的总惯量和总摩擦
    Jt = Jm + (N**2) * JL
    Bt = Bm + (N**2) * BL
    
    # 根据手推公式构建开环传递函数 (包含 K=181.17)
    # 分子：N * K * K1 * Ks * Ki
    num = N * K * K1 * Ks * Ki
    
    # 分母：s * [ (Ra + La*s + K1*K2)*(Jt*s + Bt) + Ki*Kb ] (Kt=0的简化版)
    den_term1 = (Ra + La * s + K1 * K2) * (Jt * s + Bt)
    den_term2 = Ki * Kb
    
    Gp = num / (s * (den_term1 + den_term2))
    
    return Gp

# ==========================================
# 第二部分：算法评估接口 (供 LLM Agent 调用)
# ==========================================

def evaluate_problem_1_pd(Kd):
    """
    针对问题 (1)：PD 控制器调参接口 (固定 Kp = 1)
    """
    s = ctrl.tf('s')
    Kp = 1.0 
    Gc = Kp + Kd * s
    
    Gp = build_plant()
    sys_open = ctrl.series(Gc, Gp)
    sys_closed = ctrl.feedback(sys_open, 1)
    
    # 1. 动态指标：阶跃响应
    info = ctrl.step_info(sys_closed)
    
    # 2. 稳态指标：斜坡输入的稳态误差 (理论值计算，保证 Agent 获取精确数据)
    # 对于 I 型系统，斜坡误差 ess = 1 / Kv，其中 Kv = lim(s->0) s * G_open(s)
    s_Gopen = ctrl.minreal(s * sys_open)
    Kv = ctrl.dcgain(s_Gopen)
    e_ss = 1.0 / Kv if Kv != 0 else float('inf')
    
    metrics = {
        "overshoot": info['Overshoot'],
        "tr": info['RiseTime'],
        "ts": info['SettlingTime'],
        "e_ss_ramp": e_ss
    }
    
    # 画图验证并保存
    plot_simulation(sys_closed, "Problem 1: PD Control", f"Kp={Kp}, Kd={Kd:.4f}", input_type='ramp')
    
    return metrics

def evaluate_problem_2_pi(Kp, Ki):
    """
    针对问题 (2)：PI 控制器调参接口
    """
    s = ctrl.tf('s')
    Gc = Kp + Ki / s
    
    Gp = build_plant()
    sys_open = ctrl.series(Gc, Gp)
    sys_closed = ctrl.feedback(sys_open, 1)
    
    # 1. 动态指标：阶跃响应
    info = ctrl.step_info(sys_closed)
    
    # 2. 稳态指标：加速度输入的稳态误差 (理论值计算)
    # PI控制器加入后系统变为 II 型，加速度误差 ess = 1 / Ka，其中 Ka = lim(s->0) s^2 * G_open(s)
    s2_Gopen = ctrl.minreal((s**2) * sys_open)
    Ka = ctrl.dcgain(s2_Gopen)
    e_ss = 1.0 / Ka if Ka != 0 else float('inf')
    
    metrics = {
        "overshoot": info['Overshoot'],
        "tr": info['RiseTime'],
        "ts": info['SettlingTime'],
        "e_ss_acc": e_ss
    }
    
    # 画图验证并保存
    plot_simulation(sys_closed, "Problem 2: PI Control", f"Kp={Kp:.4f}, Ki={Ki:.4f}", input_type='acc')
    
    return metrics

def evaluate_problem_3_pid(Kp, Ki, Kd):
    """
    针对问题 (3)：PID 控制器调参接口 (时域规格)
    """
    s = ctrl.tf('s')
    # 完整的 PID 控制器传递函数
    Gc = Kp + Ki / s + Kd * s
    
    Gp = build_plant()
    sys_open = ctrl.series(Gc, Gp)
    sys_closed = ctrl.feedback(sys_open, 1)
    
    # 1. 动态指标：阶跃响应
    info = ctrl.step_info(sys_closed)
    
    # 2. 稳态指标：加速度输入的稳态误差 (PID 包含 1/s，受控体也有 1/s，系统为 II 型)
    s2_Gopen = ctrl.minreal((s**2) * sys_open)
    Ka = ctrl.dcgain(s2_Gopen)
    e_ss = 1.0 / Ka if Ka != 0 else float('inf')
    
    metrics = {
        "overshoot": info['Overshoot'],
        "tr": info['RiseTime'],
        "ts": info['SettlingTime'],
        "e_ss_acc": e_ss
    }
    
    # 画图验证并保存 (复用之前的时域画图函数)
    plot_simulation(sys_closed, "Problem 3: PID Control (Time Domain)", 
                    f"Kp={Kp:.4f}, Ki={Ki:.4f}, Kd={Kd:.4f}", input_type='acc')
    
    return metrics

def evaluate_problem_4_pid_freq(Kp, Ki, Kd):
    """
    针对问题 (4)：PID 控制器调参接口 (频域规格)
    """
    s = ctrl.tf('s')
    Gc = Kp + Ki / s + Kd * s
    
    Gp = build_plant()
    sys_open = ctrl.series(Gc, Gp)
    sys_closed = ctrl.feedback(sys_open, 1)
    
    # 1. 时域稳态指标：加速度输入的稳态误差
    s2_Gopen = ctrl.minreal((s**2) * sys_open)
    Ka = ctrl.dcgain(s2_Gopen)
    e_ss = 1.0 / Ka if Ka != 0 else float('inf')
    
    # 2. 频域指标提取
    # 2.1 相位裕度 (Phase Margin) -> 使用开环系统 sys_open
    gm, pm, wg, wp = ctrl.margin(sys_open)
    
    # 2.2 谐振峰值 (Mr) 和 带宽 (BW) -> 使用闭环系统 sys_closed
    # 生成对数分布的频率向量 (从 1 rad/s 到 100000 rad/s，采样 5000 个点保证精度)
    omega = np.logspace(0, 5, 5000) 
    mag, phase, _ = ctrl.bode(sys_closed, omega=omega, plot=False)
    
    # 谐振峰值 Mr 是闭环幅频响应的最大绝对值
    Mr = float(np.max(mag))
    
    # 带宽 BW 的定义：闭环幅频特性下降到直流增益的 1/sqrt(2) (即 -3dB) 时的频率
    # 因为系统是 II 型，闭环直流增益理论上为 1 (0dB)
    threshold = mag[0] / np.sqrt(2) 
    below_threshold_indices = np.where(mag < threshold)[0]
    
    if len(below_threshold_indices) > 0:
        BW = float(omega[below_threshold_indices[0]])
    else:
        BW = float('inf') # 如果带宽极大，超出计算范围
        
    metrics = {
        "e_ss_acc": e_ss,
        "phase_margin": float(pm) if not np.isnan(pm) else 0.0,
        "Mr": Mr,
        "BW": BW
    }
    
    # 调用专属的频域画图函数
    plot_frequency_simulation(sys_closed, "Problem 4: PID Control (Freq Domain)", 
                              f"Kp={Kp:.2f}, Ki={Ki:.2f}, Kd={Kd:.2f}", metrics)
    
    return metrics

# ==========================================
# 第三部分：自动绘图工具
# ==========================================

def plot_simulation(sys_closed, title, param_text, input_type='ramp'):
    """
    自动生成阶跃响应和特定输入(斜坡/加速度)响应的对比图
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # 图 1：阶跃响应
    T_step, y_step = ctrl.step_response(sys_closed)
    axs[0].plot(T_step, y_step, 'b-', label='系统输出')
    axs[0].axhline(1, color='k', linestyle='--', label='参考输入')
    axs[0].set_title(f"{title} - 阶跃响应\n({param_text})")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(True)
    axs[0].legend()
    
    # 图 2：稳态误差响应
    T = np.linspace(0, 10, 2000)
    if input_type == 'ramp':
        r = T # 斜坡输入
        input_label = '斜坡输入 (r=t)'
    else:
        r = 0.5 * T**2 # 加速度输入
        input_label = '加速度输入 ($r=0.5t^2$)'
        
    T_out, y_out = ctrl.forced_response(sys_closed, T, U=r)
    axs[1].plot(T, r, 'k--', label=input_label)
    axs[1].plot(T_out, y_out, 'g-', label='系统追踪输出')
    axs[1].set_title(f"追踪性能观测 ({input_label})")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Amplitude")
    axs[1].grid(True)
    axs[1].legend()
    
    plt.tight_layout()
    save_folder = "simulation_results"
    os.makedirs(save_folder, exist_ok=True)
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"result_{time_stamp}.png"
    save_path = os.path.join(save_folder, file_name)
    plt.savefig(save_path)
    # plt.close() # 跑Agent时取消注释以静默保存，不弹出窗口

def plot_frequency_simulation(sys_closed, title, param_text, metrics):
    """
    自动生成闭环幅频响应图 (标出带宽) 和 加速度稳态误差追踪图
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # 图 1：闭环幅频响应 (Bode图的幅频部分)
    omega = np.logspace(0, 5, 2000)
    mag, _, _ = ctrl.bode(sys_closed, omega=omega, plot=False)
    mag_db = 20 * np.log10(mag) # 转换为 dB 方便查看
    
    axs[0].semilogx(omega, mag_db, 'b-', label='闭环幅频特性')
    
    # 标出 -3dB (带宽高频截止线)
    axs[0].axhline(20 * np.log10(mag[0] / np.sqrt(2)), color='r', linestyle='--', label='-3dB 带宽阈值线')
    axs[0].axvline(metrics['BW'], color='g', linestyle='-.', label=f"BW = {metrics['BW']:.1f} rad/s")
    
    axs[0].set_title(f"{title} - 闭环幅频响应\n(Mr={metrics['Mr']:.3f}, PM={metrics['phase_margin']:.1f}°)")
    axs[0].set_xlabel("Frequency (rad/s)")
    axs[0].set_ylabel("Magnitude (dB)")
    axs[0].grid(True, which="both", ls="--", alpha=0.5)
    axs[0].legend()
    
    # 图 2：加速度稳态误差响应
    T = np.linspace(0, 5, 2000)
    r = 0.5 * T**2
    T_out, y_out = ctrl.forced_response(sys_closed, T, U=r)
    
    axs[1].plot(T, r, 'k--', label='加速度输入 ($r=0.5t^2$)')
    axs[1].plot(T_out, y_out, 'g-', label='系统追踪输出')
    axs[1].set_title(f"追踪性能观测 (ess={metrics['e_ss_acc']:.4f})")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Amplitude")
    axs[1].grid(True)
    axs[1].legend()
    
    plt.tight_layout()
    
    save_folder = "simulation_results"
    os.makedirs(save_folder, exist_ok=True)
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"result_freq_{time_stamp}.png"
    save_path = os.path.join(save_folder, file_name)
    plt.savefig(save_path)

# ==========================================
# 第四部分：测试代码模块，agent接口可直接忽略该部分
# ==========================================
if __name__ == '__main__':
    print("--- 正在测试问题 1 (PD控制器) ---")
    metrics_pd = evaluate_problem_1_pd(Kd=0.05)
    for key, value in metrics_pd.items():
        print(f"{key}: {value}")
        
    print("\n--- 正在测试问题 2 (PI控制器) ---")
    metrics_pi = evaluate_problem_2_pi(Kp=1.0, Ki=0.5)
    for key, value in metrics_pi.items():
        print(f"{key}: {value}")

    print("\n--- 正在测试问题 3 (PID控制器(时域)) ---")
    metrics_pid = evaluate_problem_3_pid(Kp=1.0, Ki=0.5,Kd=1)
    for key, value in metrics_pid.items():
        print(f"{key}: {value}") 

    print("\n--- 正在测试问题 4 (PID控制器(频域)) ---")
    metrics_pid_freq = evaluate_problem_4_pid_freq(Kp=1.0, Ki=0.5,Kd=1)
    for key, value in metrics_pid_freq.items():
        print(f"{key}: {value}") 

    plt.show() # 展示生成的图表