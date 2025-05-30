import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Global parameters
gna = 120  # mS*cm^-2
gk = 36  # mS*cm^-2
gleak = 0.3  # mS*cm^-2
cm = 1  # uFcm^-2
ena = 50  # mV
ek = -77  # mV
eleak = -54.4  # mV


noise_i_amp = 15
noise_m_amp = (0.2)**(1/3)
noise_h_amp = 0.1

params = {
    'noise_i_amp': 13
}


# Sodium current variables:

def a_m(v):
    return 0.1*(v+40)/(1-np.exp(-(v+40)/10))
def b_m(v):
    return 4*np.exp(-(v+65)/18)

def a_h(v):
    return 0.07*np.exp(-(v+65)/20)
def b_h(v):
    return 1/(np.exp(-(v+35)/10)+1)

def ina(v, m, h):
    return gna*m**3*h*(v-ena)

# Potassium current variables:

def a_n(v):
    return 0.01*(v+55)/(1-np.exp(-(v+55)/10))

def b_n(v):
    return 0.125*np.exp(-(v+65)/80)

def ik(v, n):
    return gk*n**4*(v-ek)


# Other currents:

def ileak(v):
    return gleak*(v-eleak)

def istim(t, amp = 10.0):
    return amp if 20 <= t <= 60 else 0.0

def noise_i(noise_mem):
    samp = np.random.normal(0, 1)
    noise_i = params['noise_i_amp']*samp  # Units: (uA/cm^2)
    noise_mem.append(noise_i)
    return noise_i


# Legacy
def noise_m():
    samp = np.random.normal(0, 1)
    return noise_m_amp*samp  # Units: 1/s

def noise_h():
    samp = np.random.normal(0, 1)
    return noise_h_amp*samp  # Units: 1/s



# HH Ordinary Differential Equations system

def hh_ode_det(t, num_state):
    v, m, h, n = num_state

    dm = a_m(v)*(1-m)-b_m(v)*m
    dh = a_h(v)*(1-h)-b_h(v)*h
    dn = a_n(v)*(1-n)-b_n(v)*n

    dv = (istim(t) - ina(v, m, h) - ik(v, n) - ileak(v) )/ cm

    return [dv, dm, dh, dn]


def hh_ode_istoch(t, num_state, noise_mem):
    v, m, h, n = num_state

    i_noise = noise_i(noise_mem)
    dm = a_m(v)*(1-m)-b_m(v)*m
    dh = a_h(v)*(1-h)-b_h(v)*h
    dn = a_n(v)*(1-n)-b_n(v)*n

    dv = (i_noise + istim(t) - ina(v, m, h) - ik(v, n) - ileak(v) )/ cm

    return [dv, dm, dh, dn]

def hh_ode_mstoch(t, num_state):
    v, m, h, n = num_state

    dm = a_m(v)*(1-m)-b_m(v)*m + noise_m()
    dh = a_h(v)*(1-h)-b_h(v)*h
    dn = a_n(v)*(1-n)-b_n(v)*n

    dv = (istim(t) - ina(v, m, h) - ik(v, n) - ileak(v)) / cm

    return [dv, dm, dh, dn]

def hh_ode_hstoch(t, num_state):
    v, m, h, n = num_state

    dm = a_m(v) * (1 - m) - b_m(v) * m
    dh = a_h(v) * (1 - h) - b_h(v) * h + noise_h()
    dn = a_n(v) * (1 - n) - b_n(v) * n

    dv = (istim(t) - ina(v, m, h) - ik(v, n) - ileak(v)) / cm

    return [dv, dm, dh, dn]

"""
Simulation functions:
"""
def run_sim(ode_set, init_y, t_span=(0, 100)):
    t_eval = np.linspace(*t_span, 1000)
    sol = solve_ivp(ode_set, t_span, init_y, t_eval=t_eval, method='RK45')
    return sol

def run_sim_2(ode_set, init_y, t_span=(0, 100)):
    noise_mem = []

    def wrapped_ode(t, y):
        return ode_set(t, y, noise_mem)

    t_eval = np.linspace(*t_span, 1000)
    sol = solve_ivp(wrapped_ode, t_span, init_y, t_eval=t_eval, method='RK45')
    return sol, noise_mem


def plot_sim_res(sol, title=''):
    # getting the variables from simulation solution
    v = sol.y[0]
    m = sol.y[1]
    h = sol.y[2]
    n = sol.y[3]
    t = sol.t

    # calculating the currents at each timestep
    ina_vals = np.array( [ina(vv,mm,hh) for vv,mm,hh in zip(v, m, h)] )
    ik_vals = np.array( [ik(vv,nn) for vv,nn in zip(v, n)] )
    ileak_vals = np.array( [ileak(vv) for vv in v] )
    istim_vals = np.array( [istim(tt) for tt in t] )

    clrs = plt.get_cmap('tab10')

    # Top plot -> Stim magnitude
    # Mid plot -> memb potential
    # Bot plot -> currents
    fig1 = plt.figure(figsize=(8, 12))
    gs = gridspec.GridSpec(3, 1, height_ratios=[0.25, 1, 1])

    ax_stim = fig1.add_subplot(gs[0])
    ax_v = fig1.add_subplot(gs[1], sharex=ax_stim)
    ax_i = fig1.add_subplot(gs[2], sharex=ax_stim)

    ax_stim.plot(t, istim_vals, color=clrs(4))
    ax_stim.set_ylabel('I_stim (mA/cm^2)')
    ax_stim.set_title('Stimulation current')
    ax_stim.grid(True)

    ax_v.plot(t, v, label='V (mV)', color=clrs(0))
    ax_v.set_ylabel('Voltage (mV)')
    ax_v.set_title('Membrane potential v. time')
    ax_v.grid(True)
    ax_v.legend()
    ax_v.set_ylim(-80, 60)

    ax_i.plot(t, ina_vals, label='I_Na', color=clrs(1))
    ax_i.plot(t, ik_vals, label='I_K', color=clrs(2))
    ax_i.plot(t, ileak_vals, label='I_leak', color=clrs(3))
    ax_i.set_xlabel('Time (ms)')
    ax_i.set_ylabel('Current (mA/cm^2)')
    ax_i.set_title('Ion currents v. time')
    ax_i.grid(True)
    ax_i.legend()
    ax_i.set_ylim(-900, 900)

    fig1.suptitle(title)
    fig1.tight_layout(rect=[0, 0, 1, 0.97])

    # Top row -> currents
    # Bot row -> gating variables
    fig2, axs2 = plt.subplots(2, 2, figsize=(16, 8), sharex=True)

    axs2[0, 0].plot(t, ina_vals, label='I_Na', color=clrs(1))
    axs2[0, 0].set_ylabel('Current (mA/cm^2)')
    axs2[0, 0].set_title('Sodium current (I_Na)')
    axs2[0, 0].grid(True)
    axs2[0, 0].legend()

    axs2[0, 1].plot(t, ik_vals, label='I_K', color=clrs(2))
    axs2[0, 1].set_title('Potassium current (I_K)')
    axs2[0, 1].grid(True)
    axs2[0, 1].legend()

    # adaptive axis limits
    ymin = min(np.min(ina_vals), np.min(ik_vals))
    ymax = max(np.max(ina_vals), np.max(ik_vals))
    axs2[0, 0].set_ylim(ymin, ymax)
    axs2[0, 1].set_ylim(ymin, ymax)

    axs2[1, 0].plot(t, m**3, label='m^3', color=clrs(4))
    axs2[1, 0].plot(t, h, label='h', color=clrs(5))
    axs2[1, 0].set_xlabel('Time (ms)')
    axs2[1, 0].set_ylabel('Gate state')
    axs2[1, 0].set_title('Na gates (m^3, h)')
    axs2[1, 0].grid(True)
    axs2[1, 0].legend()

    axs2[1, 1].plot(t, n**4, label='n^4', color=clrs(6))
    axs2[1, 1].set_xlabel('Time (ms)')
    axs2[1, 1].set_title('K gates (n^4)')
    axs2[1, 1].grid(True)
    axs2[1, 1].legend()

    ymin = -0.1
    ymax = 1.1
    axs2[1, 0].set_ylim(ymin, ymax)
    axs2[1, 1].set_ylim(ymin, ymax)

    fig2.suptitle(title)
    fig2.tight_layout()
    plt.show()

def plot_noise_stim_ration(sol, noise_mem, title=''):
    v, m, h, n = sol.y
    t = sol.t

    # Compute currents
    ina_vals = np.array([ina(vv, mm, hh) for vv, mm, hh in zip(v, m, h)])
    ik_vals = np.array([ik(vv, nn) for vv, nn in zip(v, n)])
    ileak_vals = np.array([ileak(vv) for vv in v])
    istim_vals = np.array([istim(tt) for tt in t])

    clrs = plt.get_cmap('tab10')

    fig1 = plt.figure(figsize=(8, 16))
    gs = gridspec.GridSpec(4, 1, height_ratios=[0.25, 0.25, 1, 1])

    ax_stim = fig1.add_subplot(gs[0])
    ax_noise = fig1.add_subplot(gs[1])
    ax_v = fig1.add_subplot(gs[2], sharex=ax_stim)
    ax_i = fig1.add_subplot(gs[3], sharex=ax_stim)

    ax_stim.plot(t, istim_vals, color=clrs(4), linewidth=1)
    ax_noise.plot(t, noise_mem[:len(t)], color=clrs(4), linewidth=1)  # Clip just in case
    ax_v.plot(t, v, color=clrs(0), label='V (mV)')
    ax_i.plot(t, ina_vals, color=clrs(1), label='I_Na')
    ax_i.plot(t, ik_vals, color=clrs(2), label='I_K')
    ax_i.plot(t, ileak_vals, color=clrs(3), label='I_leak')

    ax_stim.set_ylabel('I_stim (mA/cm²)')
    ax_stim.set_title('Stimulation current')
    ax_stim.grid(True)
    ax_stim.set_ylim(-45, 45)

    ax_noise.set_ylabel('I_noise (mA/cm²)')
    ax_noise.set_title('Noise current')
    ax_noise.grid(True)
    ax_noise.set_ylim(-45, 45)

    ax_v.set_ylabel('Voltage (mV)')
    ax_v.set_title('Membrane potential v. time')
    ax_v.grid(True)
    ax_v.legend()
    ax_v.set_ylim(-80, 60)

    ax_i.set_ylabel('Current (mA/cm²)')
    ax_i.set_xlabel('Time (ms)')
    ax_i.set_title('Ion currents v. time')
    ax_i.grid(True)
    ax_i.legend()
    ax_i.set_ylim(-900, 900)

    fig1.suptitle(title)
    fig1.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def plot_curr_noise_increment(amp_start = 0, amp_stop = 50, amp_increment = 5, t_span=(0, 200), show=False):
    amps = list(range(amp_start, amp_stop, amp_increment))
    frs = []

    if show:
        clrs = plt.get_cmap('tab10')
        fig, axs = plt.subplots(len(amps), 1, figsize=(12, len(amps)*4))

    for i, amp in enumerate(amps):
        print(f'Sim for amp={amp}')
        params['noise_i_amp'] = amp
        sol = run_sim(hh_ode_istoch, y0, t_span=t_span)
        v = sol.y[0]
        t = sol.t

        thresh = 0
        spks = np.where((v[1:] > thresh) & (v[:-1] <= thresh))[0]
        n_spks = len(spks)
        sim_dur_s = ( t[-1] - t[0] ) / 1000
        rate = n_spks / sim_dur_s
        frs.append(rate)

        if show:
            clr = clrs(i % 10)
            axs[i].plot(t, v, color=clr, label=f'V (mV) for noise magnitude of {amp} mA/cm^2')
            axs[i].set_ylim(-90, 60)
            axs[i].set_ylabel('V (mV)')
            axs[i].legend(loc='upper right')
            axs[i].grid(True)

    if show:
        axs[-1].set_xlabel('Time (ms)')
        fig.suptitle("Spontaneous activity in HH neuron model for different magnitude of current noise", fontsize=14)

        plt.show()

    return amps, frs

def plot_firing_rate_vs_noise(amps, firing_rates):

    plt.figure(figsize=(8, 4))
    plt.plot(amps, firing_rates, marker='o', linestyle='-', color='tab:blue')

    plt.title('Firing rate v. current noise amplitude')
    plt.xlabel('Noise amp (mA/cm^2)')
    plt.ylabel('Firing rate (Hz)')
    plt.xticks(ticks=np.arange(0, 201, 10))
    plt.grid(True)
    plt.tight_layout()
    plt.show()


V0 = -65.0
m0 = a_m(V0) / (a_m(V0) + b_m(V0))
h0 = a_h(V0) / (a_h(V0) + b_h(V0))
n0 = a_n(V0) / (a_n(V0) + b_n(V0))
y0 = [V0, m0, h0, n0]


# With stimulation
"""
# Deterministic
det_sol = run_sim(hh_ode_det, y0)
plot_sim_res(det_sol, 'Deterministic HH model with external current stim.: ')

# Stochastic
inoise_sol = run_sim(hh_ode_istoch, y0)
plot_sim_res(inoise_sol, 'HH model with current noise and external current stim: ')
"""

"""
mnoise_sol = run_sim(hh_ode_mstoch, y0)
plot_sim_res(mnoise_sol, 'Na activating gate (m) noise HH model: ')

hnoise_sol = run_sim(hh_ode_hstoch, y0)
plot_sim_res(hnoise_sol, 'Na inactivating gate (n) noise HH model: ')
"""

# Without stimulation:
"""
# Deterministic
det_sol = run_sim(hh_ode_det, y0)
plot_sim_res(det_sol, 'Deterministic HH model WITHOUT external current stim.: ')

# Stochastic
inoise_sol = run_sim(hh_ode_istoch, y0)
plot_sim_res(inoise_sol, f'HH model with current noise amp={noise_i_amp}, WITHOUT external current stim: ')
"""

"""amps, frs = plot_curr_noise_increment(0, 201, 10, t_span=(0,800), show=False)
plot_firing_rate_vs_noise(amps, frs)
"""

# Signal noise ratio
inoise_sol, noise_mem = run_sim_2(hh_ode_istoch, y0)
plot_noise_stim_ration(inoise_sol, noise_mem,f"Comparison of stimulus current v. noise current magnitude (amp={params['noise_i_amp']}): ")