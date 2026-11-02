#
#  ISC License
#
#  Copyright (c) 2022, Autonomous Vehicle Systems Lab, University of Colorado at Boulder
#
#  Permission to use, copy, modify, and/or distribute this software for any
#  purpose with or without fee is hereby granted, provided that the above
#  copyright notice and this permission notice appear in all copies.
#
#  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
#  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
#  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
#  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
#  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
#  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
#  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
#

r"""
Overview
--------

This scenario simulates two space objects, and active servicer approach another defund debris satellite.
The servicer spacecraft begin on a drifting approach trajectory after which it perfoms a drifting
safety-ellipse to circumnavigate the debris, and finally parks in a 2:1 bounded ellipse.

The script illustrates how the simulation can be run for fixed periods of time after which
some flight software modules change their input subscript to switch between the three possible
attitude pointing modes 1) Hill pointing, 2) spacecraft point at the debris object and
3) sun pointing of the solar panels.

To do relative motion maneuvers, methods are used to change the instantaneous relative
velocity.

The script is found in the folder ``basilisk/examples`` and executed by using::

      python3 scenarioRendezVous.py

The simulation layout is shown in the following illustration.  A single simulation process is created
which contains both the servicer spacecraft and associated the Flight Software (FSW) algorithm
modules, as well as the debris object.

.. image:: /_images/static/test_scenarioRendezVous.svg
   :align: center

When the simulation completes several plots are shown for the servicer MRP attitude history, the rate
tracking errors, the RW motor torque components, as well as the RW wheel speeds.

The simulation starts with the spacecraft entering Hill frame pointing in a along-track drift mode for
1/8th of an orbit.  Next, the logo side of the spacecraft is aimed at the debris object for 2 orbits.
Here a maneuver is performed to create a drifting circumnavigation.  Finally, the spacecraft
points the solar panels at the sun for 1/2 of an orbit.

Illustration of Simulation Results
----------------------------------

::

    show_plots = True

Note that in the RW motor torque plot both the required control torque :math:`\hat u_B` and the true
motor torque :math:`u_B` are shown.  This illustrates that with this maneuver the RW devices are being
saturated, and the attitude still eventually stabilizes.

.. image:: /_images/Scenarios/scenarioRendezVous1.svg
   :align: center

.. image:: /_images/Scenarios/scenarioRendezVous2.svg
   :align: center

.. image:: /_images/Scenarios/scenarioRendezVous3.svg
   :align: center


"""

#
# Basilisk Scenario Script and Integrated Test
#
# Purpose:  Simulation of a servicer approaching a LEO spacecraft target
#
# Author:   Hanspeter Schaub
# Creation Date:  July 3, 2022
#

import os
import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import locationPointing
from Basilisk.fswAlgorithms import (mrpFeedback, attTrackingError,
                                    rwMotorTorque, hillPoint, dockingGNC)
from Basilisk.simulation import reactionWheelStateEffector, simpleNav, spacecraft, ephemerisConverter, extForceTorque
from Basilisk.utilities import (SimulationBaseClass, macros,
                                orbitalMotion, simIncludeGravBody,
                                simIncludeRW, unitTestSupport, vizSupport)
from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.simulation import prescribedLinearTranslation
from scipy.optimize import linprog
from datetime import datetime
from scipy import linalg
import xml.etree.ElementTree as ET
import time

try:
    from Basilisk.simulation import vizInterface
except ImportError:
    pass

# The path to the location of Basilisk
# Used to get the location of supporting data.
from Basilisk import __path__
bskPath = __path__[0]
fileName = os.path.basename(os.path.splitext(__file__)[0])

def load_config(xml_file="config tunned.xml"):
    '''
    Configurations upload function
    '''
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Helper function to convert comma-separated strings to float lists
    def parse_list(val_str):
        return [float(x.strip()) for x in val_str.split(',')]
    
    config = {}
    
    # Spacecraft Properties
    config['servicer_mass'] = float(root.find('.//Servicer/Mass').text)
    config['servicer_I'] = parse_list(root.find('.//Servicer/Inertia').text)
    config['L_servicer'] = float(root.find('.//Servicer/DockingPortL').text)
    
    config['chief_mass'] = float(root.find('.//Chief/Mass').text)
    config['chief_I'] = parse_list(root.find('.//Chief/Inertia').text)
    config['L_chief'] = float(root.find('.//Chief/DockingPortL').text)
    
    config['m_tip'] = float(root.find('.//TipMass/Mass').text)
    config['r_tip'] = float(root.find('.//TipMass/Radius').text)
    config['h_tip'] = float(root.find('.//TipMass/Height').text)
    config['F_impact'] = float(root.find('.//TipMass/ForceImpact').text)
    config['t_contact'] = float(root.find('.//TipMass/TimeContact').text)
    config['tip_offset'] = float(root.find('.//TipMass/TipOffset').text)
    config['CAPTURE_POS_TOL'] = float(root.find('.//TipMass/CapturePosTol').text)
    config['CAPTURE_VEL_TOL'] = float(root.find('.//TipMass/CaptureVelTol').text)

    # Controller Variables
    config['max_thrust'] = float(root.find('.//ControllerVariables/MaxThrust').text)
    config['step_time'] = int(root.find('.//ControllerVariables/StepTime').text)
    
    # Calculate derived steps
    horizon_time = int(root.find('.//ControllerVariables/HorizonTime').text)
    config['N_horizon'] = horizon_time // config['step_time']
    config['CLOSE_ENTER_IDX'] = config['N_horizon']
    config['CLOSE_EXIT_IDX'] = config['N_horizon'] - 3
    config['TRANSFER_STEPS'] = int(np.ceil(3.0 / config['step_time']))
    
    config['standoff_distance'] = float(root.find('.//ControllerVariables/StandoffDistance').text)
    config['hold_distance'] = float(root.find('.//ControllerVariables/HoldDistance').text)
    config['approach_speed'] = float(root.find('.//ControllerVariables/ApproachSpeed').text)
    
    config['TERMINAL_BOX_POS'] = float(root.find('.//ControllerVariables/TerminalBoxPos').text)
    config['TERMINAL_BOX_VEL'] = float(root.find('.//ControllerVariables/TerminalBoxVel').text)
    config['CONFIDENCE_WINDOW'] = int(root.find('.//ControllerVariables/ConfidenceWindow').text)
    config['CONFIDENCE_FRACTION'] = float(root.find('.//ControllerVariables/ConfidenceFraction').text)
    config['HK_THRESHOLD'] = float(root.find('.//ControllerVariables/HkThreshold').text)
    
    # CBF Variables
    config['cbf_enabled'] = root.find('.//CBFVariables/Enabled').text.lower() == 'true'
    config['p_target'] = parse_list(root.find('.//CBFVariables/PTarget').text)
    config['gamma'] = float(root.find('.//CBFVariables/Gamma').text)
    config['max_cbf_iter'] = int(root.find('.//CBFVariables/MaxIterations').text)
    
    # Noise & Observer Variables
    config['svo_horizon'] = int(root.find('.//NoiseObserverVariables/SvoHorizon').text)
    config['w_max_val'] = float(root.find('.//NoiseObserverVariables/WMaxVal').text)
    config['w_max_svo'] = float(root.find('.//NoiseObserverVariables/WMaxSvo').text)
    config['v_max_val'] = float(root.find('.//NoiseObserverVariables/VMaxVal').text)
    
    # Measurement Noise configuration
    config['pMatrix_pos_var'] = float(root.find('.//NoiseObserverVariables/PMatrixPosVar').text)
    config['pMatrix_mrp_var'] = float(root.find('.//NoiseObserverVariables/PMatrixMrpVar').text)
    config['pMatrix_rate_var'] = float(root.find('.//NoiseObserverVariables/PMatrixRateVar').text)
    config['pMatrix_vel_var'] = float(root.find('.//NoiseObserverVariables/PMatrixVelVar').text)
    
    config['walk_pos'] = float(root.find('.//NoiseObserverVariables/WalkBoundsPos').text)
    config['walk_mrp'] = float(root.find('.//NoiseObserverVariables/WalkBoundsMrp').text)
    config['walk_rate'] = float(root.find('.//NoiseObserverVariables/WalkBoundsRate').text)
    config['walk_vel'] = float(root.find('.//NoiseObserverVariables/WalkBoundsVel').text)

    return config

#time logs
def log_print(*args, **kwargs):
    # Fetch the current date and time
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Print the time, followed by whatever arguments were passed
    print(f"[{current_time}]", *args, **kwargs)

# Plotting functions
def plot_attitude_error(timeData, dataSigmaBR):
    """Plot the attitude errors."""
    plt.figure(1)
    for idx in range(3):
        plt.plot(timeData, dataSigmaBR[:, idx],
                 color=unitTestSupport.getLineColor(idx, 3),
                 label=r'$\sigma_' + str(idx) + '$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel(r'Attitude Error $\sigma_{B/R}$')


def plot_rw_motor_torque(timeData, dataUsReq, dataRW, numRW):
    """Plot the RW actual motor torques."""
    plt.figure(2)
    for idx in range(3):
        plt.plot(timeData, dataUsReq[:, idx],
                 '--',
                 color=unitTestSupport.getLineColor(idx, numRW),
                 label=r'$\hat u_{s,' + str(idx) + '}$')
        plt.plot(timeData, dataRW[idx],
                 color=unitTestSupport.getLineColor(idx, numRW),
                 label='$u_{s,' + str(idx) + '}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel('RW Motor Torque (Nm)')

def plot_rate_error(timeData, dataOmegaBR):
    """Plot the body angular velocity rate tracking errors."""
    plt.figure(3)
    for idx in range(3):
        plt.plot(timeData, dataOmegaBR[:, idx],
                 color=unitTestSupport.getLineColor(idx, 3),
                 label=r'$\omega_{BR,' + str(idx) + '}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel('Rate Tracking Error (rad/s) ')

def plot_rw_speeds(timeData, dataOmegaRW, numRW):
    """Plot the RW spin rates."""
    plt.figure(4)
    for idx in range(numRW):
        plt.plot(timeData, dataOmegaRW[:, idx] / macros.RPM,
                 color=unitTestSupport.getLineColor(idx, numRW),
                 label=r'$\Omega_{' + str(idx) + '}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel('RW Speed (RPM) ')

def plot_distance_and_thrust(timeData, distanceData, thrustData, relPosData):
    """Plot the distance, relative components, and thrust magnitude on shared axes."""
    fig = plt.figure(9, figsize=(10, 5))
    ax1 = fig.add_subplot(111)

    # Plot Total Distance and the X/Y/Z Components on the Left Y-Axis
    ax1.plot(timeData, distanceData, 'k', linewidth=2, label='Total Distance')
    ax1.plot(timeData, relPosData[:, 0], label='X (Chief - Servicer)')
    ax1.plot(timeData, relPosData[:, 1], label='Y (Chief - Servicer)')
    ax1.plot(timeData, relPosData[:, 2], label='Z (Chief - Servicer)')
    
    ax1.set_xlabel('Time [min]')
    ax1.set_ylabel('Distance / Position (m)')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Plot Thrust on the Right Y-Axis
    ax2 = ax1.twinx()  
    color2 = unitTestSupport.getLineColor(1, 3)
    ax2.plot(timeData, thrustData, '--', color=color2, label='Thrust Magnitude', alpha=0.7)
    ax2.set_ylabel('Thrust (N)', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Combine legends from both axes and move to top so they don't block the graph
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3)
    
    # Adjust layout so the legend fits
    plt.tight_layout()

def plot_cz_inclusion(timeData, inclusionData, slackMatchData, posErrorData, tReqData, hkData, N_horizon, svo_horizon, max_thrust, w_svo, v_svo):
    """Plot the tightest Constrained Zonotope, slack match, pos est error, Thrust norm."""
    
    # Calculate RMS for Position Error (t > 3 min)
    time_arr = np.array(timeData)
    error_arr = np.array(posErrorData)
    
    valid_idx = time_arr > 3.0
    if np.any(valid_idx):
        rms_error = np.sqrt(np.mean(error_arr[valid_idx]**2))
        rms_text = f"{rms_error:.4f} m"
    else:
        rms_text = "N/A (run too short)"

    fig = plt.figure(6, figsize=(14, 6)) # Made wider
    ax1 = fig.add_subplot(111)
    
    # Primary Y-Axis: discrete set transitions
    ax1.step(timeData, inclusionData, where='post', color='purple', linewidth=2, label='Current State Location')
    ax1.axhline(y=N_horizon, color='r', linestyle='--', label=f'Terminal Set (K_{N_horizon})')
    
    ax1.set_ylim(N_horizon - 11, N_horizon + 1)
    ax1.set_yticks(list(range(N_horizon - 10, N_horizon + 1)) + [N_horizon - 11], 
                   labels=[f"K_{i}" for i in range(N_horizon - 10, N_horizon + 1)] + ["Outside"])
    
    ax1.set_xlabel('Time [min]')
    ax1.set_ylabel('Tightest CZ Set Inclusion')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Secondary Y-Axis: Slack match variance
    ax2 = ax1.twinx()
    ax2.plot(timeData, hkData, color='dodgerblue', alpha=0.6, linewidth=1.5, label='h_k')
    ax2.set_ylabel('h_k variable magnitude', color='dodgerblue')
    ax2.tick_params(axis='y', labelcolor='dodgerblue')

    # Third Y-Axis: Position Estimation Error
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))      
    ax3.plot(timeData, posErrorData, color='darkorange', alpha=0.8, linewidth=1, linestyle='-', label='Pos Est Error (m)')
    ax3.set_ylabel('Position Estimation Error (m)', color='darkorange')
    ax3.tick_params(axis='y', labelcolor='darkorange')

    # Fourth Y-Axis: Applied Force
    ax4 = ax1.twinx()
    ax4.spines['right'].set_position(('outward', 120))      
    ax4.plot(timeData, tReqData, color='g', alpha=0.7, linewidth=0.8, linestyle='-', label='Applied Force (N)')
    ax4.set_ylabel('Applied Force Norm (N)', color='g')
    ax4.tick_params(axis='y', labelcolor='g')

    plt.title(f'Set Variance, Slack, Est Error, & Thrust, SVO Horizon = {svo_horizon}, Max Thrust = {max_thrust}, SVO w = {w_svo}, SVO V = {v_svo}\nRMS Pos Error = {rms_text}')
    
    # Combine legends from all axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3 + lines4, labels1 + labels2 + labels3 + labels4, loc='lower left')
    
    fig.tight_layout()

def plot_3d_relative_trajectory(true_state_history, L_chief, L_servicer, delta_disc, phase_id_history):
    import matplotlib.patches as mpatches
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    states = np.array(true_state_history)
    if len(states) == 0:
        return fig
        
    # Hill frame: X is Radial, Y is Along-track, Z is Cross-track
    x = states[:, 0]
    y = states[:, 1]
    z = states[:, 2]
    
    #  Plot the Servicer Trajectory
    ax.plot(x, y, z, color='steelblue', linewidth=2, label='Servicer Trajectory')
    ax.scatter(x[0], y[0], z[0], color='green', marker='o', s=80, zorder=5, label='Start')
    
    docked_indices = np.flatnonzero(
    np.asarray(phase_id_history[:len(states)]) == 5
    )

    if docked_indices.size:
        idx = docked_indices[0]
        ax.scatter(
            x[idx], y[idx], z[idx],
            color='purple', marker='*', s=150, zorder=5,
            label='Capture Position'
        )
        
    #  Setup 3D spheres
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    
    # Keep-Out Zone Sphere
    r_koz = L_chief + L_servicer + delta_disc
    x_koz = r_koz * np.outer(np.cos(u), np.sin(v))
    y_koz = r_koz * np.outer(np.sin(u), np.sin(v))
    z_koz = r_koz * np.outer(np.ones_like(u), np.cos(v))
    
    ax.plot_surface(x_koz, y_koz, z_koz, color='firebrick', alpha=0.1, 
                    edgecolor='firebrick', linewidth=0.2)
                    
    # Draw Chief Body Sphere
    x_chief = L_chief * np.outer(np.cos(u), np.sin(v))
    y_chief = L_chief * np.outer(np.sin(u), np.sin(v))
    z_chief = L_chief * np.outer(np.ones_like(u), np.cos(v))
    
    ax.plot_surface(x_chief, y_chief, z_chief, color='black', alpha=0.3,
                    edgecolor='black', linewidth=0.2)

    ax.set_xlabel('Radial (m)')
    ax.set_ylabel('Along-track (m)')
    ax.set_zlabel('Cross-track (m)')
    ax.set_title('3D Servicer Trajectory vs. CBF Keep-Out Zone')
    
    # Enforce equal aspect ratio in 3D (crucial for spheres)
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    handles, labels = ax.get_legend_handles_labels()
    koz_patch = mpatches.Patch(color='firebrick', alpha=0.2, label=f'CBF Keep-Out Zone')
    chief_patch = mpatches.Patch(color='black', alpha=0.5, label=f'Chief Body ')
    handles.extend([koz_patch, chief_patch])
    
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    
    plt.tight_layout()
    return fig

def plot_estimation_error(timeData, est_states, true_states, svo_bounds, svo_horizon):
    """Plots the difference between the SVO estimated state and the true Basilisk state, 
    overlaid with the exact CCG upper and lower bounds."""
    est_states = np.array(est_states)
    true_states = np.array(true_states)
    error = est_states - true_states
    
    # svo_bounds shape: (N, 6, 2) -> (Time, States, Min/Max)
    if svo_bounds is not None:
        svo_bounds = np.array(svo_bounds)
    
    fig = plt.figure(8, figsize=(12, 10))
    axes = fig.subplots(2, 1, sharex=True)
    
    colors = ['steelblue', 'darkorange', 'forestgreen']
    labels_p = ['X', 'Y', 'Z']
    labels_v = ['Vx', 'Vy', 'Vz']
    
    # Position Error Panel 
    for i in range(3):
        axes[0].plot(timeData, error[:, i], color=colors[i], linewidth=2, label=f'{labels_p[i]} Error')
        
        if svo_bounds is not None:
            # The bounds are relative to the center (x_est), so they map directly onto the error (x_est - x_true = 0)
            axes[0].fill_between(timeData, svo_bounds[:, i, 0], svo_bounds[:, i, 1], 
                                 color=colors[i], alpha=0.15, label=f'{labels_p[i]} CCG Bound')
            axes[0].plot(timeData, svo_bounds[:, i, 0], color=colors[i], linestyle='--', alpha=0.5, linewidth=1)
            axes[0].plot(timeData, svo_bounds[:, i, 1], color=colors[i], linestyle='--', alpha=0.5, linewidth=1)
            
    axes[0].set_ylabel('Position Error (m)')
    axes[0].set_title(f'SVO Position Estimation Error vs True State, SVO Horizon = {svo_horizon}')
    
    # Deduplicate legend items
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[0].legend(by_label.values(), by_label.keys(), loc='upper right', ncol=3)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # --- Velocity Error Panel ---
    for i in range(3):
        axes[1].plot(timeData, error[:, i+3], color=colors[i], linewidth=2, label=f'{labels_v[i]} Error')
        
        if svo_bounds is not None:
            axes[1].fill_between(timeData, svo_bounds[:, i+3, 0], svo_bounds[:, i+3, 1], 
                                 color=colors[i], alpha=0.15, label=f'{labels_v[i]} CCG Bound')
            axes[1].plot(timeData, svo_bounds[:, i+3, 0], color=colors[i], linestyle='--', alpha=0.5, linewidth=1)
            axes[1].plot(timeData, svo_bounds[:, i+3, 1], color=colors[i], linestyle='--', alpha=0.5, linewidth=1)
            
    axes[1].set_ylabel('Velocity Error (m/s)')
    axes[1].set_xlabel('Time [min]')
    axes[1].set_title(f'SVO Velocity Estimation Error vs True State')
    
    handles, labels = axes[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[1].legend(by_label.values(), by_label.keys(), loc='upper right', ncol=3)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    # Clip Y-axis to ignore initial transients and focus on steady-state
    if len(error) > 60:
        steady_pos = error[60:, 0:3]
        steady_vel = error[60:, 3:6]
        max_pos = np.max(np.abs(steady_pos))
        max_vel = np.max(np.abs(steady_vel))
        
        if svo_bounds is not None and not np.all(np.isnan(svo_bounds[60:])):
            max_pos = max(max_pos, np.nanmax(np.abs(svo_bounds[60:, 0:3, :])))
            max_vel = max(max_vel, np.nanmax(np.abs(svo_bounds[60:, 3:6, :])))
            
        axes[0].set_ylim(-max_pos * 1.5, max_pos * 1.5)
        axes[1].set_ylim(-max_vel * 1.5, max_vel * 1.5)

    plt.tight_layout()
    return fig

def state_inclusion(state_vec, K_set):
    """ Evaluates if a state vector lies within a Constrained Zonotope using an LP """
    #Pulling the asked set
    G = K_set['G']
    c = K_set['c'].flatten()
    A_k = K_set['A']
    b_k = K_set['b']
    
    ng = G.shape[1]
    c_obj = np.zeros(ng) 
    
    if A_k.size > 0:
        A_eq = np.vstack((G, A_k))
        b_eq = np.concatenate((state_vec - c, b_k.flatten()))
    else:
        A_eq = G
        b_eq = state_vec - c
        
    bounds = [(-1.0, 1.0)] * ng
    #determining if there is a solution
    res = linprog(  c_obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs',
        options={'maxiter': 150, 'time_limit': 0.2, 'primal_feasibility_tolerance': 1e-4} )
    if res.status not in (0, 2):
        return False
    return res.status == 0          # 2 = genuinely infeasible

def calculate_set_scaling(state_vec, K_set):
    """ Evaluates the exact set-scaling value t such that state_vec in c + G*xi, ||xi||_inf <= t, A*xi = b """
    G = K_set['G']
    c = K_set['c'].flatten()
    A_k = K_set['A']
    b_k = K_set['b']
    
    ng = G.shape[1]
    c_obj = np.zeros(ng + 1)
    c_obj[-1] = 1.0 # Minimize t (the uniform scaling factor)
    
    A_eq_G = np.hstack((G, np.zeros((G.shape[0], 1))))
    b_eq_G = state_vec - c
    
    if A_k.size > 0:
        A_eq_A = np.hstack((A_k, np.zeros((A_k.shape[0], 1))))
        b_eq_A = b_k.flatten()
        A_eq = np.vstack((A_eq_G, A_eq_A))
        b_eq = np.concatenate((b_eq_G, b_eq_A))
    else:
        A_eq = A_eq_G
        b_eq = b_eq_G
        
    # Inequality constraints: xi - t <= 0,  -xi - t <= 0
    I_ng = np.eye(ng)
    ones_col = -np.ones((ng, 1))
    A_ub = np.vstack((np.hstack((I_ng, ones_col)), np.hstack((-I_ng, ones_col))))
    b_ub = np.zeros(2 * ng)
    
    bounds = [(None, None)] * ng + [(0.0, None)]
    
    res = linprog(c_obj, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    return res.x[-1] if res.success else np.nan

def plot_cbf_safety_score(timeData, h_history, h_error_time, h_error_history, safety_margin=0.05):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    min_len = min(len(timeData), len(h_history))
    t = np.array(timeData[:min_len])
    h = np.array(h_history[:min_len])
    
    min_len_err = min(len(h_error_time), len(h_error_history))
    t_err = np.array(h_error_time[:min_len_err])
    h_err = np.array(h_error_history[:min_len_err])

    # Panel 1: Full Range (Symlog Scale) 
    ax1.plot(t, h, label=r'CBF Safety Score ($h_k$)', color='dodgerblue', linewidth=2)
    ax1.axhline(0, color='red', linestyle='--', linewidth=2, label='Absolute Collision Boundary (h=0)')
    ax1.axhline(safety_margin, color='orange', linestyle=':', linewidth=2, label='Safety Margin Floor')
    
    if len(h_err) > 0:
        # Match interpolation to calculate Relative Error for the macroscopic view
        h_matched = np.interp(t_err, t, h)
        rel_err = np.abs(h_err) / np.maximum(h_matched, 1e-3)
        
        ax1_rel = ax1.twinx()
        ax1_rel.plot(t_err, rel_err, label='Relative Pred Error', color='purple', linewidth=1.5, linestyle='-.', alpha=0.5)
        ax1_rel.set_ylabel('Relative Prediction Error $|h_{pred} - h_k| / h_k$', color='purple')
        ax1_rel.tick_params(axis='y', labelcolor='purple')
        
    ax1.set_yscale('symlog', linthresh=0.1) # Compresses everything above 0.1 logarithmically
    ax1.set_ylim(-0.1, max(20.0, np.percentile(h, 98) if len(h)>0 else 50.0))
    ax1.set_xlabel('Time [min]')
    ax1.set_ylabel('CBF Value (Symlog Scale)')
    ax1.set_title('Full Mission Overview (Logarithmic Scaling)')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    handles1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(handles1, labels1, loc='upper right')

    #  Panel 2: Zoomed Final Approach (Linear Scale) 
    zoom_t_thresh = 2.0
    zoom_idx = t > zoom_t_thresh
    
    if not np.any(zoom_idx):
        zoom_idx = t >= 0.0 # Fallback if run was short

    ax2.plot(t[zoom_idx], h[zoom_idx], label=r'CBF Safety Score ($h_k$)', color='dodgerblue', linewidth=2)
    ax2.axhline(0, color='red', linestyle='--', linewidth=2, label='Absolute Collision Boundary (h=0)')
    ax2.axhline(safety_margin, color='orange', linestyle=':', linewidth=2, label='Safety Margin Floor')
    
    if len(h_err) > 0:
        zoom_err_idx = t_err > zoom_t_thresh
        ax2.plot(t_err[zoom_err_idx], h_err[zoom_err_idx], 
                 label='Raw Prediction Error ($h_{pred} - h_k$)', color='purple', linewidth=1.5, linestyle='-.')

    # Lock Y-axis tight to the boundary
    y_max_zoom = max(safety_margin * 5, np.percentile(h[zoom_idx], 95) if len(h[zoom_idx]) > 0 else 0.5)
    y_min_zoom = min(-0.02, np.min(h[zoom_idx]) - 0.01 if len(h[zoom_idx]) > 0 else -0.02)
    
    ax2.set_ylim(y_min_zoom, y_max_zoom)
    ax2.set_xlabel('Time [min]')
    ax2.set_ylabel('CBF Value & Error (Linear Scale)')
    ax2.set_title(f'Zoom: Proximity Approach ($t > {zoom_t_thresh}$ min)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles2, labels2, loc='upper right')

    fig.suptitle('CBF Evaluation and Forward Invariance Verification', fontsize=15)
    fig.tight_layout()
    
    return fig

def plot_hill_frame_traj(true_state_history, L_chief, L_servicer, delta_disc, phase_id_history):
    fig, ax = plt.subplots(figsize=(8, 8))
    states = np.array(true_state_history)

    if len(states) == 0:
        return fig

    # Hill frame: Index 0 is Radial, Index 1 is Along-track
    radial = states[:, 0]
    along_track = states[:, 1]

    # Plot the servicer's path
    ax.plot(along_track, radial, color='steelblue', linewidth=2, label='Servicer Trajectory')
    ax.scatter(along_track[0], radial[0], color='green', marker='o', s=80, zorder=4, label='Start')
    docked_indices = np.flatnonzero(
    np.asarray(phase_id_history[:len(states)]) == 5
    )

    if docked_indices.size:
        idx = docked_indices[0]
        ax.scatter(
            along_track[idx], radial[idx],
            color='purple', marker='*', s=150, zorder=5,
            label='Capture Position'
        )

    # Draw the mathematical Keep-Out Zone (Minkowski sum of both radii)
    exclusion_radius = L_chief + L_servicer + delta_disc
    keepout_circle = plt.Circle((0, 0), exclusion_radius, color='firebrick', alpha=0.2, label=f'CBF Keep-Out Zone ({exclusion_radius}m)')
    
    # Draw the physical Chief satellite
    chief_body = plt.Circle((0, 0), L_chief, color='black', alpha=0.8, label=f'Chief Body ({L_chief}m)')

    ax.add_patch(keepout_circle)
    ax.add_patch(chief_body)

    ax.set_xlabel('Along-track (m)')
    ax.set_ylabel('Radial (m)')
    ax.set_title('Servicer Trajectory vs. CBF Keep-Out Zone')
    ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='black', linewidth=0.5, alpha=0.5)
    
    # Force equal aspect ratio so the circles don't distort into ellipses
    ax.axis('equal')  
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='best')
    plt.tight_layout()
    return fig

def solve_intercept(rc, vc, r_t0, v_t0, s, port_C, dcm_CN, omega_C, T0):
    w = vc - v_t0 #relative velocity before the tether is fired
    T = T0 #initial guess
    for _ in range(5):
        dcm_fut = rbk.PRV2C(omega_C * T) @ dcm_CN #Rotated the chief
        q = rc + dcm_fut.T @ port_C - r_t0
        a, b, c = w @ w - s*s, 2.0 * (q @ w), q @ q
        T = (-b - np.sqrt(b*b - 4*a*c)) / (2*a)   # a<0 -> unique positive root
    u = (q + w*T) / (s*T)
    return u / np.linalg.norm(u), float(T), q, w

def run(show_plots, setup_name = None, cfg_overrides=None):
    """
    Args:
        show_plots (bool): Determines if the script should display plots
        setup_name (str, optional): Prefix or custom name for the saved .npz file.
        cfg_overrides (dict, optional): Dictionary of config parameters to override.
    """
    cfg = load_config('config.xml')

    if cfg_overrides:
        base_step_time = cfg['step_time']
        cfg.update(cfg_overrides)
        if 'step_time' in cfg_overrides and 'CONFIDENCE_WINDOW' not in cfg_overrides:
            hold_window_sec = cfg['CONFIDENCE_WINDOW'] * base_step_time
            cfg['CONFIDENCE_WINDOW'] = max(5, int(round(hold_window_sec / cfg['step_time'])))
        if 'horizon_time' in cfg_overrides or 'step_time' in cfg_overrides:
            h_time = cfg_overrides.get('horizon_time', cfg['N_horizon'] * base_step_time)
            cfg['N_horizon'] = int(h_time) // int(cfg['step_time'])
            cfg['CLOSE_ENTER_IDX'] = cfg['N_horizon']
            cfg['CLOSE_EXIT_IDX'] = cfg['N_horizon'] - 3
            cfg['TRANSFER_STEPS'] = int(np.ceil(3.0 / cfg['step_time']))

    # Short aliases for clean filenames
    KEY_ALIASES = {
        'svo_horizon': 'Nsvo',
        'step_time': 'Ts',
        'horizon_time': 'Th',
        'N_horizon': 'Nmpc',
        'w_max_svo': 'wSvo',
        'w_max_val': 'wVal',
        'v_max_val': 'vMax',
        'pMatrix_pos_var': 'pPos',
        'walk_pos': 'walkPos',
        'max_thrust': 'Fmax',
        'gamma': 'gam',
        'HK_THRESHOLD': 'hk',
        'pos_only_sensing': 'PosOnly',
        'pMatrix_vel_var': 'pVel',
        'walk_vel': 'walkVel',
        'noise_mode': 'nMode',
        'ext_noise_amp': 'nAmp',
        'rng_seed': 'seed',
        'delta_disc_override': 'dDisc',
        'alpha_cbf_1': 'alpha'
    }

    def fmt_val(v):
        if isinstance(v, float):
            return f"{v:g}"
        elif isinstance(v, (list, tuple, np.ndarray)):
            # Round lists/arrays to 3 decimal places to prevent massive filenames
            return "[" + ",".join([f"{x:.3f}" for x in v]) + "]"
        else:
            return str(v)

    # Auto-build setup_name from cfg_overrides (or baseline config if no overrides)
    if cfg_overrides:
        override_tag = "_".join(
            f"{KEY_ALIASES.get(k, k)}{fmt_val(v)}" for k, v in cfg_overrides.items()
        )
        setup_name = f"{setup_name}_{override_tag}" if setup_name else override_tag
    elif not setup_name or setup_name == "trash":
        setup_name = (
            f"base_Nsvo{cfg['svo_horizon']}_Ts{cfg['step_time']}"
            f"_Nmpc{cfg['N_horizon']}_wSvo{fmt_val(cfg['w_max_svo'])}_vMax{fmt_val(cfg['v_max_val'])}"
        )

    print(f"--- Running simulation: sim_data_{setup_name}.npz ---")

    rng_seed = int(cfg.get('rng_seed', 42))
    np.random.seed(rng_seed)    

    # Spacecraft Properties
    servicer_mass = cfg['servicer_mass']
    servicer_I = cfg['servicer_I']
    L_servicer = cfg['L_servicer']

    chief_mass = cfg['chief_mass']
    chief_I = cfg['chief_I']
    L_chief = cfg['L_chief']

    m_tip = cfg['m_tip']
    r_tip = cfg['r_tip']
    h_tip = cfg['h_tip']
    F_impact = cfg['F_impact']
    t_contact = cfg['t_contact']
    tip_offset = cfg['tip_offset']
    CAPTURE_POS_TOL = cfg['CAPTURE_POS_TOL']
    CAPTURE_VEL_TOL = cfg['CAPTURE_VEL_TOL']

    # Controller Variables
    max_thrust = cfg['max_thrust']
    step_time = cfg['step_time']
    N_horizon = cfg['N_horizon']
    CLOSE_ENTER_IDX = cfg['CLOSE_ENTER_IDX']
    CLOSE_EXIT_IDX = cfg['CLOSE_EXIT_IDX']
    TRANSFER_STEPS = cfg['TRANSFER_STEPS']
    standoff_distance = cfg['standoff_distance']
    hold_distance = cfg['hold_distance']
    approach_speed = cfg['approach_speed']
    TERMINAL_BOX_POS = cfg['TERMINAL_BOX_POS']
    TERMINAL_BOX_VEL = cfg['TERMINAL_BOX_VEL']
    CONFIDENCE_WINDOW = cfg['CONFIDENCE_WINDOW']
    CONFIDENCE_FRACTION = cfg['CONFIDENCE_FRACTION']
    HK_THRESHOLD = cfg['HK_THRESHOLD']
    v_max = 1

    # CBF Variables
    cbf_enabled = cfg['cbf_enabled']
    p_target = cfg['p_target']
    gamma = cfg['gamma']
    max_cbf_iter = cfg['max_cbf_iter']

    #  Noise & Observer Variables
    svo_horizon = cfg['svo_horizon']
    w_max_val = cfg['w_max_val']
    w_max_svo = cfg['w_max_svo']
    v_max_val = cfg['v_max_val']

    pMatrix_pos_var = cfg['pMatrix_pos_var']
    pMatrix_mrp_var = cfg['pMatrix_mrp_var']
    pMatrix_rate_var = cfg['pMatrix_rate_var']
    pMatrix_vel_var = cfg['pMatrix_vel_var']

    walk_pos = cfg['walk_pos']
    walk_vel = cfg['walk_vel']
    walk_mrp = cfg['walk_mrp']
    walk_rate = cfg['walk_rate']

    # Tip mass derived kinematics
    v_launch = (F_impact * t_contact) / m_tip 
    I_tip_xx = (1/12) * m_tip * (3*r_tip**2 + h_tip**2)
    I_tip_zz = (1/2) * m_tip * r_tip**2
    I_tip_flat = [I_tip_xx, 0.0, 0.0,
                0.0, I_tip_xx, 0.0,
                0.0, 0.0, I_tip_zz]


    # State machine: 'approach_hold_point' -> 'holding' -> 'approach_dock' -> 'tether_flight' -> 'transfer' -> 'docked'
    docking_phase = 'approach_hold_point'   

    current_target_distance = hold_distance
    L_servicer_standoff = L_servicer + standoff_distance   
    L_servicer_virtual = L_servicer + hold_distance

    # Set up flight modes
    def fswHillPointing():
        attError.attRefInMsg.subscribeTo(hillPointing.attRefOutMsg)
        attError.sigma_R0R = [0.0, 0.0, -0.414214]

    def runSim(simTime):
        nonlocal simulationTime
        simulationTime += macros.sec2nano(simTime)
        scSim.ConfigureStopTime(simulationTime)
        scSim.ExecuteSimulation()

    def relOrbDrift():
        rd = unitTestSupport.EigenVector3d2np(servicerPos.getState())
        vd = unitTestSupport.EigenVector3d2np(servicerVel.getState())
        rc = unitTestSupport.EigenVector3d2np(debrisPos.getState())
        vc = unitTestSupport.EigenVector3d2np(debrisVel.getState())
        rho_H, rho_Prime_H = orbitalMotion.rv2hill(rc, vc, rd, vd)
        xOff = rho_H[0]
        rho_Prime_H[0] = 0.0
        rho_Prime_H[1] = -1.5*n*xOff
        unusedPos, vd = orbitalMotion.hill2rv(rc, vc, rho_H, rho_Prime_H)
        servicerVel.setState(vd)

    def get_current_error():
        """ Calculates the true error relative to the Standoff Point (Final MPC Target) """
        rd = unitTestSupport.EigenVector3d2np(servicerPos.getState())
        vd = unitTestSupport.EigenVector3d2np(servicerVel.getState())
        rc = unitTestSupport.EigenVector3d2np(debrisPos.getState())
        vc = unitTestSupport.EigenVector3d2np(debrisVel.getState())
        
        rho_H, rho_Prime_H = orbitalMotion.rv2hill(rc, vc, rd, vd)
        debrisSigma = scObject2.dynManager.getStateObject(scObject2.hub.nameOfHubSigma)
        debrisOmega = scObject2.dynManager.getStateObject(scObject2.hub.nameOfHubOmega)
        sigma_CN = unitTestSupport.EigenVector3d2np(debrisSigma.getState())
        omega_CN_B = unitTestSupport.EigenVector3d2np(debrisOmega.getState())

        # Build DCMs
        rc_norm = np.linalg.norm(rc)
        h_vec = np.cross(rc, vc)
        r_hat = rc / rc_norm
        h_hat = h_vec / np.linalg.norm(h_vec)
        theta_hat = np.cross(h_hat, r_hat)
        dcm_NH = np.array([r_hat, theta_hat, h_hat]).T
        dcm_CN = rbk.MRP2C(sigma_CN)
        
        # Calculate Port Positions in Hill Frame
        port_C = np.array([0.0, 0.0, -L_chief])
        port_N = dcm_CN.T.dot(port_C)
        port_H_chief = dcm_NH.T.dot(port_N)
        
        port_offset = -port_H_chief * (L_servicer_standoff / L_chief)
        port_error = rho_H + port_offset - port_H_chief
        
        omega_N = dcm_CN.T.dot(omega_CN_B).flatten()
        v_N_k = np.cross(omega_N, port_N)
        v_H_k = dcm_NH.T.dot(v_N_k)
        vel_error = rho_Prime_H - v_H_k * ((L_chief + L_servicer_standoff) / L_chief)
        
        return np.concatenate((port_error, vel_error))


    def get_true_relative_state():
            """ True Hill-frame relative position/velocity (no target/port offset). """
            rd_true = unitTestSupport.EigenVector3d2np(servicerPos.getState())
            vd_true = unitTestSupport.EigenVector3d2np(servicerVel.getState())
            rc_true = unitTestSupport.EigenVector3d2np(debrisPos.getState())
            vc_true = unitTestSupport.EigenVector3d2np(debrisVel.getState())
            rho_H_true, rho_Prime_H_true = orbitalMotion.rv2hill(rc_true, vc_true, rd_true, vd_true)
            return np.concatenate((rho_H_true, rho_Prime_H_true))

    def compute_rc_sets(Ad, Bd, N_horizon, u_max, w_max_val):
        nx = 6
        nu = 3
        
        c_cz = np.zeros((nx, 1))
        # Terminal set capped at exactly 0.4 m and 0.4 m/s per axis
        G_curr = np.diag([TERMINAL_BOX_POS]*3 + [TERMINAL_BOX_VEL]*3)
        A_curr = np.array([[]])
        b_curr = np.array([])
        
        G_w = Bd * w_max_val
        G_U_action = -Bd @ (u_max * np.eye(nu))
        Ad_inv = np.linalg.inv(Ad)
        
        K_sets = [None] * (N_horizon + 1)
        K_sets[N_horizon] = {'G': G_curr, 'c': c_cz, 'A': A_curr, 'b': b_curr}
        
        # The L2 inscribed box bound to mathematically guarantee ||v||_2 <= 0.15
        v_bound = v_max # max vel per axis
        
        for t in range(N_horizon - 1, -1, -1):
            G_curr = K_sets[t+1]['G']
            c_curr = K_sets[t+1]['c']
            A_curr = K_sets[t+1]['A']
            b_curr = K_sets[t+1]['b']
            
            N_K = G_curr.shape[1]
            D_K = np.zeros((N_K, N_K))
            
            # 1. Shrink step (Pontryagin Difference)
            if A_curr.size == 0:
                inv_matrix = G_curr
                W_matrix = G_w
            else:
                inv_matrix = np.vstack((G_curr, A_curr))
                W_matrix = np.vstack((G_w, np.zeros((A_curr.shape[0], G_w.shape[1]))))
                
            M_inv_K = linalg.pinv(inv_matrix)
            
            for i in range(N_K):
                row_vec = M_inv_K[i, :] @ W_matrix
                shrink_factor = 1.0 - np.sum(np.abs(row_vec))
                if shrink_factor < 0:
                    print(f"WARNING: Pontryagin difference failed at step {t}. Disturbance too large.")
                    shrink_factor = 0.0
                D_K[i, i] = shrink_factor
                
            G_shrunk = G_curr @ D_K
            A_shrunk = A_curr @ D_K if A_curr.size > 0 else np.array([[]])
            
            G_sum = np.hstack((G_shrunk, G_U_action))
            c_sum = c_curr
            
            if A_shrunk.size == 0:
                A_sum = np.array([[]])
            else:
                A_sum = np.hstack((A_shrunk, np.zeros((A_shrunk.shape[0], G_U_action.shape[1]))))
            b_sum = b_curr
            
            # 2. Base Backward Step
            c_back = Ad_inv @ c_sum
            G_back = Ad_inv @ G_sum
            
            # 3. Velocity Intersection (Box limit)
            G_v = G_back[3:6, :]
            c_v = c_back[3:6, 0]
            
            G_append = np.zeros((nx, 3))
            G_back = np.hstack((G_back, G_append))
            
            A_intersect = np.hstack((G_v, v_bound * np.eye(3)))
            b_intersect = -c_v
            
            if A_sum.size == 0:
                A_sum = A_intersect
                b_sum = b_intersect
            else:
                A_sum = np.hstack((A_sum, np.zeros((A_sum.shape[0], 3))))
                A_sum = np.vstack((A_sum, A_intersect))
                b_sum = np.concatenate((b_sum, b_intersect))
            
            # NO ORDER REDUCTION: Exact geometry is preserved.
            K_sets[t] = {'G': G_back, 'c': c_back, 'A': A_sum, 'b': b_sum}
            
        return K_sets

    def port_state_N(parentPosObj, parentVelObj, parentSigmaObj, parentOmegaObj,
                    L_port, rho=0.0, rhoDot=0.0):
        ''' Helper for the tips kinematics'''
        r_BN_N = unitTestSupport.EigenVector3d2np(parentPosObj.getState()).flatten()
        v_BN_N = unitTestSupport.EigenVector3d2np(parentVelObj.getState()).flatten()
        sigma_BN = unitTestSupport.EigenVector3d2np(parentSigmaObj.getState()).flatten()
        omega_BN_B = unitTestSupport.EigenVector3d2np(parentOmegaObj.getState()).flatten()

        dcm_NB = rbk.MRP2C(sigma_BN).T # body -> inertial

        r_PB_B = np.array([0.0, 0.0, -(L_port + rho)]) # port + boom offset, body frame
        rPrime_PB_B = np.array([0.0, 0.0, -rhoDot])

        r_PN_N = r_BN_N + dcm_NB.dot(r_PB_B)
        v_PN_N = v_BN_N + dcm_NB.dot(np.cross(omega_BN_B, r_PB_B) + rPrime_PB_B)

        # tip mass doesn't spin relative to its current parent -> same attitude and body rate
        return r_PN_N, v_PN_N, sigma_BN.copy(), omega_BN_B.copy()

    def slave_tip(r_PN_N, v_PN_N, sigma_PN, omega_PN_B):
        """Function to drag the tip mass and define inteded states"""
        tipPos.setState(r_PN_N)
        tipVel.setState(v_PN_N)
        tipSigma.setState(sigma_PN)
        tipOmega.setState(omega_PN_B)

    def svo_bounds_history_log(svo_tuple):
        "Filling buffer to plot CCG bounds against estimation error"

        bounds_k = np.zeros((6, 2))
        bounds_xi = [(-1.0, 1.0)] * (ng_p + ng_e)
        
        has_eq = (svo_tuple.A_rows > 0) and (A_curr.size > 0)
        A_eq_lp = A_curr.reshape((svo_tuple.A_rows, svo_tuple.A_cols)) if has_eq else None
        b_eq_lp = b_curr.flatten() if has_eq else None

        ok = True
        lp_opts = {'maxiter': 150, 'time_limit': 0.2, 'primal_feasibility_tolerance': 1e-5}
        if has_eq:
            for dim in range(6):
                r_hi = linprog(-G_curr[dim], A_eq=A_eq_lp, b_eq=b_eq_lp, bounds=bounds_xi, method='highs', options = lp_opts)
                r_lo = linprog( G_curr[dim], A_eq=A_eq_lp, b_eq=b_eq_lp, bounds=bounds_xi, method='highs', options = lp_opts)
                if not (r_hi.success and r_lo.success):
                    ok = False
                    break
                bounds_k[dim] = [r_lo.fun, -r_hi.fun]
        else:
            # Unconstrained zonotope: exact bounds are +/- row L1 norms
            r_l1 = np.sum(np.abs(G_curr), axis=1)
            bounds_k[:, 0] = -r_l1
            bounds_k[:, 1] =  r_l1

        if not ok:
            bounds_k[:] = np.nan                                              # empty set: don't invent bounds
        svo_bounds_history.append(bounds_k + (c_curr - true_state)[:, None])  # 0 inside band <=> truth inside set
        return bounds_k

    # Create simulation variable names
    simTaskName = "simTask"
    simProcessName = "simProcess"

    #  Create a sim module as an empty container
    scSim = SimulationBaseClass.SimBaseClass()
    scSim.SetProgressBar(False)

    # set the simulation time variable used later on
    simulationTime = 0

    #
    #  create the simulation process
    #
    dynProcess = scSim.CreateNewProcess(simProcessName)

    # create the dynamics task and specify the integration update time
    simulationTimeStep = macros.sec2nano(1)
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))

    #
    #   setup the simulation tasks/objects
    #

    # initialize servicer spacecraft object and set properties
    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "servicer"
    # define the simulation inertia
    scObject.hub.mHub = servicer_mass  # kg - spacecraft mass
    scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(servicer_I)

    # create the debris object states
    scObject2 = spacecraft.Spacecraft()
    scObject2.ModelTag = "debris"
    # define the simulation inertia
    scObject2.hub.mHub = chief_mass  # kg
    scObject2.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(chief_I)

    # add spacecraft object to the simulation process
    scSim.AddModelToTask(simTaskName, scObject, 1)
    scSim.AddModelToTask(simTaskName, scObject2, 2)

    # clear prior gravitational body and SPICE setup definitions
    gravFactory = simIncludeGravBody.gravBodyFactory()

    # setup Earth Gravity Body
    # earth = gravFactory.createEarth()
    # earth.isCentralBody = True  # ensure this is the central gravitational body
    gravBodies = gravFactory.createBodies('sun', 'earth')
    gravBodies['earth'].isCentralBody = True
    mu = gravBodies['earth'].mu
    sunIdx = 0
    earthIdx = 1

    # attach gravity model to spacecraft
    gravFactory.addBodiesTo(scObject)
    gravFactory.addBodiesTo(scObject2)

    # setup SPICE interface for celestial objects
    timeInitString = "2022 MAY 1 00:28:30.0"
    spiceObject = gravFactory.createSpiceInterface(time=timeInitString, epochInMsg=True)
    spiceObject.zeroBase = 'Earth'
    scSim.AddModelToTask(simTaskName, spiceObject)

    #
    # add RW devices
    #
    # Make a fresh RW factory instance, this is critical to run multiple times
    rwFactory = simIncludeRW.rwFactory()

    # store the RW dynamical model type
    varRWModel = messaging.BalancedWheels

    # create each RW by specifying the RW type, the spin axis gsHat, plus optional arguments
    RW1 = rwFactory.create('Honeywell_HR16', [1, 0, 0], maxMomentum=50., Omega=100.  # RPM
                           , RWModel=varRWModel
                           )
    RW2 = rwFactory.create('Honeywell_HR16', [0, 1, 0], maxMomentum=50., Omega=200.  # RPM
                           , RWModel=varRWModel
                           )
    RW3 = rwFactory.create('Honeywell_HR16', [0, 0, 1], maxMomentum=50., Omega=300.  # RPM
                           , rWB_B=[0.5, 0.5, 0.5]  # meters
                           , RWModel=varRWModel
                           )  

    numRW = rwFactory.getNumOfDevices()

    # create RW object container and tie to spacecraft object
    # make sure the input and output names are unique to this vehicle
    rwStateEffector = reactionWheelStateEffector.ReactionWheelStateEffector()
    rwFactory.addToSpacecraft("chiefRW", rwStateEffector, scObject)

    # add RW object array to the simulation process
    scSim.AddModelToTask(simTaskName, rwStateEffector, 4)

    # add the simple Navigation sensor module.  This sets the SC attitude, rate, position
    # velocity navigation message
    sNavObject_servicer = simpleNav.SimpleNav()
    sNavObject_servicer.ModelTag = "SimpleNavigation"
    sNavObject_servicer.scStateInMsg.subscribeTo(scObject.scStateOutMsg)

    # Create 18-dimension noise arrays
    pMatrix_BSK = np.zeros((18, 18)) # covariance matrix
    # Only add variance to the 3 position states
    np.fill_diagonal(pMatrix_BSK[0:3, 0:3], pMatrix_pos_var)  
    np.fill_diagonal(pMatrix_BSK[3:6, 3:6], pMatrix_vel_var)  # Velocity noise variance (Exp C)
    
    #  Attitude MRP Noise (Indices 6:9)
    # 1e-6 variance -> standard deviation of 0.001 on the MRP components
    np.fill_diagonal(pMatrix_BSK[6:9, 6:9], pMatrix_mrp_var)   
    
    # Rate Noise (9:12)
    # 1e-8 variance -> standard deviation of 0.0001 rad/s (sqrt)
    np.fill_diagonal(pMatrix_BSK[9:12, 9:12], pMatrix_rate_var)
    
    # Random walk bounds -> max deviation
    walkBounds_BSK = [
    [walk_pos], [walk_pos], [walk_pos],    # Position
    [walk_vel], [walk_vel], [walk_vel],    # Velocity
    [walk_mrp], [walk_mrp], [walk_mrp],    # Attitude MRPs
    [walk_rate], [walk_rate], [walk_rate], # Rates
    [0.0], [0.0], [0.0],                   # Sun pointing
    [0.0], [0.0], [0.0]                    # Delta-V
    ]
    
    # Add noise to Servicer
    sNavObject_servicer.walkBounds = walkBounds_BSK
    sNavObject_servicer.PMatrix = pMatrix_BSK.tolist()
    sNavObject_servicer.RNGSeed = rng_seed  # Locks the C++ Gauss-Markov process
    scSim.AddModelToTask(simTaskName, sNavObject_servicer)

    sNavObject_chief = simpleNav.SimpleNav()
    sNavObject_chief.ModelTag = "SimpleNavigation2"
    sNavObject_chief.scStateInMsg.subscribeTo(scObject2.scStateOutMsg)
    
    # Add noise to Chief
    sNavObject_chief.walkBounds = walkBounds_BSK
    sNavObject_chief.PMatrix = pMatrix_BSK.tolist()
    sNavObject_chief.RNGSeed = rng_seed + 31 # Use a different seed so the chief's noise is uncorrelated
    scSim.AddModelToTask(simTaskName, sNavObject_chief)
    # Create an ephemeris converter to convert messages of type
    # 'SpicePlanetStateMsgPayload' to 'EphemerisMsgPayload'
    ephemObject = ephemerisConverter.EphemerisConverter()
    ephemObject.ModelTag = 'EphemData'
    ephemObject.addSpiceInputMsg(spiceObject.planetStateOutMsgs[sunIdx])
    ephemObject.addSpiceInputMsg(spiceObject.planetStateOutMsgs[earthIdx])
    scSim.AddModelToTask(simTaskName, ephemObject)

    # setup hillPoint guidance module
    hillPointing = hillPoint.hillPoint()
    hillPointing.ModelTag = "hillPoint"
    hillPointing.transNavInMsg.subscribeTo(sNavObject_servicer.transOutMsg)
    hillPointing.celBodyInMsg.subscribeTo(ephemObject.ephemOutMsgs[earthIdx])
    scSim.AddModelToTask(simTaskName, hillPointing)

    # setup spacecraftPointing guidance module
    scPointing = locationPointing.locationPointing()
    scPointing.ModelTag = "scPointing"
    scPointing.pHat_B = [1, 0, 0]
    scPointing.useBoresightRateDamping = 1
    scPointing.scTargetInMsg.subscribeTo(sNavObject_chief.transOutMsg)
    scPointing.scTransInMsg.subscribeTo(sNavObject_servicer.transOutMsg)
    scPointing.scAttInMsg.subscribeTo(sNavObject_servicer.attOutMsg)
    scSim.AddModelToTask(simTaskName, scPointing)

    # setup sunPointing guidance module
    sunPointing = locationPointing.locationPointing()
    sunPointing.ModelTag = "scPointing"
    sunPointing.pHat_B = [0, 0, 1]
    sunPointing.useBoresightRateDamping = 1
    sunPointing.celBodyInMsg.subscribeTo(ephemObject.ephemOutMsgs[sunIdx])
    sunPointing.scTransInMsg.subscribeTo(sNavObject_servicer.transOutMsg)
    sunPointing.scAttInMsg.subscribeTo(sNavObject_servicer.attOutMsg)
    scSim.AddModelToTask(simTaskName, sunPointing)

    # setup the attitude tracking error evaluation module
    attError = attTrackingError.attTrackingError()
    attError.ModelTag = "attErrorInertial3D"
    scSim.AddModelToTask(simTaskName, attError)
    attError.attRefInMsg.subscribeTo(scPointing.attRefOutMsg)
    attError.attNavInMsg.subscribeTo(sNavObject_servicer.attOutMsg)

    # create the FSW vehicle configuration message
    # use the same inertia in the FSW algorithm as in the simulation
    vehicleConfigOut = messaging.VehicleConfigMsgPayload(ISCPntB_B=servicer_I)
    vcMsg = messaging.VehicleConfigMsg().write(vehicleConfigOut)

    # create FSW RW parameter msg
    fswRwMsg = rwFactory.getConfigMessage()

    # setup the MRP Feedback control module
    mrpControl = mrpFeedback.mrpFeedback()
    mrpControl.ModelTag = "mrpFeedback"
    scSim.AddModelToTask(simTaskName, mrpControl)
    mrpControl.guidInMsg.subscribeTo(attError.attGuidOutMsg)
    mrpControl.vehConfigInMsg.subscribeTo(vcMsg)
    mrpControl.rwParamsInMsg.subscribeTo(fswRwMsg)
    mrpControl.rwSpeedsInMsg.subscribeTo(rwStateEffector.rwSpeedOutMsg)
    # Dynamically calculate RW gains based on rigid-body Euler dynamics
    I_max = max(servicer_I)  
    
    # Tuned for aggressive fine-pointing lock (allows saturation during initial large slews)
    omega_n = 0.12  
    zeta = 1.9          
    
    mrpControl.K = 4.0 * I_max * (omega_n ** 2)  
    mrpControl.Ki = -1.0                         
    mrpControl.P = 2.0 * I_max * zeta * omega_n
    mrpControl.integralLimit = 2. / mrpControl.Ki * 0.1

    # add module that maps the Lr control torque into the RW motor torques
    rwMotorTorqueObj = rwMotorTorque.rwMotorTorque()
    rwMotorTorqueObj.ModelTag = "rwMotorTorque"
    scSim.AddModelToTask(simTaskName, rwMotorTorqueObj)
    # Initialize the test module msg names
    rwMotorTorqueObj.vehControlInMsg.subscribeTo(mrpControl.cmdTorqueOutMsg)
    rwMotorTorqueObj.rwParamsInMsg.subscribeTo(fswRwMsg)
    rwStateEffector.rwMotorCmdInMsg.subscribeTo(rwMotorTorqueObj.rwMotorTorqueOutMsg)
    # Make the RW control all three body axes
    controlAxes_B = [
        1, 0, 0, 0, 1, 0, 0, 0, 1
    ]
    rwMotorTorqueObj.controlAxes_B = controlAxes_B

    # Thrusters setup
    extFTObject = extForceTorque.ExtForceTorque()
    extFTObject.ModelTag = "ContinuousTranslationalThrust"
    scObject.addDynamicEffector(extFTObject)
    scSim.AddModelToTask(simTaskName, extFTObject, 100)

    # Create the continuous force command message (Body Frame)
    forceCmdPayload = messaging.CmdForceBodyMsgPayload()
    forceCmdPayload.forceRequestBody = [0.0, 0.0, 0.0]
    forceCmdMsg = messaging.CmdForceBodyMsg().write(forceCmdPayload)
    
    # Connect the command message to the effector
    extFTObject.cmdForceBodyInMsg.subscribeTo(forceCmdMsg)
    #-------------------------------------------------

    #
    #   Setup data logging before the simulation is initialized
    #

    scObjectTip = spacecraft.Spacecraft()
    scObjectTip.ModelTag = "tipMass"
    scObjectTip.hub.mHub = m_tip
    scObjectTip.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I_tip_flat)
    gravFactory.addBodiesTo(scObjectTip)        # gravity only -- no RW, no thrusters, no effectors

    scSim.AddModelToTask(simTaskName, scObjectTip, 2)

    tipProfiler = prescribedLinearTranslation.PrescribedLinearTranslation()
    tipProfiler.ModelTag = "tipMassProfiler"
    tipProfiler.setTransHat_M(np.array([0.0, 0.0, -1.0])) # Fire straight out
    tipProfiler.setTransAccelMax(2.0) #  acceleration to reach 1.0 m/s in 500 ms
    tipProfiler.setTransPosInit(0.0)
    tipProfiler.setCoastOptionBangDuration(0.5) #500 ms of accel
    tipProfiler.setSmoothingDuration(0.1)
    scSim.AddModelToTask(simTaskName, tipProfiler)

    tipTranslationData = messaging.LinearTranslationRigidBodyMsgPayload()
    tipTranslationData.rho = 0.0
    tipTranslationData.rhoDot = 0.0
    tipTranslationMsg = messaging.LinearTranslationRigidBodyMsg().write(tipTranslationData)

    tipProfiler.linearTranslationRigidBodyInMsg.subscribeTo(tipTranslationMsg)

    numDataPoints = 100
    samplingTime = unitTestSupport.samplingTime(simulationTime, simulationTimeStep, numDataPoints)
    rwCmdLog = rwMotorTorqueObj.rwMotorTorqueOutMsg.recorder(samplingTime)
    attErrLog = attError.attGuidOutMsg.recorder(samplingTime)
    sNavLog = sNavObject_servicer.transOutMsg.recorder(samplingTime)
    sNav2Log = sNavObject_chief.transOutMsg.recorder(samplingTime)
    thrCmdLog = forceCmdMsg.recorder(samplingTime)
    rwSpeedLog = rwStateEffector.rwSpeedOutMsg.recorder(samplingTime)
    sNavAttLog = sNavObject_servicer.attOutMsg.recorder(samplingTime)
    sNav2AttLog = sNavObject_chief.attOutMsg.recorder(samplingTime)
    scSim.AddModelToTask(simTaskName, sNavAttLog)
    scSim.AddModelToTask(simTaskName, sNav2AttLog)
    scSim.AddModelToTask(simTaskName, rwCmdLog)
    scSim.AddModelToTask(simTaskName, attErrLog)
    scSim.AddModelToTask(simTaskName, sNavLog)
    scSim.AddModelToTask(simTaskName, sNav2Log)                
    scSim.AddModelToTask(simTaskName, rwSpeedLog)
    scSim.AddModelToTask(simTaskName, thrCmdLog)
    #tip mass loggers                
    tipStateLog = scObjectTip.scStateOutMsg.recorder(samplingTime)
    scSim.AddModelToTask(simTaskName, tipStateLog)

    rwSc1Log = []
    for rw in rwStateEffector.rwOutMsgs:
        rwSc1Log.append(rw.recorder(samplingTime))
        scSim.AddModelToTask(simTaskName, rwSc1Log[-1])

    # Use overrides if provided, otherwise default to the baseline
    c_sig = cfg_overrides.get('chief_sigma_init', [0.3, 0.1, 0.2]) if cfg_overrides else [0.3, 0.1, 0.2]
    c_omg = cfg_overrides.get('chief_omega_init', [0.001, -0.01, 0.03]) if cfg_overrides else [0.001, -0.01, 0.03]
    #
    #   set initial Spacecraft States
    #
    # setup debris (chief) object states
    oe2 = orbitalMotion.ClassicElements()
    oe2.a = 10000000.0  # meters
    oe2.e = 0.0
    oe2.i = 33.3 * macros.D2R
    oe2.Omega = 48.2 * macros.D2R
    oe2.omega = 90.0 * macros.D2R
    oe2.f = 0.0 * macros.D2R
    r2N, v2N = orbitalMotion.elem2rv(mu, oe2)
    scObject2.hub.r_CN_NInit = r2N  # m   - r_CN_N
    scObject2.hub.v_CN_NInit = v2N  # m/s - v_CN_N
    scObject2.hub.sigma_BNInit = [[c_sig[0]], [c_sig[1]], [c_sig[2]]]
    scObject2.hub.omega_BN_BInit = [[c_omg[0]], [c_omg[1]], [c_omg[2]]]
    n = np.sqrt(mu/oe2.a/oe2.a/oe2.a)
    orbitPeriod = 2*np.pi / n   # in seconds

    # setup the servicer (deputy) orbit using Hill frame coordinates
    # oe = copy.deepcopy(oe2)
    # oe.e = 0.000001
    # rN, vN = orbitalMotion.elem2rv(mu, oe)
    rho_H = [-10.0, -40.0, 0.0]
    rho_Prime_H = [0, -1.5*n*rho_H[0], 0]
    rN, vN = orbitalMotion.hill2rv(r2N, v2N, rho_H, rho_Prime_H)
    scObject.hub.r_CN_NInit = rN  # m   - r_CN_N
    scObject.hub.v_CN_NInit = vN  # m/s - v_CN_N
    scObject.hub.sigma_BNInit = [[0.1],[0.2], [-0.3]]
    scObject.hub.omega_BN_BInit = [[0.0], [0.0], [0.0]]

    # if this scenario is to interface with the BSK Viz, uncomment the following lines
    # to save the BSK data to a file, uncomment the saveFile line below
    if show_plots and vizSupport.vizFound:
        servicerLight = vizInterface.Light()
        servicerLight.label = "Main Light"
        servicerLight.position = [1.0, 0.0, 0.00]
        servicerLight.fieldOfView = 10.0 * macros.D2R
        servicerLight.normalVector = [1, 0, 0]
        servicerLight.range = 150.0
        servicerLight.markerDiameter = 0.1
        servicerLight.color = vizInterface.IntVector(vizSupport.toRGBA255("white"))

        sc_body_list = [scObject, scObject2, scObjectTip]

        viz = vizSupport.enableUnityVisualization(scSim, simTaskName, sc_body_list
                                                  , rwEffectorList=[rwStateEffector, None, None] # chief and tip mass have no RWs
                                                  , lightList=[[servicerLight], None, None]
                                                  , saveFile=fileName
                                                  )

        # Draw the tip mass 
        vizSupport.createCustomModel(viz,
                                     simBodiesToModify=["tipMass"],
                                     modelPath="CYLINDER",
                                     scale=[r_tip*2, r_tip*2, h_tip],
                                     color=vizSupport.toRGBA255("red"))
        
        vizSupport.createPointLine(viz,
                               fromBodyName="tipMass",
                               toBodyName="servicer", 
                               lineColor=vizSupport.toRGBA255("yellow"))

        viz.settings.trueTrajectoryLinesOn = -1
        viz.settings.orbitLinesOn = 2
        viz.settings.mainCameraTarget = "debris"

    servicer_att_module = dockingGNC.dockingGNC()
    chief_att_module = dockingGNC.dockingGNC()

    #
    #   initialize Simulation
    #
    scSim.InitializeSimulation()

    debrisPos = scObject2.dynManager.getStateObject(scObject2.hub.nameOfHubPosition)
    debrisVel = scObject2.dynManager.getStateObject(scObject2.hub.nameOfHubVelocity)
    debrisSigma = scObject2.dynManager.getStateObject(scObject2.hub.nameOfHubSigma)
    debrisOmega = scObject2.dynManager.getStateObject(scObject2.hub.nameOfHubOmega)
    servicerPos = scObject.dynManager.getStateObject(scObject.hub.nameOfHubPosition)
    servicerVel = scObject.dynManager.getStateObject(scObject.hub.nameOfHubVelocity)
    
    tipPos   = scObjectTip.dynManager.getStateObject(scObjectTip.hub.nameOfHubPosition)
    tipVel   = scObjectTip.dynManager.getStateObject(scObjectTip.hub.nameOfHubVelocity)
    tipSigma = scObjectTip.dynManager.getStateObject(scObjectTip.hub.nameOfHubSigma)
    tipOmega = scObjectTip.dynManager.getStateObject(scObjectTip.hub.nameOfHubOmega)
    servicerSigma = scObject.dynManager.getStateObject(scObject.hub.nameOfHubSigma)

    servicerOmega = scObject.dynManager.getStateObject(scObject.hub.nameOfHubOmega)

    fswHillPointing()
    relOrbDrift()
    
    # CW dynamics matrices 
    n_mean = np.sqrt(mu / (oe2.a**3))
    Ac = np.array([[0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 1],
                   [3*n_mean**2, 0, 0, 0, 2*n_mean, 0],
                   [0, 0, 0, -2*n_mean, 0, 0],
                   [0, 0, -n_mean**2, 0, 0, 0]])
    Bc = np.vstack((np.zeros((3,3)), np.eye(3)))

    mat = np.zeros((9, 9))
    mat[:6, :6] = Ac
    mat[:6, 6:] = Bc
    exact_discrete = linalg.expm(mat * step_time)
    Ad = exact_discrete[:6, :6]
    Bd = exact_discrete[:6, 6:]
    max_accel = max_thrust / servicer_mass  

    u_max_rc = (0.99 * max_accel) / np.sqrt(3.0)
    K_sets = compute_rc_sets(Ad, Bd, N_horizon, u_max_rc, w_max_val)

    servicer_att_module.setEKFDynamics(Ad.flatten().tolist(), Bd.flatten().tolist())
    pos_only_sensing = cfg.get('pos_only_sensing', True)
    Cd = np.hstack((np.eye(3), np.zeros((3, 3)))) if pos_only_sensing else np.eye(6)

    # Initialize rolling buffers for the SVO
    buf_cols = max(1, svo_horizon)
    U_history = np.zeros((3, buf_cols)) # Allowing to send last measurments in case SVO horizon is 0
    Y_history = np.zeros((6, buf_cols))

    delta_disc_default = v_max_val + 0.5 * max_accel * step_time**2 
    delta_disc = float(cfg.get('delta_disc_override', delta_disc_default))
    
    r_effective = (L_chief + L_servicer) + delta_disc
    mpc_alpha_cbf = float(cfg.get('alpha_cbf_1', 0.20))
    h_safe_sq = 0.57                    # m^2 approx clearance
    p_obs = np.array(p_target, float)   # Chief COM KOZ

    # Setup the CasADi Solver
    opti = ca.Opti()
    P_R_sq = opti.parameter(1)          # Dynamic Barrier Radius
    
    X = opti.variable(6, N_horizon + 1)
    U = opti.variable(3, N_horizon)
    P_init = opti.parameter(6)  
    opti.subject_to(X[:, 0] == P_init)
    
    # Dynamic target and Dynamic Plane Normal
    P_port_chief = opti.parameter(3, N_horizon)    
    P_port_offset = opti.parameter(3, N_horizon)       
    P_vel_chief = opti.parameter(3, N_horizon) 
    P_station_weight = opti.parameter(1)  
    dynamic_max_vel = opti.parameter(1) 
    opti.set_value(dynamic_max_vel, v_max)
    cost = 0

    # Soft Constraint Slack Variable for the dynamic safety plane
    slack_vel = opti.variable(3, N_horizon)
    opti.subject_to(ca.vec(slack_vel) >= 0)

    xi_vars = [] #latent vector
    slack_match_vars = [] # For RC set tracking
    
    Q = np.diag([1000, 1000, 1000, 1000, 1000, 1000])
    R = np.diag([100, 100, 100])
    
    # Extract K_N once
    G_N = K_sets[N_horizon]['G']
    c_N = K_sets[N_horizon]['c']
    A_N = K_sets[N_horizon]['A']
    b_N = K_sets[N_horizon]['b']
    ng_N = G_N.shape[1]

    s_cbf = opti.variable(1, N_horizon)
    opti.subject_to(ca.vec(s_cbf) >= 0)

    # conditions and constraints for the solver- cost function
    for k in range(N_horizon):
        G_k = K_sets[k]['G']
        c_k = K_sets[k]['c']
        A_k = K_sets[k]['A']
        b_k = K_sets[k]['b']
        ng_k = G_k.shape[1]

        # Evaluate dynamic spherical CBF using the parameter P_R_sq
        h_cur = ca.sumsqr(X[0:3, k]   - p_obs) - P_R_sq
        h_nxt = ca.sumsqr(X[0:3, k+1] - p_obs) - P_R_sq
        
        # Penalized slack on both decay and absolute safety floor
        if cfg.get('mpc_cbf_constraint', True):
            opti.subject_to(h_nxt >= (1 - mpc_alpha_cbf*step_time)*h_cur - s_cbf[0, k])
            opti.subject_to(h_nxt >= h_safe_sq - s_cbf[0, k])
            cost += 1e5 * ca.sumsqr(s_cbf[0, k])
        else:
            opti.subject_to(s_cbf[0, k] == 0) # Dummy constraint to bind the variable
        
        # Dynamics
        opti.subject_to(X[:, k+1] == ca.mtimes(Ad, X[:, k]) + ca.mtimes(Bd, U[:, k]))
        
        # Errors
        port_pos = X[0:3, k] + P_port_offset[:, k]
        port_error = port_pos - P_port_chief[:, k]
        
        # Vel offset scalled with the current virtual target distance
        vel_offset = -P_vel_chief[:, k]
        servicer_port_vel = X[3:6, k] + vel_offset
        
        vel_error = servicer_port_vel - P_vel_chief[:, k]
        error = ca.vertcat(port_error, vel_error)
        
        cost += P_station_weight * ca.mtimes([error.T, Q, error]) + ca.mtimes([U[:, k].T, R, U[:, k]])

        # Reachable sets tracking
        xi = opti.variable(ng_k, 1)
        xi_vars.append(xi)
        
        opti.subject_to(opti.bounded(-1.0, xi, 1.0))
        if A_k.size > 0:
            opti.subject_to(ca.mtimes(A_k, xi) == b_k)
            
        slack_match = opti.variable(6, 1)
        slack_match_vars.append(slack_match)
        opti.subject_to(error == c_k + ca.mtimes(G_k, xi) + slack_match)
        cost += 1e4 * ca.sumsqr(slack_match)

        # Thruster limits (Spherical L2 norm guarantees rotation-proof physical feasibility)
        # 0.99 margin ensures IPOPT's acceptable_tol doesn't leak over the absolute clip boundary
        opti.subject_to(ca.sumsqr(U[:, k]) <= (0.99 * max_accel)**2)

        # Soft-bounded L_inf Velocity limit (15 cm/s)
        opti.subject_to(vel_error[0] <=  dynamic_max_vel + slack_vel[0, k])
        opti.subject_to(vel_error[0] >= -dynamic_max_vel - slack_vel[0, k])
        
        opti.subject_to(vel_error[1] <=  dynamic_max_vel + slack_vel[1, k])
        opti.subject_to(vel_error[1] >= -dynamic_max_vel - slack_vel[1, k])
        
        opti.subject_to(vel_error[2] <=  dynamic_max_vel + slack_vel[2, k])
        opti.subject_to(vel_error[2] >= -dynamic_max_vel - slack_vel[2, k])
        
        # Exact L1 Penalty (Acts like a hard bound, but survives SVO spikes)
        cost += 1e4 * ca.sumsqr(slack_vel[:, k])

    opti.minimize(cost) # minimize cost function
    p_opts = {"expand": True, "print_time": False}
    s_opts = {"max_iter": 400, "acceptable_tol": 1e-3, "print_level": 0}
    opti.solver("ipopt", p_opts, s_opts)

    # Run 1 step while fswHillPointing() is still subscribed so SimpleNav is populated
    slave_tip(*port_state_N(servicerPos, servicerVel, servicerSigma, servicerOmega, L_servicer))
    runSim(step_time)

    # Subscribing messages to populate every step with attitude errors
    mekfAttPayload = messaging.NavAttMsgPayload()
    mekfAttMsg = messaging.NavAttMsg().write(mekfAttPayload, simulationTime)
    attError.attNavInMsg.subscribeTo(mekfAttMsg)

    # Create and write to attRefMsg before subscribing attError to it
    attRefPayload = messaging.AttRefMsgPayload()
    attRefMsg = messaging.AttRefMsg().write(attRefPayload, simulationTime)
    
    attError.attRefInMsg.subscribeTo(attRefMsg)
    attError.sigma_R0R = [0, 0, 0]  # Clear any old offsets

    sNavMsg_warm = sNavObject_servicer.transOutMsg.read()
    sNav2Msg_warm = sNavObject_chief.transOutMsg.read()
    rd_noisy_warm = np.array(sNavMsg_warm.r_BN_N)
    vd_noisy_warm = np.array(sNavMsg_warm.v_BN_N)
    rc_noisy_warm = np.array(sNav2Msg_warm.r_BN_N)
    vc_noisy_warm = np.array(sNav2Msg_warm.v_BN_N)
    
    rho_H_warm, rho_Prime_H_warm = orbitalMotion.rv2hill(rc_noisy_warm, vc_noisy_warm, rd_noisy_warm, vd_noisy_warm)
    z_warm = np.concatenate((rho_H_warm, rho_Prime_H_warm))
    x_hat_init = z_warm.tolist()
    servicer_att_module.initSVO(Ad.flatten().tolist(), Bd.flatten().tolist(), Cd.flatten().tolist(), w_max_svo, v_max_val, svo_horizon, x_hat_init)

    #initial guess for the solver
    X_guess = np.zeros((6, N_horizon + 1))
    X_guess[:, 0] = z_warm
    U_guess = np.zeros((3, N_horizon))
    
    # Guess a basic braking trajectory instead of a collision-bound free-drift
    brake_guess = -1.0 * z_warm[3:6]
    if np.linalg.norm(brake_guess) > max_accel:
        brake_guess = brake_guess * (max_accel / np.linalg.norm(brake_guess))
        
    for k in range(N_horizon):
        U_guess[:, k] = brake_guess
        X_guess[:, k+1] = Ad.dot(X_guess[:, k]) + Bd.dot(brake_guess)
    
    # starting guess for the solver
    opti.set_initial(X, X_guess)
    opti.set_initial(U, U_guess)
    opti.set_value(P_station_weight, 1.0)

    # --- Time & Phase Histories (sampled every step_time) ---
    simulation_time_history = []
    phase_id_history = []          # 0:approach_hold, 1:turn, 2:holding, 3:approach_dock, 4:tether, 5:docked
    PHASE_MAP = {
        'approach_hold_point': 0, 'inspection_turn': 1, 'holding': 2,
        'approach_dock': 3, 'tether_flight': 4, 'docked': 5
    }

    # SVO & State Estimation Histories
    svo_est_history = []           # x_est (CCG center projection, 6D)
    svo_c_history = []             # c_X_k (unshifted CCG center, 6D)
    svo_xhat_history = []          # x_hat (Luenberger point estimate, 6D)
    true_state_history = []        # True Hill state [rho_H, rho_Prime_H] (6D)
    z_meas_history = []            # Noisy Hill measurement z_measurement (6D)
    svo_bounds_history = []        # (N, 6, 2) [lower, upper] shifted by (c_curr - true_state)
    svo_exact_in_set_history = []  # 1 = true_state inside 6D CCG, 0 = escaped, -1 = empty set
    svo_dims_history = []          # [ng_p, ng_e, A_rows] generator & constraint counts
    svo_time_ms_history = []       # Wall-clock ms per updateSVO() call

    #  MEKF Attitude & Rate Histories 
    mekf_servicer_history = []     # [sigma_est_clean (3), omega_est_clean (3)]
    mekf_chief_history = []        # [sigma_CN_est_clean (3), omega_CN_est_clean (3)]
    true_servicer_att_history = [] # [sigma_BN_true (3), omega_BN_B_true (3)]
    true_chief_att_history = []    # [sigma_CN_true (3), omega_CN_B_true (3)]
    att_err_history = []           # ||sigma_BR|| attitude tracking error norm

    # CBF Safety Histories 
    h_history = []                 # Current CBF value h_k
    h_pred_history = []            # Predicted CBF value h(k+1) under applied u_opt_H
    h_error_history = []           # Prediction error h_k - h_pred(k-1)
    h_error_time = []
    cbf_intervened_history = []    # 1 if safety filter modified u_opt_H, 0 otherwise
    cbf_converged_history = []     # 1 if KKT Newton solver converged, 0 otherwise
    cbf_iters_history = []         # Number of KKT Newton iterations used
    cbf_time_ms_history = []       # Wall-clock ms for CBF evaluation + filtering
    last_cbf_res_uopt = None

    #  MPC, Control, & Proximity Histories 
    cz_inclusion_history = []      # Tightest reachable set index (N_horizon - 11 to N_horizon)
    true_cz_inclusion_history = []
    t_scaling_est_history = []     # Set-scaling t for the SVO estimate
    t_scaling_true_history = []    # Set-scaling t for the True physical state
    slack_match_history = []       # ||slack_match_0||^2
    mpc_status_history = []        # 1 = IPOPT success, 0 = IPOPT failed (fallback), -1 = Tier 2 PD
    mpc_time_ms_history = []       # Wall-clock ms per opti.solve()
    T_req_history = []             # ||F_req_B|| (N)
    F_applied_history = []         # 3D applied force vector in Body frame (N)
    u_opt_H_history = []           # 3D applied acceleration command in Hill frame (m/s^2)
    rel_port_pos_history = []      # 3D Chief-port to Servicer-standoff-port vector (m)
    port_distance_history = []     # ||rel_port_pos|| (m)
    virtual_target_dist_history = [] # current_target_distance (m)
    pos_err_norm_history = []      # Tracking position error norm wrt active virtual target (m)
    vel_err_norm_history = []      # Tracking velocity error norm wrt active virtual target (m/s)

    #  Handoff & Capture Scalars 
    handoff_log = []
    capture_pos_error = np.nan
    capture_vel_error = np.nan
    #  Post-launch true state flight log (1 s sub-steps; the nav recorders are noisy and coarse) ----
    flight_log = []            # rows: [t_sim_s, tip r(3) v(3), chief r v sigma omegaB (12), servicer r v sigma omegaB (12)]
    flight_log_until = -1.0    # approach_time until which sub-stepped logging stays on
    launch_info = {}
    capture_t_sim_s = np.nan
    capture_success = np.nan

    # time stamps for tip mass plots
    t_docked_min = 0 
    t_release_min = 0
    phase_entry_time = 0.0  # approach_time [s] at which the current docking_phase began
    next_log_time = 0.0

    u_opt_H = np.zeros(3)
    u_nom_H = np.zeros(3)
    F_total_applied_B = np.zeros(3)
    failure_counter = 0
    approach_time = 0.0

    recoil_impulse_remaining = 0.0

    # Inspection turn & attitude lock parameters (Hill-frame X_H - Y_H circle)
    v_turn = 0.15 # m/s tangential speed
    num_turns = 2.0  # 2 full turns (4*pi rad)
    ATT_ERR_TOL = 0.02   # Attitude error threshold (~4.6 deg)
    RATE_ERR_TOL = 0.01  # Relaxed slightly to handle MEKF process noise
    theta_turn_start = -np.pi / 2.0 
    theta_turn = theta_turn_start
    R_turn = L_chief + L_servicer + hold_distance # Total COM radius from (0,0,0)
    omega_turn_nominal = v_turn / R_turn # Angular rate [rad/s]
    accelaration_steps = 1

    # The 3D unit vector tracking the virtual target ---
    u_virt_H = np.array([np.cos(theta_turn_start), np.sin(theta_turn_start), 0.0])
    dcm_dock_flip = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])

    prev_docking_phase = docking_phase  
    phase_transition_times = {}
    u_virt_history = [] 
    u_port_history = [] 
    true_pos_err_norm_history = []
    true_vel_err_norm_history = []

    active_t_flight = 0.0

    # Launch gate: predicted miss (estimate-vs-estimate, evaluated at the same instant as the aim)
    CAPTURE_CONE_DEG = float(cfg.get('capture_cone_deg', 10.0))   # max approach angle off the port axis
    capture_angle_deg = np.nan
    capture_closing_speed = np.nan
    LAUNCH_GATE_TOL = float(cfg.get('launch_gate_tol', 0.5 * CAPTURE_POS_TOL))
    launch_gate_log = []     # (t, miss_pred_est, tightest_set, att_err_norm) while in approach_dock
    miss_pred_est = np.inf

    def snap_flight():
        g = lambda s: unitTestSupport.EigenVector3d2np(s.getState()).flatten()
        return np.concatenate((
            [simulationTime * 1e-9],
            g(tipPos), g(tipVel),
            g(debrisPos), g(debrisVel), g(debrisSigma), g(debrisOmega),
            g(servicerPos), g(servicerVel), g(servicerSigma), g(servicerOmega)))

    approach_time_limit = int(cfg.get('loop_time', 2000.0))
    print("Starting approach")
    while approach_time < approach_time_limit:

        # Debug loggers-> true states directly from Basilisk
        rd_true = unitTestSupport.EigenVector3d2np(servicerPos.getState())
        vd_true = unitTestSupport.EigenVector3d2np(servicerVel.getState())
        rc_true = unitTestSupport.EigenVector3d2np(debrisPos.getState())
        vc_true = unitTestSupport.EigenVector3d2np(debrisVel.getState())
        
        rho_H_true, rho_Prime_H_true = orbitalMotion.rv2hill(rc_true, vc_true, rd_true, vd_true)
        true_state = np.concatenate((rho_H_true, rho_Prime_H_true))
        
        # Read noisy states
        sNavMsg = sNavObject_servicer.transOutMsg.read()
        sNav2Msg = sNavObject_chief.transOutMsg.read()
        rd_noisy = np.array(sNavMsg.r_BN_N)
        vd_noisy = np.array(sNavMsg.v_BN_N)
        rc_noisy = np.array(sNav2Msg.r_BN_N)
        vc_noisy = np.array(sNav2Msg.v_BN_N)

        # Servicer MEKF C++ Execution (fuses sigma_BN and omega_BN_B -> returns 6D [sigma_est, omega_est])
        sNavAttMsg = sNavObject_servicer.attOutMsg.read()
        servicer_mekf_in = list(sNavAttMsg.sigma_BN) + list(sNavAttMsg.omega_BN_B)
        servicer_mekf_out = np.array(servicer_att_module.updateMEKF(servicer_mekf_in, step_time))
        sigma_est_clean = servicer_mekf_out[0:3]
        omega_est_clean = servicer_mekf_out[3:6]

        # Feed MEKF-cleaned attitude AND rate into the FSW attitude tracking error module
        mekfAttPayload.timeTag = sNavAttMsg.timeTag
        mekfAttPayload.sigma_BN = sigma_est_clean.tolist()
        mekfAttPayload.omega_BN_B = omega_est_clean.tolist()
        mekfAttPayload.vehSunPntBdy = sNavAttMsg.vehSunPntBdy
        mekfAttMsg.write(mekfAttPayload, simulationTime)

        # Chief MEKF C++ Execution (fuses Chief sigma_BN and omega_BN_B -> returns 6D [sigma_CN_est, omega_CN_est])
        sNav2AttMsg = sNavObject_chief.attOutMsg.read()
        chief_mekf_in = list(sNav2AttMsg.sigma_BN) # + list(sNav2AttMsg.omega_BN_B)
        chief_mekf_out = np.array(chief_att_module.updateMEKF(chief_mekf_in, step_time))
        sigma_CN_est_clean = chief_mekf_out[0:3]
        omega_CN_est_clean = chief_mekf_out[3:6]
        dcm_CN = rbk.MRP2C(sigma_CN_est_clean)
        dcm_CN_flat = dcm_CN.flatten().tolist()
        omega_CN_B_flat = omega_CN_est_clean.flatten().tolist()

        # Build Hill-to-Inertial DCM (dcm_NH) from noisy nav readings
        rc_norm = np.linalg.norm(rc_noisy)
        h_vec = np.cross(rc_noisy, vc_noisy)
        r_hat = rc_noisy / rc_norm
        h_hat = h_vec / np.linalg.norm(h_vec)
        theta_hat = np.cross(h_hat, r_hat)
        dcm_NH = np.array([r_hat, theta_hat, h_hat]).T  # Inertial from Hill
        dcm_NH_flat = dcm_NH.flatten().tolist()

        # rv2hill: Express the deputy position and velocity vector as chief by the chief Hill frame
        rho_H_noisy, rho_Prime_H_noisy = orbitalMotion.rv2hill(rc_noisy, vc_noisy, rd_noisy, vd_noisy)
        z_measurement = np.concatenate((rho_H_noisy, rho_Prime_H_noisy))

        # Exp E: Extremal noise normalized in per-axis L_inf norm (so both modes have max |v_i| = ext_amp)
        noise_mode = cfg.get('noise_mode', 'none')
        ext_amp = float(cfg.get('ext_noise_amp', 0.0))
        r_unit = rho_H_true / max(np.linalg.norm(rho_H_true), 1e-6)
        r_inf_unit = r_unit / max(np.max(np.abs(r_unit)), 1e-6)  # L_inf unit vector: max(|r_i|) == 1.0
        if noise_mode == 'bias':
            # Outward radial bias with exact L_inf amplitude = ext_amp
            z_measurement[0:3] = rho_H_true + ext_amp * r_inf_unit
        elif noise_mode == 'alternating':
            # Sign-alternating extremal noise with exact L_inf amplitude = ext_amp
            sign_k = 1.0 if int(round(approach_time / step_time)) % 2 == 0 else -1.0
            z_measurement[0:3] = rho_H_true + sign_k * ext_amp * np.ones(3)
        elif noise_mode == 'normal':
            z_measurement[0:3] = rho_H_true 

        # -----------------------------------------------------------------
        # --- Timed SVO Update (Exp A, B, C, D) ---
        t0_svo = time.perf_counter()
        svo_tuple = servicer_att_module.updateSVO(
            U_history.flatten(order='C').tolist(),
            Y_history.flatten(order='C').tolist()
        )
        svo_time_ms_history.append((time.perf_counter() - t0_svo) * 1000.0)
        current_state = np.array(svo_tuple.center_flat)

        G_curr = np.array(svo_tuple.G_flat).reshape((svo_tuple.G_rows, svo_tuple.G_cols), order='C')
        c_curr = np.array(svo_tuple.c_flat)

        if svo_tuple.A_rows > 0:
            A_curr = np.array(svo_tuple.A_flat).reshape((svo_tuple.A_rows, svo_tuple.A_cols), order='C')
            b_curr = np.array(svo_tuple.b_flat)
        else:
            A_curr = np.array([[]])
            b_curr = np.array([])

        ng_p, ng_e = svo_tuple.ng_p, svo_tuple.ng_e

        svo_est_history.append(current_state)
        svo_c_history.append(c_curr)
        svo_xhat_history.append(np.array(svo_tuple.x_hat_flat))
        true_state_history.append(true_state)
        z_meas_history.append(z_measurement)
        svo_dims_history.append([ng_p, ng_e, svo_tuple.A_rows])
        
        if svo_horizon < 0:
            # Skip 13 Python LP solves per step during the Exp B recursive timing baseline
            svo_bounds_history.append(np.full((6, 2), np.nan))
            svo_exact_in_set_history.append(1)
            rho_k = v_max_val # Fallback baseline
        else:
            raw_bounds_k = svo_bounds_history_log(svo_tuple) # append SVO bounds per component
            if np.any(np.isnan(raw_bounds_k)):
                svo_exact_in_set_history.append(-1)
                rho_k = v_max_val
            else:
                in_ccg = state_inclusion(true_state, {'G': G_curr, 'c': c_curr, 'A': A_curr, 'b': b_curr})
                svo_exact_in_set_history.append(1 if in_ccg else 0)
                
                # 1. Calculate the bounding radius of the uncertainty set around its natural center (c_curr)
                dev_lo = np.abs(raw_bounds_k[:3, 0])
                dev_hi = np.abs(raw_bounds_k[:3, 1])
                rho_k_c = np.linalg.norm(np.maximum(dev_lo, dev_hi))
                
                # 2. Add the offset of the MPC's starting point (x_est) from the center (c)
                # This guarantees the sphere covers the true set even though we measure from x_est
                offset_c_to_xest = np.linalg.norm(current_state[:3] - c_curr[:3])
                rho_k = rho_k_c + offset_c_to_xest
        
        # Dynamically inflate the Keep-Out Zone by the exact size of the uncertainty set
        r_effective_k = r_effective + rho_k

        # Log MEKF vs. True Attitude & Rates
        mekf_servicer_history.append(servicer_mekf_out)
        mekf_chief_history.append(chief_mekf_out)
        true_servicer_att_history.append(np.concatenate((
            unitTestSupport.EigenVector3d2np(servicerSigma.getState()).flatten(),
            unitTestSupport.EigenVector3d2np(servicerOmega.getState()).flatten()
        )))
        true_chief_att_history.append(np.concatenate((
            unitTestSupport.EigenVector3d2np(debrisSigma.getState()).flatten(),
            unitTestSupport.EigenVector3d2np(debrisOmega.getState()).flatten()
        )))
        
        # Log Port-to-Standoff relative position and distance at step_time rate
        port_C_vec = np.array([0.0, 0.0, -L_chief])
        port_B_vec = np.array([0.0, 0.0, -L_servicer_standoff])
        port_N_chief_step = dcm_CN.T.dot(port_C_vec)
        port_N_servicer_step = rbk.MRP2C(sigma_est_clean).T.dot(port_B_vec)
        rel_port_pos_step = (rc_noisy + port_N_chief_step) - (rd_noisy + port_N_servicer_step)
        rel_port_pos_history.append(rel_port_pos_step)
        port_distance_history.append(np.linalg.norm(rel_port_pos_step))

        # -----------------------------------------------------------------

        # CBF current evaluation & prediction error tracking (Timed for Exp B & E)
        t0_cbf_curr = time.perf_counter()
        h_k = servicer_att_module.evaluateCBF_Current(p_target, r_effective_k)
        dt_cbf_curr_ms = (time.perf_counter() - t0_cbf_curr) * 1000.0
        h_k_finite = np.isfinite(h_k)
        h_history.append(h_k)
        if last_cbf_res_uopt is not None:
            h_error_history.append(h_k - last_cbf_res_uopt)
            h_error_time.append(approach_time / 60.0)

        # Immediate collision break for ablation testing
        if h_k < 0.0:
            print(f"\n[CRITICAL FATAL] Structural collision detected at t={approach_time/60.0:.2f} min! (h_k = {h_k:.3f})")
            print("Terminating simulation early to preserve crash state.")
            # Append final state histories to ensure array lengths match up to the crash point
            simulation_time_history.append(approach_time / 60.0)
            phase_id_history.append(PHASE_MAP.get(docking_phase, -1))
            virtual_target_dist_history.append(current_target_distance)
            pos_err_norm_history.append(0.0)
            vel_err_norm_history.append(0.0)
            true_pos_err_norm_history.append(0.0)
            true_vel_err_norm_history.append(0.0)
            att_err_history.append(att_err_norm)
            cz_inclusion_history.append(-1)
            true_cz_inclusion_history.append(-1)
            t_scaling_est_history.append(np.nan)
            t_scaling_true_history.append(np.nan)
            slack_match_history.append(0.0)
            # PAD THE MISSING CONTROL ARRAYS SO MIN_LEN DOES NOT TRUNCATE THE CRASH
            T_req_history.append(0.0)
            F_applied_history.append(np.zeros(3))
            u_opt_H_history.append(np.zeros(3))
            mpc_status_history.append(-1)
            mpc_time_ms_history.append(0.0)
            cbf_time_ms_history.append(0.0)
            cbf_intervened_history.append(0)
            cbf_converged_history.append(0)
            cbf_iters_history.append(0)
            break
            
        # Update virtual servicer length based on phase
        if docking_phase in ['approach_hold_point', 'inspection_turn', 'holding']:
            current_target_distance = hold_distance
        elif docking_phase == 'approach_dock':
            current_target_distance = max(standoff_distance, current_target_distance - approach_speed * step_time)
        else:
            current_target_distance = standoff_distance
        
        L_servicer_virtual = L_servicer + current_target_distance

        t_travel_nom = standoff_distance / v_launch  # Nominal 5.0 seconds
        port_C = np.array([0.0, 0.0, -L_chief])
        
        # Base tracking (MPC Translation Target) tracks the CURRENT port
        dcm_CN_active_flat = dcm_CN.flatten().tolist()
        port_H_chief = dcm_NH.T.dot(dcm_CN.T.dot(port_C))
        u_port_H = port_H_chief / max(np.linalg.norm(port_H_chief), 1e-6)
        
        #  Predictive Intercept Aiming (Attitude Target)
        if docking_phase in ['approach_dock', 'tether_flight']:
            # Current servicer port in Inertial frame (using MEKF est)
            dcm_BN_est = rbk.MRP2C(sigma_est_clean)
            port_B = np.array([0.0, 0.0, -L_servicer])
            r_port_serv_N = dcm_BN_est.T.dot(port_B)
            
            r_tip_current_N = rd_noisy + r_port_serv_N
            omega_BN_N_est = dcm_BN_est.T.dot(omega_est_clean)
            v_tip_current_N = vd_noisy + np.cross(omega_BN_N_est, r_port_serv_N)
            
            aim_mode = cfg.get('aim_mode', 'intercept')   # 'intercept' | 'oracle' | 'boresight'
            if aim_mode == 'oracle':
                # Ablation: same solver, but fed TRUE states (isolates navigation-knowledge error)
                r_t_or, v_t_or, _, _ = port_state_N(servicerPos, servicerVel, servicerSigma, servicerOmega, L_servicer)
                rc_or = unitTestSupport.EigenVector3d2np(debrisPos.getState()).flatten()
                vc_or = unitTestSupport.EigenVector3d2np(debrisVel.getState()).flatten()
                dcm_CN_or = rbk.MRP2C(unitTestSupport.EigenVector3d2np(debrisSigma.getState()).flatten())
                om_or = unitTestSupport.EigenVector3d2np(debrisOmega.getState()).flatten()
                u_dir, t_flight_planned, q_vec, w_vec = solve_intercept(
                    rc_or, vc_or, r_t_or, v_t_or, v_launch, port_C, dcm_CN_or, om_or, t_travel_nom)
            elif aim_mode == 'boresight':
                # Ablation: no lead, just point at the chief port as it is now
                q0 = rc_noisy + dcm_CN.T.dot(port_C) - r_tip_current_N
                u_dir = q0 / np.linalg.norm(q0)
                t_flight_planned = t_travel_nom
                q_vec = v_launch * t_flight_planned * u_dir   # so miss_pred_est == pure pointing error
                w_vec = np.zeros(3)
            else:
                u_dir, t_flight_planned, q_vec, w_vec = solve_intercept(
                    rc_noisy, vc_noisy, r_tip_current_N, v_tip_current_N,
                    v_launch, port_C, dcm_CN, omega_CN_est_clean.flatten(), t_travel_nom
                )
            
            z_B_req_N = -u_dir
            
            # Construct full target DCM (Align Z for intercept, match Chief X for roll)
            prv_rot = omega_CN_est_clean * t_flight_planned
            dcm_CN_future = rbk.PRV2C(prv_rot) @ dcm_CN
            dcm_dock_base = dcm_dock_flip @ dcm_CN_future
            x_target_N = dcm_dock_base[0, :]
            
            y_B_req_N = np.cross(z_B_req_N, x_target_N)
            y_B_req_N /= max(np.linalg.norm(y_B_req_N), 1e-6)
            x_B_req_N = np.cross(y_B_req_N, z_B_req_N)
            
            dcm_BN_dock = np.array([x_B_req_N, y_B_req_N, z_B_req_N])

            # Predicted miss: how far the *estimated* boresight is from the required one, times lever arm
            u_est_bore = -dcm_BN_est.T[:, 2]
            miss_pred_est = float(np.linalg.norm(q_vec + w_vec * t_flight_planned
                                                 - v_launch * t_flight_planned * u_est_bore))
        else:
            dcm_BN_dock = dcm_dock_flip @ dcm_CN
            t_flight_planned = t_travel_nom
            q_vec = np.zeros(3)
            w_vec = np.zeros(3)
            miss_pred_est = np.inf

        # Compute port-to-port error relative to the final standoff target using SVO & MEKF estimates
        port_C = np.array([0.0, 0.0, -L_chief])
        port_N_est = dcm_CN.T.dot(port_C)
        port_H_chief_est = dcm_NH.T.dot(port_N_est)
        port_offset_standoff = -port_H_chief_est * (L_servicer_standoff / L_chief)
        
        port_error_est = current_state[0:3] + port_offset_standoff - port_H_chief_est
        
        omega_N_est = dcm_CN.T.dot(omega_CN_est_clean).flatten()
        v_H_k_est = dcm_NH.T.dot(np.cross(omega_N_est, port_N_est))
        vel_error_est = current_state[3:6] - v_H_k_est * ((L_chief + L_servicer_standoff) / L_chief)
        current_error_state = np.concatenate((port_error_est, vel_error_est))
        
        # True Error relative to dockin position
        true_error_to_final = get_current_error()

        # Track which set we are in (Estimated State) across the entire horizon
        tightest_set = -1 
        for offset in range(N_horizon + 1):
            check_idx = N_horizon - offset
            if state_inclusion(current_error_state, K_sets[check_idx]): 
                tightest_set = check_idx
                break
                
        # Track which set we are in (TRUE State)  across the entire horizon
        true_tightest_set = -1
        for offset in range(N_horizon + 1):
            check_idx = N_horizon - offset
            if state_inclusion(true_error_to_final, K_sets[check_idx]):
                true_tightest_set = check_idx
                break
                
        # Calculate exactly how close we are to violating the ACTIVE estimate set (K_j*)
        # If -1 (outside K_0), evaluate against K_0 to see how far we are from entering the funnel
        active_idx = max(0, tightest_set)
        t_scaling_est = calculate_set_scaling(current_error_state, K_sets[active_idx])
        t_scaling_true = calculate_set_scaling(true_error_to_final, K_sets[active_idx])
                

        #  DOCKING TRIGGER: evaluated BEFORE runSim so aim, gate and launch share one instant ----
        if docking_phase == 'approach_dock':
            launch_gate_log.append((approach_time, miss_pred_est, tightest_set, att_err_norm))
            if int(approach_time) % 20 == 0:
                log_print(f"[LAUNCH GATE] miss_pred={miss_pred_est*100:.2f}cm (tol {LAUNCH_GATE_TOL*100:.1f}cm) "
                          f"set={tightest_set}/{N_horizon} dist={current_target_distance:.2f} att={att_err_norm:.4f}")

        if (docking_phase == 'approach_dock' and tightest_set == N_horizon
                and miss_pred_est < LAUNCH_GATE_TOL):
            print(f"Target reached at {approach_time/60:.1f} min")
            print(f"Predicted Miss at Launch: {miss_pred_est*100:.2f} cm, planned flight {t_flight_planned:.2f} s")


            # Free-flight launch along the true boresight
            dcm_BN_true = rbk.MRP2C(unitTestSupport.EigenVector3d2np(servicerSigma.getState()).flatten())
            u_true = -dcm_BN_true.T[:, 2]
            r_tip_N, v_tip_N_base, sig_tip, omg_tip = port_state_N(servicerPos, servicerVel, servicerSigma, servicerOmega, L_servicer)
            slave_tip(r_tip_N, v_tip_N_base + v_launch * u_true, sig_tip, omg_tip)

            recoil_impulse_remaining = m_tip * v_launch  # N.s, applied in this iteration's force command

            # Launch diagnostic: split the expected miss into knowledge error vs pointing error
            r_t_d, v_t_d, _, _ = port_state_N(servicerPos, servicerVel, servicerSigma, servicerOmega, L_servicer)
            u_or, T_or, q_or, w_or = solve_intercept(
                unitTestSupport.EigenVector3d2np(debrisPos.getState()).flatten(),
                unitTestSupport.EigenVector3d2np(debrisVel.getState()).flatten(),
                r_t_d, v_t_d, v_launch, port_C,
                rbk.MRP2C(unitTestSupport.EigenVector3d2np(debrisSigma.getState()).flatten()),
                unitTestSupport.EigenVector3d2np(debrisOmega.getState()).flatten(), t_travel_nom)
            miss_knowledge = v_launch * T_or * np.linalg.norm(u_dir - u_or)       # aim solved from noisy vs true data
            miss_pointing  = v_launch * T_or * np.linalg.norm(u_true - u_dir)     # true boresight vs aim command
            miss_true_pred = np.linalg.norm(q_or + w_or * T_or - v_launch * T_or * u_true)
            print(f"Launch diag: knowledge={miss_knowledge*100:.1f}cm pointing={miss_pointing*100:.1f}cm "
                  f"-> true-state predicted miss={miss_true_pred*100:.1f}cm (aim_mode={cfg.get('aim_mode','intercept')})")

            handoff_log.append({
                'phase': 'tether_launch', 't': approach_time,
                'true_state': get_true_relative_state(),
                'frozen_est': current_state.copy(),
            })

            docking_phase = 'tether_flight'
            active_t_flight = t_flight_planned
            t_release_min = simulationTime * macros.NANO2MIN
            phase_entry_time = approach_time

            # first flight-log sample = the launch instant (tip state already overridden above)
            flight_log.append(snap_flight())
            flight_log_until = approach_time + t_flight_planned + 4 * step_time
            launch_info = dict(
                u_aim=np.array(u_dir), u_true=np.array(u_true), u_oracle=np.array(u_or),
                t_flight_planned=float(t_flight_planned), t_sim_s=float(simulationTime * 1e-9),
                miss_knowledge=float(miss_knowledge), miss_pointing=float(miss_pointing),
                miss_true_pred=float(miss_true_pred), miss_pred_est=float(miss_pred_est))

        # Calculate the true 3D angle between where we are looking and the physical port
        dot_prod = np.clip(np.dot(u_virt_H, u_port_H), -1.0, 1.0)
        angle_error = np.arccos(dot_prod)

        if docking_phase in ['approach_hold_point', 'inspection_turn']:
            # Up-vector frame (turn phase only)
            z_B_H = u_virt_H
            up_vec = np.array([0.0, 0.0, 1.0]) if abs(z_B_H[2]) < 0.98 else np.array([1.0, 0.0, 0.0])
            x_B_H = np.cross(up_vec, z_B_H); x_B_H /= max(np.linalg.norm(x_B_H), 1e-6)
            y_B_H = np.cross(z_B_H, x_B_H)
            dcm_BN_upvec = np.array([x_B_H, y_B_H, z_B_H]) @ dcm_NH.T

            if docking_phase == 'inspection_turn':
                omega_turn = omega_turn_nominal * accelaration_steps/25
                if accelaration_steps < 25:
                    accelaration_steps += 1
                else: 
                    accelaration_steps = 25
                omega_turn_H = np.array([0.0, 0.0, omega_turn])
            else:
                omega_turn_H = np.zeros(3) 
                            
            dcm_BN_req = dcm_BN_upvec
            omega_RN_N = dcm_NH.dot(np.array([0.0, 0.0, n_mean]) + omega_turn_H)
            
            attRefPayload.sigma_RN = rbk.C2MRP(dcm_BN_req)
            attRefPayload.omega_RN_N = omega_RN_N
            attRefMsg.write(attRefPayload, simulationTime)

            # Update CasADi MPC Horizon Targets
            port_chief_future = np.zeros((3, N_horizon))
            port_offset_future = np.zeros((3, N_horizon))
            vel_chief_future = np.zeros((3, N_horizon))
            
            u_rad_k = u_virt_H.copy()
            for k_idx in range(N_horizon):
                port_chief_future[:, k_idx]  =  L_chief * u_rad_k
                port_offset_future[:, k_idx] = -L_servicer_virtual * u_rad_k
                vel_H = (L_chief + L_servicer_virtual) * np.cross(omega_turn_H, u_rad_k)
                vel_chief_future[:, k_idx]   =  0.5 * vel_H
                
                step_angle = np.linalg.norm(omega_turn_H) * step_time
                if step_angle > 1e-8:
                    axis = omega_turn_H / np.linalg.norm(omega_turn_H)
                    u_rad_k = u_rad_k * np.cos(step_angle) + np.cross(axis, u_rad_k) * np.sin(step_angle)
                    
        else: # holding, approach_dock, tether_flight, docked
            # Sync the logging vector so the Gaze Tracker plot shows the perfect lock
            u_virt_H = u_port_H.copy()

            dcm_BN_req = dcm_BN_dock
            attRefPayload.sigma_RN = rbk.C2MRP(dcm_BN_req)
            attRefPayload.omega_RN_N = dcm_CN.T.dot(omega_CN_est_clean)
            attRefMsg.write(attRefPayload, simulationTime)
            
            traj = servicer_att_module.propagateTargetKinematics(\
                dcm_CN_active_flat, omega_CN_B_flat, dcm_NH_flat,  
                int(N_horizon), step_time, L_chief, L_servicer_virtual
            )
            port_chief_future = np.array(list(traj.port_chief_flat)).reshape((N_horizon, 3)).T
            port_offset_future = np.array(list(traj.port_offset_flat)).reshape((N_horizon, 3)).T
            vel_chief_future = np.array(list(traj.vel_chief_flat)).reshape((N_horizon, 3)).T

        opti.set_value(P_port_chief, port_chief_future)
        opti.set_value(P_port_offset, port_offset_future)
        opti.set_value(P_vel_chief, vel_chief_future)
        opti.set_value(P_init, current_state)
        opti.set_value(P_R_sq, r_effective_k**2)

        # Evaluate general attitude tracking error relative to the active reference
        dcm_BR = rbk.MRP2C(sigma_est_clean) @ dcm_BN_req.T
        att_err_norm = np.linalg.norm(rbk.C2MRP(dcm_BR))
        
        # Compute command rate in the Body frame using the actual estimated DCM ---
        dcm_BN_est = rbk.MRP2C(sigma_est_clean)
        omega_RN_B_cmd = dcm_BN_est @ np.array(attRefPayload.omega_RN_N, dtype=float).flatten()
        rate_err_norm = np.linalg.norm(omega_est_clean - omega_RN_B_cmd) 
        current_slack_val = 0.0 # in case the solver fails
        
        # Two-Tier Controller Architecture
        if docking_phase in ['approach_hold_point', 'inspection_turn', 'holding', 'approach_dock']: # TIER 1: CasADi MPC Loop
            
            filter_alpha_cbf = float(cfg.get('alpha_cbf_1', 0.20))
            safety_margin = 0.1
            
            t0_mpc = time.perf_counter()
            try: 
                # Optimal unconstrained input
                sol = opti.solve()
                mpc_time_ms_history.append((time.perf_counter() - t0_mpc) * 1000.0)
                mpc_status_history.append(1)

                X_guess = np.array(sol.value(X))
                opti.set_initial(X, X_guess)
                opti.set_initial(U, sol.value(U))

                u_opt_H_raw = np.array(sol.value(U[:, 0])).flatten()
                # Compute L_inf norm of the set slack variable (0 means perfectly inside)
                current_slack_val = np.max(np.abs(sol.value(slack_match_vars[0])))
                                
                # Evaluate, calculate decay, and project entirely in C++
                t0_cbf = time.perf_counter()
                if cfg.get('cpp_safety_filters', True):
                    cbf_res_uopt = servicer_att_module.evaluateCBF_Uopt(
                        u_opt_H_raw.tolist(), p_target, r_effective_k,
                        h_k, filter_alpha_cbf, step_time, safety_margin, max_accel
                    )
                    u_opt_H = np.array(cbf_res_uopt.u_safe_flat)
                    cbf_converged_history.append(1 if cbf_res_uopt.converged else 0)
                    cbf_iters_history.append(cbf_res_uopt.iterations)
                    if not cbf_res_uopt.converged:
                        log_print(f"CBF infeasible within thrust ball (h_pred={cbf_res_uopt.h:.4f}); applying best-effort push.")
                else:
                    u_opt_H = u_opt_H_raw
                    cbf_converged_history.append(1)
                    cbf_iters_history.append(0)
                    
                cbf_time_ms_history.append(dt_cbf_curr_ms + (time.perf_counter() - t0_cbf) * 1000.0)
                
                intervened = 1 if np.linalg.norm(u_opt_H - u_opt_H_raw) > 1e-6 else 0
                cbf_intervened_history.append(intervened)
                if intervened:
                    log_print(f"Input safety applied: modified to respect decay boundary.")

            except Exception as e:
                mpc_time_ms_history.append((time.perf_counter() - t0_mpc) * 1000.0)
                mpc_status_history.append(0)
                cbf_intervened_history.append(0)
                cbf_converged_history.append(0)
                cbf_iters_history.append(0)
                cbf_time_ms_history.append(dt_cbf_curr_ms)

                log_print(f"Nominal MPC Failed ({e}). Executing fallback braking.")
                log_print(f"[dbg] t={approach_time:.0f}s rho_k={rho_k:.2f} R_k={r_effective_k:.2f} "
                f"target_radius={np.linalg.norm(port_chief_future[:,0]-port_offset_future[:,0]):.2f} "
                f"iters={opti.stats().get('iter_count')} "
                f"max_s_cbf={np.max(opti.debug.value(s_cbf)):.3g}")
                u_opt_H = -1.0 * current_state[3:6]
                norm_fallback = np.linalg.norm(u_opt_H)
                if norm_fallback > max_accel:
                    u_opt_H = u_opt_H * (max_accel / norm_fallback)
                
                # Wipe casadi memory and set guess to the braking trajectory
                X_reset = np.zeros((6, N_horizon + 1))
                X_reset[:, 0] = current_state
                U_reset = np.zeros((3, N_horizon))
                for k_r in range(N_horizon):
                    U_reset[:, k_r] = u_opt_H
                    X_reset[:, k_r + 1] = Ad.dot(X_reset[:, k_r]) + Bd.dot(u_opt_H)
                opti.set_initial(X, X_reset)
                opti.set_initial(U, U_reset)
                failure_counter += 1
                if failure_counter >= 30:
                    print(f"Sequential solver failures -> breaking")
                    break
        

            u_opt_N = dcm_NH.dot(u_opt_H)
            u_opt_B = rbk.MRP2C(sigma_est_clean).dot(u_opt_N)
            
            F_total_applied_B = np.clip(servicer_mass * u_opt_B, -max_thrust, max_thrust)
            T_req_history.append(np.linalg.norm(F_total_applied_B))
            F_applied_history.append(F_total_applied_B.copy())
            u_opt_H_history.append(u_opt_H.copy())
            
            forceCmdPayload.forceRequestBody = F_total_applied_B.tolist()
            forceCmdMsg.write(forceCmdPayload, simulationTime)
            
            slave_tip(*port_state_N(servicerPos, servicerVel, servicerSigma, servicerOmega, L_servicer))


        else: # TIER 2: (Docked / Tether Flight)
            mpc_time_ms_history.append(0.0)
            mpc_status_history.append(-1)

            # Ghost Target COM
            target_pos = port_chief_future[:, 0] - port_offset_future[:, 0]
            
            # Velocity constrained to the current L_servicer_virtual radius, vel at k
            vel_offset_0 = vel_chief_future[:, 0]
            target_vel = vel_chief_future[:, 0] + vel_offset_0
            
            # Base acceleration from known rotation, vel at k+1
            vel_offset_1 = vel_chief_future[:, 1]
            target_vel_next = vel_chief_future[:, 1] + vel_offset_1
            target_accel = (target_vel_next - target_vel) / step_time

            
            #  Errors from SVO states
            pos_err = current_state[0:3] - target_pos
            vel_err = current_state[3:6] - target_vel
            
            # Feedforward acell + PD
            Kp = 0.07
            Kd = 0.3
            u_nom_H = target_accel - Kp * pos_err - Kd * vel_err
            
            # Calculate recoil equivalent acceleration to reserve headroom
            a_recoil = 0.0
            if recoil_impulse_remaining > 0:
                force_mag = min(recoil_impulse_remaining / step_time, max_thrust)
                a_recoil = force_mag / servicer_mass
            available_accel = max_accel - a_recoil

            # Evaluate CBF at the current state to get h and the spatial gradient
            t0_cbf = time.perf_counter()
            cbf_res = servicer_att_module.evaluateCBF_Gradient(p_target, r_effective_k)
            alpha_fast = float(cfg.get('alpha_fast', 0.30))

            norm_u_nom = np.linalg.norm(u_nom_H)
            u_nom_capped = u_nom_H if norm_u_nom <= available_accel else u_nom_H * (available_accel / norm_u_nom)

            if cfg.get('cpp_safety_filters', True):
                # QP filter (passes available_accel * mass as the headroom limit)
                u_opt_H_QP = servicer_att_module.filterQPCBF_3D(
                    u_nom_H.tolist(), list(cbf_res.grad_u_flat), h_k, alpha_fast, step_time,
                    available_accel * servicer_mass, servicer_mass, pos_err.tolist(), current_state[3:6].tolist(),
                    rc_noisy.tolist(), vc_noisy.tolist(), mu
                )
                u_opt_H = np.array(u_opt_H_QP[:3])
                qp_is_feasible = int(u_opt_H_QP[3])
                cbf_iters_history.append(cbf_res.iterations)
            else:
                u_opt_H = u_nom_capped
                qp_is_feasible = 1
                cbf_iters_history.append(0)
                
            cbf_time_ms_history.append(dt_cbf_curr_ms + (time.perf_counter() - t0_cbf) * 1000.0)
            cbf_intervened_history.append(1 if np.linalg.norm(u_opt_H - u_nom_capped) > 1e-6 else 0)
            cbf_converged_history.append(qp_is_feasible)
            current_slack_val = 0.0
            
            # Rotate Hill accel -> Inertial -> Servicer Body using MEKF sigma_est_clean and existing dcm_NH
            dcm_BN_est = rbk.MRP2C(sigma_est_clean)
            # Use the sliced 3-element u_opt_H array, not the raw 4-element u_opt_H_QP array
            u_opt_B_f = dcm_BN_est.dot(dcm_NH.dot(u_opt_H))
            F_req_B = servicer_mass * u_opt_B_f
            
            F_recoil_physical_B = np.zeros(3)

            # Recoil calculation
            if recoil_impulse_remaining > 0:
                force_mag = min(recoil_impulse_remaining / step_time, max_thrust)
                F_req_B += np.array([0.0, 0.0, -force_mag])
                F_recoil_physical_B = np.array([0.0, 0.0, force_mag])
                recoil_impulse_remaining -= force_mag * step_time

            F_req_B = np.clip(F_req_B, -max_thrust, max_thrust)

            F_total_applied_B = F_req_B + F_recoil_physical_B
            forceCmdPayload.forceRequestBody = F_total_applied_B.tolist()
            forceCmdMsg.write(forceCmdPayload, simulationTime)
            
            # Move the tip mass
            if docking_phase == 'approach_dock':
                slave_tip(*port_state_N(servicerPos, servicerVel, servicerSigma, servicerOmega, L_servicer)) 
            elif docking_phase == 'tether_flight':
                # Let Basilisk dynamics propagate the tip mass ballistically in open loop.
                r_tip_N = unitTestSupport.EigenVector3d2np(tipPos.getState()).flatten()
                v_tip_N = unitTestSupport.EigenVector3d2np(tipVel.getState()).flatten()
                r_cport_N, v_cport_N, _, _ = port_state_N(debrisPos, debrisVel, debrisSigma, debrisOmega, L_chief)
                
                dr = r_tip_N - r_cport_N
                dv = v_tip_N - v_cport_N
                t_star = -(dr @ dv) / max(dv @ dv, 1e-9)
                miss = float(np.linalg.norm(dr + dv * t_star))
                sig_c_true = unitTestSupport.EigenVector3d2np(debrisSigma.getState()).flatten()
                n_out_N = rbk.MRP2C(sig_c_true).T.dot(np.array([0.0, 0.0, -1.0]))   # chief port outward normal
                dv_nominal = -v_launch * n_out_N                                    # nominal head-on arrival
                closing_speed = float(-dv @ n_out_N)
                lateral_speed = float(np.linalg.norm(dv - (dv @ n_out_N) * n_out_N))
                approach_angle_deg = float(np.degrees(np.arctan2(lateral_speed, max(closing_speed, 1e-9))))
                vel_miss = lateral_speed
                
                # Trigger capture validation when passed closest approach or timeout
                if t_star <= 0 or (approach_time - phase_entry_time) > (active_t_flight + step_time):
                    capture_pos_error = miss
                    capture_vel_error = vel_miss
                    t_docked_min = simulationTime * macros.NANO2MIN
                    
                    capture_angle_deg = approach_angle_deg
                    capture_closing_speed = closing_speed
                    capture_t_sim_s = simulationTime * 1e-9
                    if miss < CAPTURE_POS_TOL and approach_angle_deg < CAPTURE_CONE_DEG and closing_speed > 0.0:
                        print(f"  closing={closing_speed*100:.1f}cm/s lateral={lateral_speed*100:.1f}cm/s approach angle={approach_angle_deg:.1f}deg (cone {CAPTURE_CONE_DEG:.1f}deg)")
                        print(f"Capture SUCCESS at {t_docked_min:.2f} min! Miss: {miss*100:.1f}cm, Vel: {vel_miss*100:.1f}cm/s")
                        docking_phase = 'docked'
                        capture_success = 1.0
                    else:
                        print(f"  closing={closing_speed*100:.1f}cm/s lateral={lateral_speed*100:.1f}cm/s approach angle={approach_angle_deg:.1f}deg (cone {CAPTURE_CONE_DEG:.1f}deg)")
                        print(f"Capture FAILED at {t_docked_min:.2f} min. Miss: {miss*100:.1f}cm, Vel: {vel_miss*100:.1f}cm/s")
                        docking_phase = 'failed'
                        capture_success = 0.0

            elif docking_phase == 'docked':
                # Post-capture constraint: tip is physically locked to the chief
                slave_tip(*port_state_N(debrisPos, debrisVel, debrisSigma, debrisOmega, L_chief))
                
            elif docking_phase == 'failed':
                # If missed, just let the tip mass float away ballistically (do nothing)
                pass
                
            T_req_history.append(np.linalg.norm(F_req_B))
            F_applied_history.append(F_total_applied_B.copy())
            u_opt_H_history.append(u_opt_H.copy())

        # SVO reconstruction using MEKF sigma_est_clean and existing dcm_NH
        actual_accel_B = F_total_applied_B / servicer_mass
        actual_accel_N = dcm_BN_est.T.dot(actual_accel_B)
        actual_accel_H = dcm_NH.T.dot(actual_accel_N)

        #  Observer buffers update
        Y_history[:, :-1] = Y_history[:, 1:]
        Y_history[:, -1] = z_measurement
        U_history[:, :-1] = U_history[:, 1:]
        U_history[:, -1] = actual_accel_H

        # # CBF prediction error tracking
        pred_alpha_cbf = 0.4 # For margin
        last_cbf_res_uopt = servicer_att_module.evaluateCBF_Uopt(
            actual_accel_H.tolist(), p_target, r_effective_k,
            h_k, pred_alpha_cbf, step_time, safety_margin, max_accel
        ).h
        h_pred_history.append(last_cbf_res_uopt)
        u_virt_history.append(u_virt_H.copy()) 
        u_port_history.append(u_port_H.copy()) 

        if flight_log_until > 0 and approach_time <= flight_log_until:
            n_sub = max(1, int(round(step_time)))
            for _ in range(n_sub):
                runSim(step_time / n_sub)
                flight_log.append(snap_flight())
        else:
            runSim(step_time)
        approach_time += step_time
        
        if docking_phase in ['approach_dock', 'docked']:
            opti.set_value(P_station_weight, 50.0)
        else:
            opti.set_value(P_station_weight, 1.0)

        # Compute tracking error relative to the currently active virtual target (using SVO state)
        pos_err_vec = current_state[0:3] + port_offset_future[:, 0] - port_chief_future[:, 0]
        vel_err_vec = current_state[3:6] - 2.0 * vel_chief_future[:, 0]
        pos_err_norm = np.linalg.norm(pos_err_vec)
        vel_err_norm = np.linalg.norm(vel_err_vec)

        # Compute the TRUE physical tracking error relative to the active virtual target
        true_pos_err_vec = true_state[0:3] + port_offset_future[:, 0] - port_chief_future[:, 0]
        true_vel_err_vec = true_state[3:6] - 2.0 * vel_chief_future[:, 0]
        true_pos_err_norm = np.linalg.norm(true_pos_err_vec)
        true_vel_err_norm = np.linalg.norm(true_vel_err_vec)

        # Evaluate general attitude tracking error relative to the active reference
        dcm_BR = rbk.MRP2C(sigma_est_clean) @ dcm_BN_req.T
        att_err_norm = np.linalg.norm(rbk.C2MRP(dcm_BR))
        
        # Fix: Compute command rate in the Body frame using the actual estimated DCM
        dcm_BN_est = rbk.MRP2C(sigma_est_clean)
        omega_RN_B_cmd = dcm_BN_est @ np.array(attRefPayload.omega_RN_N, dtype=float).flatten()
        rate_err_norm = np.linalg.norm(omega_est_clean - omega_RN_B_cmd)

        ok_pos = pos_err_norm < 2*TERMINAL_BOX_POS
        ok_vel = vel_err_norm < 2*TERMINAL_BOX_VEL
        ok_att = att_err_norm < ATT_ERR_TOL
        ok_rate = rate_err_norm < RATE_ERR_TOL

        transition_ready = True if (ok_pos and ok_vel and ok_att and ok_rate) else False
        
        # Reach hold point & check for attitude error
        if docking_phase == 'approach_hold_point':
            if transition_ready:
                log_print(f"Reached 8m hold point (Pos Error: {pos_err_norm:.2f}m) and attitude locked. "
                          f"Starting {num_turns:.0f}-lap inspection turn.")
                docking_phase = 'inspection_turn'
                
        # Advance inspection circle angle
        elif docking_phase == 'inspection_turn':            
            if (theta_turn - theta_turn_start) >= num_turns * 2.0 * np.pi and transition_ready:
                log_print(f"Completed {num_turns:.0f} turns. Transitioning to -Z_C hold point for stabilization.")
                docking_phase = 'holding'
            else:
                theta_turn += omega_turn_H[2] * step_time   
                
            step_angle = np.linalg.norm(omega_turn_H) * step_time
            if step_angle > 1e-8:
                axis = omega_turn_H / np.linalg.norm(omega_turn_H)
                u_virt_H = u_virt_H * np.cos(step_angle) + np.cross(axis, u_virt_H) * np.sin(step_angle)
                u_virt_H /= max(np.linalg.norm(u_virt_H), 1e-6) 
                    
        elif docking_phase == 'holding':            
            if transition_ready: 
                docking_phase = 'approach_dock'
        else:
            opti.set_value(dynamic_max_vel, TERMINAL_BOX_VEL)          
                
        # Append per-step histories for post-processing
        phase_id_history.append(PHASE_MAP.get(docking_phase, -1))
        virtual_target_dist_history.append(current_target_distance)
        pos_err_norm_history.append(pos_err_norm)
        vel_err_norm_history.append(vel_err_norm)
        true_pos_err_norm_history.append(true_pos_err_norm)  # <-- NEW
        true_vel_err_norm_history.append(true_vel_err_norm)  # <-- NEW
        att_err_history.append(att_err_norm)
        cz_inclusion_history.append(tightest_set)
        true_cz_inclusion_history.append(true_tightest_set)
        t_scaling_est_history.append(t_scaling_est)
        t_scaling_true_history.append(t_scaling_true)
        simulation_time_history.append(approach_time / 60.0)
        slack_match_history.append(current_slack_val)

        # Record phase transition
        if docking_phase != prev_docking_phase:
            phase_transition_times[docking_phase] = approach_time / 60.0
            prev_docking_phase = docking_phase
        # ----------------------
        

        if approach_time >= next_log_time:
            log_print("Approach Time", approach_time)
            next_log_time += 100.0

    if launch_gate_log and docking_phase == 'approach_dock':
        lg = np.array(launch_gate_log)
        print(f"[LAUNCH GATE] never fired. min miss_pred={np.min(lg[:,1])*100:.2f}cm, "
              f"steps with set==N: {int(np.sum(lg[:,2]==N_horizon))}/{len(lg)}")
    print("Final drift sim")
    runSim(10.0)

    #
    #   retrieve the logged datq
    # 

    dataUsReq = rwCmdLog.motorTorque[:, range(numRW)]

    posServicer = sNavLog.r_BN_N 
    posChief = sNav2Log.r_BN_N
    velChief = sNav2Log.v_BN_N

    sigmaServicer = sNavAttLog.sigma_BN
    sigmaChief = sNav2AttLog.sigma_BN
    
    # The ports are on the -Z axis of both spacecraft's body frames
    port_B = np.array([0.0, 0.0, -L_servicer_standoff])
    relPortPos = np.zeros_like(posServicer)
    
    # Calculate the inertial position of the ports at every time step
    for i in range(len(posServicer)): 
        # Chief Port in Inertial Frame
        dcm_CN = rbk.MRP2C(sigmaChief[i])
        port_N_chief = dcm_CN.T.dot(port_C)
        portPosChief = posChief[i] + port_N_chief
        
        # Servicer Standoff Port in Inertial Frame
        dcm_BN = rbk.MRP2C(sigmaServicer[i])
        port_N_servicer = dcm_BN.T.dot(port_B)
        portPosServicer = posServicer[i] + port_N_servicer
        
        # Distance between the two targets
        relPortPos[i] = portPosChief - portPosServicer

    # Calculate total standoff-to-port distance magnitude
    dataPortDistance = np.linalg.norm(relPortPos, axis=1)

    # Extract the continuous 3D force vector directly
    dataThr = thrCmdLog.forceRequestBody
    # The force is already in Newtons, so we just grab the components directly
    force_x = dataThr[:, 0]
    force_y = dataThr[:, 1]
    force_z = dataThr[:, 2]
    
    # Calculate the magnitude for plotting
    dataThrustMag = np.sqrt(force_x**2 + force_y**2 + force_z**2)

    dataSigmaBR = attErrLog.sigma_BR
    dataOmegaBR = attErrLog.omega_BR_B
    dataOmegaRW = rwSpeedLog.wheelSpeeds[:, range(numRW)]
    dataRW = []
    for i in range(numRW):
        dataRW.append(rwSc1Log[i].u_current)
    np.set_printoptions(precision=16)

    #
    #   plot the results
    #
    timeData = attErrLog.times() * macros.NANO2MIN
    plt.close("all")  # clears out plots from earlier test runs

    plot_attitude_error(timeData, dataSigmaBR)
    figureList = {}
    pltName = fileName + "1"
    figureList[pltName] = plt.figure(1)

    plot_rw_motor_torque(timeData, dataUsReq, dataRW, numRW)
    pltName = fileName + "2"
    figureList[pltName] = plt.figure(2)

    plot_rate_error(timeData, dataOmegaBR)
    plot_rw_speeds(timeData, dataOmegaRW, numRW)
    pltName = fileName + "3"
    figureList[pltName] = plt.figure(4)

    plot_distance_and_thrust(timeData, dataPortDistance, dataThrustMag, relPortPos)
    pltName = fileName + "distance_components"
    figureList[pltName] = plt.figure(9)

    # Truncate ALL lists so no array size mismatches cause crashes
    min_len = min( len(svo_c_history), len(svo_xhat_history), len(z_meas_history),
        len(simulation_time_history), len(svo_est_history), len(true_state_history),
        len(T_req_history), len(h_history), len(svo_bounds_history),
        len(port_distance_history), len(rel_port_pos_history))    

    simulation_time_history = simulation_time_history[:min_len]
    cz_inclusion_history = cz_inclusion_history[:min_len]
    slack_match_history = slack_match_history[:min_len]
    svo_est_history = svo_est_history[:min_len]
    true_state_history = true_state_history[:min_len]
    T_req_history = T_req_history[:min_len]
    h_history = h_history[:min_len]
    svo_bounds_history = svo_bounds_history[:min_len]

    # Calculate the position error norm
    svo_est_arr = np.array(svo_est_history)
    true_state_arr = np.array(true_state_history)
    pos_error_norm = np.linalg.norm(svo_est_arr[:, 0:3] - true_state_arr[:, 0:3], axis=1)

    # Call the plotting function ONCE with all 6 data arrays
    plot_cz_inclusion(simulation_time_history, cz_inclusion_history, slack_match_history, pos_error_norm, T_req_history, h_history, N_horizon, svo_horizon, max_thrust, w_max_svo, v_max_val)
    pltName = fileName + "cz_inclusion"
    figureList[pltName] = plt.figure(6)


    # Retrieve tip mass state
    posTip    = tipStateLog.r_BN_N
    velTip    = tipStateLog.v_BN_N
    sigmaTip  = tipStateLog.sigma_BN
    omegaTip  = tipStateLog.omega_BN_B

    # Pull chief attitude from sNav2AttLog
    sigmaChief_log = sNav2AttLog.sigma_BN      # shape (N, 3)
    omegaChief_log = sNav2AttLog.omega_BN_B    # shape (N, 3)

    # Calculate true relative angular rate and attitude in a common frame
    omega_rel_norm = np.zeros(len(omegaTip))
    sigma_rel = np.zeros_like(sigmaTip)
    sigma_rel_norm = np.zeros(len(sigmaTip))
    
    # The required docking attitude includes a 180-degree flip
    dcm_dock_flip_plot = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])

    for i in range(len(sigmaTip)):
        C_tip = rbk.MRP2C(sigmaTip[i])
        C_chief = rbk.MRP2C(sigmaChief_log[i])
        
        # Map body rates to Inertial frame for valid subtraction
        omega_tip_N = C_tip.T.dot(omegaTip[i])
        omega_chief_N = C_chief.T.dot(omegaChief_log[i])
        omega_rel_norm[i] = np.linalg.norm(omega_tip_N - omega_chief_N)
        
        # Relative attitude wrt desired docking orientation
        C_chief_dock = dcm_dock_flip_plot @ C_chief
        C_rel = C_tip @ C_chief_dock.T
        sigma_rel[i] = rbk.C2MRP(C_rel)
        sigma_rel_norm[i] = np.linalg.norm(sigma_rel[i])

    # Compute tip-to-servicer-port and tip-to-chief-port distances
    dist_tip_to_servicer = np.zeros(len(posTip))
    dist_tip_to_chief    = np.zeros(len(posTip))

    for i in range(len(posTip)):
        dcm_BN = rbk.MRP2C(sigmaServicer[i])
        port_servicer_N = posServicer[i] + dcm_BN.T.dot(np.array([0., 0., -L_servicer]))
        
        dcm_CN = rbk.MRP2C(sigmaChief[i])
        port_chief_N = posChief[i] + dcm_CN.T.dot(np.array([0., 0., -L_chief]))
        
        dist_tip_to_servicer[i] = np.linalg.norm(posTip[i] - port_servicer_N)
        dist_tip_to_chief[i]    = np.linalg.norm(posTip[i] - port_chief_N)


    plot_estimation_error(simulation_time_history, svo_est_history, true_state_history, svo_bounds_history, svo_horizon)
    pltName = fileName + "estimation_error"
    figureList[pltName] = plt.figure(8)
    
    # CBF over time   
    fig_cbf_score = plot_cbf_safety_score(simulation_time_history, h_history, h_error_time, h_error_history, safety_margin=0.05)
    pltName = fileName + "CBF_Score"
    figureList[pltName] = fig_cbf_score
    
    # Generate the physical geometric avoidance proof
    fig_keepout = plot_hill_frame_traj(true_state_history, L_chief, L_servicer, delta_disc, phase_id_history)
    pltName = fileName + "KeepOut_Zone_2D"
    figureList[pltName] = fig_keepout

    # Generate the 3D trajectory plot
    fig_3d = plot_3d_relative_trajectory(true_state_history, L_chief, L_servicer, delta_disc, phase_id_history)
    pltName = fileName + "KeepOut_Zone_3D"
    figureList[pltName] = fig_3d

    # Safely truncate arrays to match lengths before saving
    min_len_state = min(len(simulation_time_history), len(true_state_history))
    
    # Unpack handoff_log (launch snapshot) into pure numpy arrays
    if handoff_log:
        launch_t_sec = np.array([h['t'] for h in handoff_log])
        launch_true_state = np.array([h['true_state'] for h in handoff_log])
        launch_frozen_est = np.array([h['frozen_est'] for h in handoff_log])
    else:
        launch_t_sec = np.array([])
        launch_true_state = np.empty((0, 6))
        launch_frozen_est = np.empty((0, 6))


    
    np.savez_compressed(
        f"sim_data_{setup_name}.npz",

        time_sim_min=np.array(simulation_time_history[:min_len]),
        phase_id=np.array(phase_id_history[:min_len]),

        # Exp A, B, C, D: SVO & State Estimation
        true_state=np.array(true_state_history[:min_len]),          # (N, 6) [rho_H, rho_Prime_H]
        z_meas=np.array(z_meas_history[:min_len]),                  # (N, 6) noisy measurement
        svo_est=np.array(svo_est_history[:min_len]),                # (N, 6) x_est (CCG center)
        svo_c=np.array(svo_c_history[:min_len]),                    # (N, 6) c_X_k (unshifted center)
        svo_xhat=np.array(svo_xhat_history[:min_len]),              # (N, 6) Luenberger point estimate
        svo_bounds=np.array(svo_bounds_history[:min_len]),          # (N, 6, 2) [lo, hi] shifted by c - x_true
        svo_exact_in_set=np.array(svo_exact_in_set_history[:min_len]), # (N,) 1=in, 0=escape, -1=empty
        svo_dims=np.array(svo_dims_history[:min_len]),              # (N, 3) [ng_p, ng_e, nc_p]
        svo_time_ms=np.array(svo_time_ms_history[:min_len]),        # (N,) ms per updateSVO

        # Attitude & MEKF Estimation
        mekf_servicer=np.array(mekf_servicer_history[:min_len]),    # (N, 6) [sigma_est, omega_est]
        mekf_chief=np.array(mekf_chief_history[:min_len]),          # (N, 6) [sigma_CN_est, omega_CN_est]
        true_servicer_att=np.array(true_servicer_att_history[:min_len]), # (N, 6)
        true_chief_att=np.array(true_chief_att_history[:min_len]),  # (N, 6)
        att_err_norm=np.array(att_err_history[:min_len]),           # (N,) ||sigma_BR||

        # Exp E: CBF Safety & Invariance
        time_h=np.array(simulation_time_history[:min_len]),
        h_score=np.array(h_history[:min_len]),                      # (N,) h_k
        h_pred=np.array(h_pred_history[:min_len]),                  # (N,) predicted h_{k+1}
        h_error_time=np.array(h_error_time),                        # (N-1,)
        h_error=np.array(h_error_history),                          # (N-1,) h_k - h_pred
        cbf_intervened=np.array(cbf_intervened_history[:min_len]),  # (N,) 1 if modified u
        cbf_converged=np.array(cbf_converged_history[:min_len]),    # (N,) 1 if KKT converged
        cbf_iters=np.array(cbf_iters_history[:min_len]),            # (N,) KKT Newton iterations
        cbf_time_ms=np.array(cbf_time_ms_history[:min_len]),        # (N,) ms per CBF step

        # Exp F, G: MPC, Control Effort (Delta-v), & Target Tracking
        cz_inclusion=np.array(cz_inclusion_history[:min_len]),      # (N,) tightest K_set index
        true_cz_inclusion=np.array(true_cz_inclusion_history[:min_len]),
        t_scaling_est=np.array(t_scaling_est_history[:min_len]),
        t_scaling_true=np.array(t_scaling_true_history[:min_len]),
        slack_match=np.array(slack_match_history[:min_len]),        # (N,) ||slack_match_0||^2
        mpc_status=np.array(mpc_status_history[:min_len]),          # (N,) 1=ok, 0=fail, -1=Tier2
        mpc_time_ms=np.array(mpc_time_ms_history[:min_len]),        # (N,) ms per opti.solve()
        thrust=np.array(T_req_history[:min_len]),                   # (N,) ||F_req_B|| (N)
        F_applied_B=np.array(F_applied_history[:min_len]),          # (N, 3) applied force (N)
        u_opt_H=np.array(u_opt_H_history[:min_len]),                # (N, 3) Hill accel cmd (m/s^2)
        distance=np.array(port_distance_history[:min_len]),         # (N,) port-to-standoff dist (m)
        rel_port_pos=np.array(rel_port_pos_history[:min_len]),      # (N, 3) port-to-standoff vec (m)
        virtual_target_dist=np.array(virtual_target_dist_history[:min_len]),
        pos_err_norm=np.array(pos_err_norm_history[:min_len]),
        vel_err_norm=np.array(vel_err_norm_history[:min_len]),
        true_pos_err_norm=np.array(true_pos_err_norm_history[:min_len]),  # <-- NEW
        true_vel_err_norm=np.array(true_vel_err_norm_history[:min_len]),  # <-- NEW

        # --- 2. Handoff, Capture, & Docking Snapshot Data (Exp F) ---
        t_phase_turn=np.float64(phase_transition_times.get('inspection_turn', np.nan)),
        t_phase_hold=np.float64(phase_transition_times.get('holding', np.nan)),
        t_phase_app_dock=np.float64(phase_transition_times.get('approach_dock', np.nan)),
        t_release_min=np.float64(t_release_min),
        u_virt_H=np.array(u_virt_history[:min_len]), 
        u_port_H=np.array(u_port_history[:min_len]), 
        t_docked_min=np.float64(t_docked_min),
        capture_pos_error=np.float64(capture_pos_error),
        capture_vel_error=np.float64(capture_vel_error),
        capture_angle_deg=np.float64(capture_angle_deg),
        capture_closing_speed=np.float64(capture_closing_speed),
        capture_t_sim_s=np.float64(capture_t_sim_s),
        capture_success=np.float64(capture_success),
        flight_log=np.array(flight_log) if len(flight_log) else np.empty((0, 31)),
        gate_log=np.array(launch_gate_log) if len(launch_gate_log) else np.empty((0, 4)),
        launch_u_aim=np.array(launch_info.get('u_aim', np.full(3, np.nan))),
        launch_u_true=np.array(launch_info.get('u_true', np.full(3, np.nan))),
        launch_u_oracle=np.array(launch_info.get('u_oracle', np.full(3, np.nan))),
        launch_t_flight_planned=np.float64(launch_info.get('t_flight_planned', np.nan)),
        launch_t_sim_s=np.float64(launch_info.get('t_sim_s', np.nan)),
        launch_miss_knowledge=np.float64(launch_info.get('miss_knowledge', np.nan)),
        launch_miss_pointing=np.float64(launch_info.get('miss_pointing', np.nan)),
        launch_miss_true_pred=np.float64(launch_info.get('miss_true_pred', np.nan)),
        launch_miss_pred_est=np.float64(launch_info.get('miss_pred_est', np.nan)),
        cfg_v_launch=np.float64(v_launch),
        cfg_m_tip=np.float64(m_tip),
        cfg_servicer_mass_kg=np.float64(servicer_mass),
        cfg_capture_cone_deg=np.float64(CAPTURE_CONE_DEG),
        cfg_launch_gate_tol=np.float64(LAUNCH_GATE_TOL),
        cfg_terminal_box_pos=np.float64(TERMINAL_BOX_POS),
        cfg_terminal_box_vel=np.float64(TERMINAL_BOX_VEL),
        cfg_aim_mode=np.str_(cfg.get('aim_mode', 'intercept')),
        launch_t_sec=launch_t_sec,
        launch_true_state=launch_true_state,
        launch_frozen_est=launch_frozen_est,

        #  1-Hz Basilisk Recorder Logs (for RW & Tip Handoff Plots) 
        bsk_time_min=np.array(timeData),
        bsk_sigma_BR=np.array(dataSigmaBR),
        bsk_omega_BR_B=np.array(dataOmegaBR),
        bsk_rw_cmd_torque=np.array(dataUsReq),
        bsk_rw_actual_torque=np.array(dataRW),
        bsk_rw_speeds_rpm=np.array(dataOmegaRW) / macros.RPM,
        bsk_dist_tip_to_servicer=np.array(dist_tip_to_servicer),
        bsk_dist_tip_to_chief=np.array(dist_tip_to_chief),
        bsk_tip_omega=np.array(omegaTip),
        bsk_tip_sigma=np.array(sigmaTip),
        bsk_tip_omega_rel_norm=np.array(omega_rel_norm),
        bsk_tip_sigma_rel_norm=np.array(sigma_rel_norm),

        # Run Configuration Metadata (Self-Contained Parameters) 
        cfg_svo_horizon=np.int64(svo_horizon),
        cfg_step_time=np.float64(step_time),
        cfg_N_horizon=np.int64(N_horizon),
        cfg_w_max_val=np.float64(w_max_val),
        cfg_w_max_svo=np.float64(w_max_svo),
        cfg_v_max_val=np.float64(v_max_val),
        cfg_max_thrust=np.float64(max_thrust),
        cfg_servicer_mass=np.float64(servicer_mass),
        cfg_L_chief=np.float64(L_chief),
        cfg_L_servicer=np.float64(L_servicer),
        cfg_standoff_distance=np.float64(standoff_distance),
        cfg_hold_distance=np.float64(hold_distance),
        cfg_delta_disc=np.float64(delta_disc),
        cfg_r_effective=np.float64(r_effective),
        cfg_gamma=np.float64(gamma),
        cfg_hk_threshold=np.float64(HK_THRESHOLD),
        cfg_capture_pos_tol=np.float64(CAPTURE_POS_TOL),
        cfg_capture_vel_tol=np.float64(CAPTURE_VEL_TOL),
        cfg_walk_pos=np.float64(walk_pos),
        cfg_rng_seed=np.int64(rng_seed)
    )

    if show_plots:
        plt.show()

    # close the plots being saved off to avoid over-writing old and new figures
    plt.close("all")

    gravFactory.unloadSpiceKernels()

    return figureList


#
# This statement below ensures that the unit test scrip can be run as a
# stand-along python script
#
if __name__ == "__main__":
    run(
        True      # show_plots
    )