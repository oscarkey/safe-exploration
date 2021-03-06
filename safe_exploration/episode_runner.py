# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 11:11:23 2017

@author: tkoller
"""
import time
import warnings

import numpy as np
from scipy.spatial.qhull import ConvexHull

from . import utils_ellipsoid
from .safempc_cem import MpcResult
from .sampling_models import MonteCarloSafetyVerification
from .utils import generate_initial_samples, unavailable
from .utils_config import create_solver, create_env
from .utils_sacred import SacredAggregatedMetrics

try:
    import matplotlib.pyplot as plt
    _has_matplotlib = True
except:
    _has_matplotlib = False


def run_episodic(conf, metrics: SacredAggregatedMetrics, visualize=False):
    """ Run episode setting """

    warnings.warn("Need to check relative dynamics")

    X_all = []
    y_all = []
    cc_all = []
    exit_codes_all = []
    safety_failure_all = []
    for k in range(conf.n_scenarios):

        env = create_env(conf, conf.env_name, conf.env_options)
        solver, safe_policy = create_solver(conf, env)

        solver.init_solver(conf.cost)
        if conf.init_mode is None:
            X = None
            y = None
            have_initial_samples = False
        else:
            X, y = generate_initial_samples(env, conf, conf.relative_dynamics, solver,
                                            safe_policy)
            if conf.plot_initial_samples:
                axes = plt.axes()
                plotted = env.plot_states(axes, [X[:, :env.n_s]], includes_initial_samples=True)
                if plotted:
                    plt.show()
            solver.update_model(X, y, opt_hyp=True, reinitialize_solver=True, replace_old=False)
            metrics.save_array(X, f'initial_samples_{k}')
            have_initial_samples = True

        X_list = [X]
        y_list = [y]
        exit_codes_k = []
        safety_failure_k = []
        cc_k = []

        episode = 0
        total_steps = 0
        while total_steps < conf.total_steps:
            print(f'Starting episode {episode+1} in scenario {k+1}/{conf.n_scenarios} '
                  f'after {total_steps}/{conf.total_steps} steps')

            # If we have nearly reached the maximum number of desired steps, restrict the episode length.
            max_episode_steps = min(conf.n_steps, conf.total_steps - total_steps)

            xx, yy, cc, exit_codes_i, safety_failure = do_rollout(
                env, max_episode_steps, scenario_id=k, episode_id=episode, metrics=metrics,
                cost=conf.rl_immediate_cost,
                solver=solver,
                plot_ellipsoids=conf.plot_ellipsoids,
                plot_trajectory=conf.plot_trajectory,
                save_plots_to_sacred=conf.save_plots_to_sacred,
                plot_episode_trajectory=conf.plot_episode_trajectory,
                render=conf.render,
                obs_frequency=conf.obs_frequency)

            if X is None:
                X = xx
                y = yy
            else:
                X = np.vstack((X, xx))
                y = np.vstack((y, yy))

            X_list += [xx]
            y_list += [yy]
            cc_k += [cc]
            exit_codes_k += [exit_codes_i]
            safety_failure_k += [safety_failure]

            metrics.save_array(xx, f'states_actions_{k}_{episode}')

            if have_initial_samples:
                states_excl_initial_samples = np.vstack(X_list[1:])[:, :env.n_s]
            else:
                states_excl_initial_samples = np.vstack(X_list)[:, :env.n_s]
            metrics.log_scalar('sample_variance', states_excl_initial_samples.var(), episode)
            if states_excl_initial_samples.shape[0] >= 3:
                sample_volume = ConvexHull(states_excl_initial_samples).volume
            else:
                sample_volume = 0.
            metrics.log_scalar('sample_volume', sample_volume, episode)

            if conf.plot_states:
                axes = plt.axes()
                states = [x[:, :env.n_s] for x in X_list]
                plotted = env.plot_states(axes, states, have_initial_samples)
                if plotted:
                    if conf.save_plots_to_sacred:
                        metrics.save_figure(plt.gcf(), f'training_points_{k}_{episode}')
                        plt.clf()
                    else:
                        plt.show()

            training_start_time = time.time()
            solver.update_model(X, y, opt_hyp=conf.opt_hyp, reinitialize_solver=True)
            training_end_time = time.time()

            metrics.log_scalar('training_time', training_end_time - training_start_time, episode)
            metrics.log_scalar('num_samples', X.shape[0], episode)

            # Returned states does not include initial state (why?).
            total_steps += xx.shape[0] + 1
            episode += 1

        exit_codes_all += [exit_codes_k]
        safety_failure_all += [safety_failure_k]
        cc_all += [cc_k]
        X_all += [X_list]
        y_all += [y_list]

    metrics.flush()

    if not conf.data_savepath is None:
        savepath_data = "{}/{}".format(conf.save_path, conf.data_savepath)
        a, b = solver.lin_model
        np.savez(savepath_data, X=X, y=y, a=a, b=b, init_mode=conf.init_mode)

    if conf.save_results:
        save_name_results = conf.save_name_results
        if save_name_results is None:
            save_name_results = "results_episode"

        savepath_results = conf.save_path + "/" + save_name_results

        results_dict = dict()
        results_dict["cc_all"] = cc_all
        results_dict["X_all"] = X_all
        results_dict["y_all"] = y_all
        results_dict["exit_codes"] = exit_codes_all
        results_dict["safety_failure_all"] = safety_failure_all

        np.save(savepath_results, results_dict)

        # TO-DO: may wanna do this aswell
        # gp_dict = gp.to_dict()
        # save_data_gp_path = "{}/res_gp".format(save_path)
        # np.save(save_data_gp_path,gp_dict)


@unavailable(not _has_matplotlib, "matplotlib", conditionals=["plot_ellipsoids,plot_trajectory"])
def do_rollout(env, n_steps, scenario_id: int, episode_id: int, metrics: SacredAggregatedMetrics, solver=None,
               cost=None,
               plot_trajectory=True, save_plots_to_sacred=False,
               verbosity=1, sampling_verification=False,
               plot_ellipsoids=False, plot_episode_trajectory=False, render=False,
               check_system_safety=False, savedir_trajectory_plots=None, mean=None,
               std=None, obs_frequency=1):
    """ Perform a rollout on the system

    """

    state = env.reset(mean, std)

    xx = np.zeros((1, env.n_s + env.n_u))
    yy = np.zeros((1, env.n_s))
    exit_codes = np.zeros((1, 1))
    obs = state

    cc = []
    n_successful = 0
    mpc_results = []
    total_time_in_solver = 0
    env_result = -1
    safety_failure = False
    if plot_trajectory:
        fig, ax = env.plot_safety_bounds()

        ell = None

    if sampling_verification:
        gp = solver.gp
        sampler = MonteCarloSafetyVerification(gp)

    if check_system_safety:
        n_inside = 0
        n_test_safety = 0

    for i in range(n_steps):
        p_traj = None
        q_traj = None
        k_fb = None
        k_ff = None

        if solver is None:
            action = env.random_action()
            exit_code = 5
        else:
            t_start_solver = time.time()
            action, mpc_result = solver.get_action(state)  # ,lqr_only = True)
            t_end_solver = time.time()

            t_solver = t_end_solver - t_start_solver
            total_time_in_solver += t_solver

            exit_code = 1 if mpc_result in (MpcResult.FOUND_SOLUTION, MpcResult.PREVIOUS_SOLUTION) else 0
            mpc_results.append(mpc_result)

            if verbosity > 0:
                print(("total time solver in ms: {}".format(t_solver)))

        action, next_state, observation, done, env_result = env.step(action)
        if not cost is None:
            c = [cost(next_state)]
            cc += c
            if verbosity > 0:
                print(("Immediate cost for current step: {}".format(c)))
        if verbosity > 0:
            print(("\n==== Applied normalized action at time step {} ====".format(i)))
            print(action)
            print("\n==== Next state (normalized) ====")
            print(next_state)
            print("==========================\n")
        if render:
            env.render()

        # Plot the trajectory planned by the MPC solver
        if plot_trajectory:
            if not solver is None and plot_ellipsoids and solver.has_openloop:
                p_traj, q_traj, k_fb, k_ff = solver.get_trajectory_openloop(
                    state, get_controls=True)

                if not ell is None:
                    for j in range(len(ell)):
                        ell[j].remove()
                ax, ell = env.plot_ellipsoid_trajectory(p_traj, q_traj, ax=ax,
                                                        color="r")
                fig.canvas.draw()
                # plt.draw()

                plt.show(block=False)
                plt.pause(0.5)
            ax = env.plot_state(ax)
            fig.canvas.draw()
            plt.show(block=False)
            plt.pause(0.2)
            if not savedir_trajectory_plots is None:
                save_name = "img_step_{}.png".format(i)
                save_path = "{}/{}".format(savedir_trajectory_plots, save_name)
                plt.savefig(save_path)

        # Verify whether the GP distribution is inside the ellipsoid over multiple
        # steps via sampling
        if sampling_verification:
            if p_traj is None:
                p_traj, q_traj, k_fb, k_ff = solver.get_trajectory_openloop(
                    state,
                    get_controls=True)

            _, s_all = sampler.sample_n_step(state[:, None], k_fb, k_ff, p_traj,
                                             n_samples=300)
            safety_ratio, _ = sampler.inside_ellipsoid_ratio(s_all, q_traj, p_traj)
            if verbosity > 0:
                print(("\n==== GP samples inside Safety Ellipsoids (time step {}) "
                       "====".format(i)))
                print(safety_ratio)
                print("==========================\n")

        # check if the true system is inside the one-step ellipsoid by checking if the
        # next state is inside p,q ellipsoid
        if not solver is None:
            if check_system_safety:
                if p_traj is None:
                    p_traj, q_traj, k_fb, k_ff = solver.get_trajectory_openloop(
                        state,
                        get_controls=True)
                bool_inside = utils_ellipsoid.sample_inside_ellipsoid(
                    next_state, p_traj[0, :, None], q_traj[0])

                n_test_safety += 1
                if bool_inside:
                    n_inside += 1
                if verbosity > 0:
                    print((
                        "\n==== Next state inside uncertainty ellipsoid:{}"
                        " ====\n".format(bool_inside)))

        state_action = np.hstack((state, action))
        xx = np.vstack((xx, state_action))
        yy = np.vstack((yy, observation))

        exit_codes = np.vstack((exit_codes, exit_code))
        n_successful += 1
        state = next_state
        if done:
            safety_failure = True
            break

    metrics.log_scalar('episode_length', n_successful, episode_id)
    metrics.log_scalar('mpc_found_solution_count', mpc_results.count(MpcResult.FOUND_SOLUTION), episode_id)
    metrics.log_scalar('mpc_previous_solution_count', mpc_results.count(MpcResult.PREVIOUS_SOLUTION), episode_id)
    metrics.log_scalar('safe_controller_fallback_count', mpc_results.count(MpcResult.SAFE_CONTROLLER), episode_id)
    metrics.log_scalar('mean_time_in_solver', float(total_time_in_solver) / n_successful, episode_id)
    metrics.log_scalar('env_result', env_result, episode_id)
    metrics.log_non_scalars(env.collect_metrics(), episode_id)
    if solver is not None:
        metrics.log_non_scalars(solver.collect_metrics(), episode_id)

    if plot_episode_trajectory:
        axes = plt.axes()
        plotted = env.plot_current_trajectory(axes)
        if plotted:
            if save_plots_to_sacred:
                metrics.save_figure(plt.gcf(), f'trajectories_{scenario_id}_{episode_id}')
                plt.clf()
            else:
                plt.show()

    if n_successful == 0:
        warnings.warn("Agent survived 0 steps, cannot collect data")
        xx = []
        yy = []
        exit_codes = []
        cc = []
    else:
        xx = xx[1:-1:obs_frequency, :]
        yy = yy[1:-1:obs_frequency, :]
        exit_codes = exit_codes[1:, :]

    print(("Agent survived {} steps".format(n_successful)))
    if verbosity > 0:
        print("========== State/Action Trajectory ===========")
        print(xx)
        if check_system_safety and n_test_safety > 0:
            print("\n======= percentage system steps inside safety bounds =======")
            print((float(n_inside) / n_test_safety))
    return xx, yy, cc, exit_codes, safety_failure
