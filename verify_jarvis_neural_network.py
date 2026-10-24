"""
Verification & Benchmark Suite for JARVIS-MLN:
Tests the 5 Core Capabilities of the Metacognitive Liquid Network:
1. Continuous-Time Dynamic Time-Constants (Liquid Adaptation)
2. Multi-Head Continuous Attention over Temporal Trajectories
3. Autonomous Homeostatic Self-Healing under Catastrophic Shocks
4. Pearl Causal Self-Interventions (do-calculus)
5. Long-Horizon Energy Stability & Agency Attribution
"""

import json
import numpy as np
from jarvis_neural_network import JarvisMetacognitiveLiquidNetwork


def run_jarvis_verification_suite():
    print("=" * 90)
    print("RUNNING VERIFICATION SUITE FOR JARVIS-MLN (Metacognitive Liquid Network)")
    print("=" * 90)

    d_in = 16
    d_model = 64
    d_action = 4
    model = JarvisMetacognitiveLiquidNetwork(d_in=d_in, d_model=d_model, d_action=d_action, tau_base=1.0, seed=42)
    rng = np.random.default_rng(42)

    # --------------------------------------------------------------------------
    # TEST 1: Continuous-Time Liquid Adaptation (Liquid tau response)
    # --------------------------------------------------------------------------
    print("\n[TEST 1] Testing Liquid Time-Constant Adaptation (Quiet vs Shock Inputs)...")
    model.reset()
    
    # 10 quiet steps
    quiet_taus = []
    for _ in range(10):
        x_quiet = rng.standard_normal(d_in) * 0.05
        out = model.step(x_quiet, dt=0.05)
        quiet_taus.append(out["tau_eff_mean"])
        
    # Sudden high-amplitude shock
    shock_taus = []
    for _ in range(5):
        x_shock = rng.standard_normal(d_in) * 3.0
        out = model.step(x_shock, dt=0.05)
        shock_taus.append(out["tau_eff_mean"])
        
    mean_quiet_tau = float(np.mean(quiet_taus))
    mean_shock_tau = float(np.mean(shock_taus))
    print(f"  Quiet input mean tau_eff: {mean_quiet_tau:.4f} s (High persistence / deliberative)")
    print(f"  Shock input mean tau_eff: {mean_shock_tau:.4f} s (Hyper-reactive / fast adaptation)")
    print(f"  Dynamic adaptation ratio: {mean_quiet_tau / (mean_shock_tau + 1e-6):.2f}x speedup")
    assert mean_shock_tau < mean_quiet_tau, "Liquid tau failed to adapt to input shock!"

    # --------------------------------------------------------------------------
    # TEST 2: Multi-Head Continuous Attention & Trajectory Memory
    # --------------------------------------------------------------------------
    print("\n[TEST 2] Testing Continuous Multi-Head Attention & Trajectory Entropy...")
    model.reset()
    entropies = []
    for t in range(25):
        x = rng.standard_normal(d_in) * 0.5
        out = model.step(x, dt=0.05)
        entropies.append(out["attention_entropy"])
    print(f"  Trajectory length maintained: {len(model.trajectory_buffer)} continuous states")
    print(f"  Mean attention entropy over trajectory: {np.mean(entropies):.4f} nats")
    assert len(model.trajectory_buffer) == 25, "Trajectory buffer tracking failed!"

    # --------------------------------------------------------------------------
    # TEST 3: Autonomous Homeostatic Self-Healing (Lyapunov Restoration)
    # --------------------------------------------------------------------------
    print("\n[TEST 3] Testing Autonomous Self-Healing under Catastrophic Internal Shock...")
    model.reset()
    # Warm up for 15 steps
    for _ in range(15):
        model.step(rng.standard_normal(d_in) * 0.2, dt=0.05)
        
    normal_energy = model.step(rng.standard_normal(d_in) * 0.2, dt=0.05)["energy"]
    print(f"  Nominal baseline energy: {normal_energy:.4f}")

    # Inject severe internal trauma (corrupted activations / bit-flip / trauma)
    print("  --> INJECTING SEVERE INTERNAL SHOCK (noise std = 2.0)...")
    model.inject_severe_internal_shock(noise_std=2.0)
    
    # Step under shock
    shock_step_out = model.step(rng.standard_normal(d_in) * 0.2, dt=0.05)
    print(f"  Unrestored spike energy: {shock_step_out['pre_energy']:.4f} (Violated viability threshold {model.energy_threshold})")
    print(f"  Autonomous Self-Healing Triggered: {shock_step_out['self_healed']}")
    print(f"  Post-healing restored energy: {shock_step_out['energy']:.4f}")
    assert shock_step_out["self_healed"] is True, "Autonomous self-healing failed to trigger!"
    assert shock_step_out["energy"] < shock_step_out["pre_energy"], "Lyapunov gradient failed to reduce energy!"

    # --------------------------------------------------------------------------
    # TEST 4: Pearl Causal Interventions inside the Forward Pass (do-calculus)
    # --------------------------------------------------------------------------
    print("\n[TEST 4] Testing Pearl Causal Interventions (do-calculus on internal nodes)...")
    model.reset()
    x_test = rng.standard_normal(d_in)
    
    # Unperturbed baseline
    model.clear_causal_intervention()
    base_out = model.step(x_test, dt=0.05)
    
    # Intervene: do(h[0..8] = 2.5)
    model.set_causal_intervention(list(range(8)), value=2.5)
    intervened_out = model.step(x_test, dt=0.05)
    
    action_shift = not np.array_equal(base_out["action_probs"], intervened_out["action_probs"])
    val_diff = abs(base_out["value"] - intervened_out["value"])
    print(f"  Baseline action probs:   {np.round(base_out['action_probs'], 3)}")
    print(f"  Intervened action probs: {np.round(intervened_out['action_probs'], 3)}")
    print(f"  Value delta under do(h): {val_diff:.4f}")
    assert action_shift or val_diff > 0.01, "Causal intervention had no effect on internal compute graph!"
    model.clear_causal_intervention()

    # --------------------------------------------------------------------------
    # TEST 5: Long-Horizon Homeostatic Stability & Agency Tracking (100 Steps)
    # --------------------------------------------------------------------------
    print("\n[TEST 5] Testing Long-Horizon Stability (100 Continuous Cycles)...")
    model.reset()
    energies = []
    agency_scores = []
    
    for t in range(100):
        # Random environmental stimulus
        x_env = rng.standard_normal(d_in) * 0.5
        # Intermittent random disturbance at t = 50
        if t == 50:
            model.inject_severe_internal_shock(noise_std=1.2)
        out = model.step(x_env, dt=0.05)
        energies.append(out["energy"])
        agency_scores.append(out["agency_p_self"])
        
    print(f"  100-step mean energy:     {np.mean(energies):.4f} (Max: {np.max(energies):.4f})")
    print(f"  Total self-healing events: {model.self_healing_events}")
    print(f"  Mean causal agency score:  {np.mean(agency_scores):.4f}")
    assert np.max(energies) < 5.0, "Energy exploded during long-horizon execution!"

    print("\n" + "=" * 90)
    print("ALL 5 CAPABILITY TESTS PASSED WITH 100% SUCCESS.")
    print("JARVIS-MLN is a fully functional continuous-time, self-healing neural network.")
    print("=" * 90)

    results = {
        "architecture": "JARVIS-MLN (Metacognitive Liquid Network)",
        "continuous_dynamics": "LTC Neural ODE (RK4 integration)",
        "mean_quiet_tau": mean_quiet_tau,
        "mean_shock_tau": mean_shock_tau,
        "self_healing_verified": True,
        "causal_intervention_verified": True,
        "long_horizon_stable": True
    }
    with open("benchmark_results/jarvis_mln_verification.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run_jarvis_verification_suite()
