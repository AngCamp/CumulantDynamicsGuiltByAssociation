"""Write hippocampus_config_v3.json: brief pauses, short running bouts, and
pyramidal place coding that stays active during stillness."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.dirname(HERE)                      # where hippocampus.py lives

def main():
    c = json.load(open(os.path.join(SIM, "hippocampus_config.json")))
    P = c["populations"]

    # ---- longer session: short bouts need more of them for statistics
    c["duration_s"] = 300.0
    c["output_dir"] = "output_v3"

    # ---- bout structure: STILL 0.5-3 s (brief pauses), WALK 1-5 s (bouts)
    c["states"]["dwell_mean_s"] = {"WALK": 2.5, "STILL": 1.2}
    c["states"]["dwell_sigma_log"] = 0.5
    # stopping must be fast enough that a 1 s pause actually reaches speed 0,
    # otherwise the SWR trigger (speed == 0) almost never fires
    c["locomotion"]["ramp_tau_s"] = 0.40          # was 1.2
    # shorter still periods -> raise the SWR rate to keep enough events
    c["swr"]["rate_hz"] = 2.0                     # was 0.6

    # ---- pyramidal: higher drive, deeper theta (as in v2)
    P["pyramidal"]["states"]["WALK"]["baseline"] = 0.30
    P["pyramidal"]["states"]["WALK"]["osc_depth"] = 0.60
    P["pyramidal"]["states"]["SWR"]["baseline"] = 0.30
    # ---- NEW: place coding stays on during stillness
    # baseline set so the STILL rate lands near 1 Hz (rate ~ exp(beta*v),
    # beta=4, so -0.25 in baseline is a factor exp(-1.0) = 0.37);
    # stim_gain stays high so place coding is still present during stillness
    P["pyramidal"]["states"]["STILL"]["baseline"] = 0.00     # was -0.10 / 0.25
    P["pyramidal"]["states"]["STILL"]["stim_gain"] = 0.70    # was 0.45

    # ---- interneurons pulled back so they do not swamp the pyramidal cells
    P["basket_pv"]["states"]["WALK"]["baseline"] = 0.35
    P["basket_pv"]["states"]["SWR"]["baseline"] = 1.20
    P["olm"]["states"]["WALK"]["baseline"] = 0.25
    P["chandelier"]["states"]["WALK"]["baseline"] = 0.25
    c["connectivity"]["pv_to_pyr"]["weight"] = -0.030
    c["connectivity"]["chandelier_veto"]["strength"] = 0.35
    c["connectivity"]["olm_gate"]["strength"] = 0.20

    out = os.path.join(SIM, "hippocampus_config_v3.json")
    json.dump(c, open(out, "w"), indent=2)
    print("wrote", out)
    print(f"  duration {c['duration_s']:.0f} s; dwell means "
          f"WALK {c['states']['dwell_mean_s']['WALK']} s, "
          f"STILL {c['states']['dwell_mean_s']['STILL']} s "
          f"(sigma_log {c['states']['dwell_sigma_log']})")
    print(f"  SWR rate {c['swr']['rate_hz']} Hz; stop ramp tau "
          f"{c['locomotion']['ramp_tau_s']} s")
    print(f"  pyramidal STILL: baseline "
          f"{P['pyramidal']['states']['STILL']['baseline']}, "
          f"stim_gain {P['pyramidal']['states']['STILL']['stim_gain']}")

if __name__ == "__main__":
    main()
