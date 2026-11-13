import csv, os, statistics

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for tag in ["dog_pace", "dog_spin"]:
    p = os.path.join(root, "outputs", "eval", tag, "steps.csv")
    rows = list(csv.DictReader(open(p)))
    for key in ["yaw_err_rad", "speed_mps", "sim_z", "pose_err_rad", "root_pos_err_m"]:
        v = [abs(float(r[key])) if key == "yaw_err_rad" else float(r[key]) for r in rows]
        print("%-9s %-16s mean=%.4f max=%.4f" % (tag, key, statistics.mean(v), max(v)))
    print("%-9s steps=%d" % (tag, len(rows)))
