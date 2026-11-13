import csv, os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(root, "outputs", "eval", "dog_spin", "steps.csv")
rows = [r for r in csv.DictReader(open(p)) if int(r["episode"]) == 0]
for i in range(0, 60, 2):
    print("t=%.2f yaw_err=%+.3f root_pose_r=%.3f speed=%.3f"
          % (float(rows[i]["time"]), float(rows[i]["yaw_err_rad"]),
             float(rows[i]["root_pose_reward"]), float(rows[i]["speed_mps"])))
yaw = [abs(float(r["yaw_err_rad"])) for r in rows]
print("first100 mean %.3f  last100 mean %.3f" % (sum(yaw[:100]) / 100, sum(yaw[-100:]) / 100))
