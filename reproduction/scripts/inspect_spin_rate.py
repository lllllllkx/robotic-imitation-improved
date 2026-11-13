import csv, math, os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for name in ["dog_spin", "dog_pace"]:
    p = os.path.join(root, "outputs", "reference", name + ".csv")
    rows = list(csv.DictReader(open(p)))
    yaws = []
    for r in rows:
        x, y, z, w = (float(r[k]) for k in ["root_qx", "root_qy", "root_qz", "root_qw"])
        yaws.append(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    un = [yaws[0]]
    for a in yaws[1:]:
        d = a - un[-1]
        while d > math.pi:
            d -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
        un.append(un[-1] + d)
    dur = float(rows[-1]["time"])
    print("%s frames=%d dur=%.3f s  yaw_total=%+.3f rad  yaw_rate=%.3f rad/s"
          % (name, len(rows), dur, un[-1] - un[0], (un[-1] - un[0]) / dur))
