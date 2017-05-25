DELIM = "\t"
"""
Filter out checkins in a   particular region
"""
def filter_gowalla(param):
    with open("../dataset/gowalla/checkins.txt") as f:
        with open(param.dataset, "w") as f_out:
            lines = ""
            for line in f:
                parts = line.split()
                lat, lon = float(parts[1]), float(parts[2])
                if param.x_min <= lat <= param.x_max and param.y_min <= lon <= param.y_max:
                    userId, locId = parts[0], parts[3]
                    lines += DELIM.join(list(map(str, [userId, locId, lat, lon, "\n"])))

            f_out.write(lines)

    f.close()
    f_out.close()

def filter_yelp(param):
    with open("../dataset/yelp/checkins.txt") as f:
        with open(param.dataset, "w") as f_out:
            lines = ""
            for line in f:
                parts = line.split()
                lat, lon = float(parts[2]), float(parts[3])
                if param.x_min <= lat <= param.x_max and param.y_min <= lon <= param.y_max:
                    userId, locId = parts[0], parts[1]
                    lines += DELIM.join(list(map(str, [userId, locId, lat, lon, "\n"])))

            f_out.write(lines)

    f.close()
    f_out.close()