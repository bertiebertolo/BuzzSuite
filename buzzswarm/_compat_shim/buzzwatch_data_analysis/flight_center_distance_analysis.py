def calculate_cage_centroid(cage_border_points):
    xs = [p[0] for p in cage_border_points]
    ys = [p[1] for p in cage_border_points]
    return (sum(xs)/len(xs), sum(ys)/len(ys))
