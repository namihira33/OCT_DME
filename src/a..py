import math
from shapely.geometry import Polygon
import itertools

class Line:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    
    def get_points(self):
        return [(self.x1, self.y1), (self.x2, self.y2)]
    
    def get_center(self):
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def length(self):
        return math.sqrt((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2)

def find_polygons_from_lines(nearby_lines):
    polygons = []
    for n in range(3, 6):  # 3辺から5辺の多角形を考慮
        for combination in itertools.permutations(nearby_lines, n):
            points = []
            for i, line in enumerate(combination):
                if i == 0:
                    points.append(line.get_points()[0])
                    points.append(line.get_points()[1])
                else:
                    if line.get_points()[0] == points[-1]:
                        points.append(line.get_points()[1])
                    elif line.get_points()[1] == points[-1]:
                        points.append(line.get_points()[0])
                    else:
                        break
            else:
                if len(points) == n + 1 and points[0] == points[-1]:
                    polygon = Polygon(points)
                    if polygon.is_valid and not polygon.is_empty:
                        polygons.append(polygon)
    return polygons

# テストデータ
line1 = Line(0, 0, 1, 0)
line2 = Line(1, 0, 1, 1)
line3 = Line(1, 1, 0, 1)
line4 = Line(0, 1, 0, 0)
line5 = Line(0, 0, 1, 1)

nearby_lines = [line1, line2, line3, line4, line5]

# 多角形を見つける
polygons = find_polygons_from_lines(nearby_lines)

# 結果を表示
for i, polygon in enumerate(polygons):
    print(f"多角形 {i + 1}:")
    print(list(polygon.exterior.coords))
    print(f"面積: {polygon.area}")
    print()