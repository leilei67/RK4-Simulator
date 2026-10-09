from manim import *
import numpy as np

earth_y  = -1
star_y   =  0
galaxy_y =  0.65
b        =  0.4
deflection = 0.15

# ── launch angle from vertical ──
theta0 = 0.55   # radians

class MyScene(Scene):
    def construct(self):
        axes = Axes(
            x_range=[-1, 1, 2],
            y_range=[-1, 1, 2],
            x_length=8,
            y_length=6,
            tips=True,
            axis_config={"tick_size": 0.03},
        )

        labels = axes.get_axis_labels("x", "y")

        lens = Dot(axes.coords_to_point(0, 0), radius=0.23, color=YELLOW)
        lens_glow = Dot(axes.coords_to_point(0, 0), radius=0.25, color=YELLOW, fill_opacity=0.5)
        source = Dot(axes.coords_to_point(0, galaxy_y), radius=0.15, color=PURPLE)
        earth = Dot(axes.coords_to_point(0, earth_y), radius=0.15, color=BLUE)

        self.play(Create(axes), Write(labels))
        self.play(FadeIn(lens), FadeIn(lens_glow), FadeIn(source), FadeIn(earth))

        def generate_trajectory(theta, steps=3000, dt=0.005, strength=0.18, soft_r=0.25):
            x, y = 0.0, earth_y

            vx = np.sin(theta)
            vy = np.cos(theta)

            points = [axes.coords_to_point(x, y)]

            for _ in range(steps):
                r = np.sqrt(x**2 + y**2)
                softened_r = max(r, soft_r)

                ax = -strength * x / softened_r**3
                ay = -strength * y / softened_r**3

                vx += ax * dt
                vy += ay * dt

                v = np.sqrt(vx**2 + vy**2)
                vx /= v
                vy /= v

                x += vx * dt
                y += vy * dt

                points.append(axes.coords_to_point(x, y))

                if np.sqrt((x)**2 + (y)**2) > 1.2:
                    break

            return points

        def draw_line(theta, strength=0.18, soft_r=0.25):
            points = generate_trajectory(theta, strength=strength, soft_r=soft_r)
            path = VMobject(color=YELLOW, stroke_width=2)
            path.set_points_as_corners(points)
            self.play(Create(path), run_time=2, rate_func=linear)
            return theta, path, points
        
        def get_y_at_x0(points):
            coords = [axes.point_to_coords(p) for p in points]
            # find point closest to x=0, skipping the start
            min_x_idx = min(range(10, len(coords)), key=lambda i: abs(coords[i][0]))
            return coords[min_x_idx][1]

        def drawline3(theta1, path1, points1, theta2, path2, points2, strength=0.15):
            theta3 = (theta1+theta2)/2
            theta3, path3, points3= draw_line(theta3, strength=strength)
            return theta3, path3, points3

        theta1, path1, points1 = draw_line(theta0, strength= 0.18)
        theta2, path2, points2 = draw_line(theta0/2, strength= 0.18, soft_r=0.23)

        # ── Label initial rays ────────────────────────────────
        earth_point = axes.coords_to_point(0, earth_y)
        L = 0.4

        # theta_max line (path1, the larger angle)
        ray_line_max = Line(
            earth_point,
            axes.coords_to_point(L * np.sin(theta1), earth_y + L * np.cos(theta1)),
            color=GREEN, stroke_width=1.5,
        )
        vert_line = Line(
            earth_point,
            axes.coords_to_point(0, earth_y + L),
            color=WHITE, stroke_width=1.5,
        )
        arc_max = Angle(ray_line_max, vert_line, radius=0.6, color=GREEN)
        arc_mid_max = arc_max.point_from_proportion(0.5)
        dir_max = arc_mid_max - earth_point
        dir_max /= np.linalg.norm(dir_max)
        label_max = MathTex(r"\theta_{E,\max}", color=GREEN).scale(0.8)
        label_max.move_to(earth_point + dir_max * 1)

        # theta_min line (path2, the smaller angle)
        ray_line_min = Line(
            earth_point,
            axes.coords_to_point(L * np.sin(theta2), earth_y + L * np.cos(theta2)),
            color=RED, stroke_width=1.5,
        )
        arc_min = Angle(ray_line_min, vert_line, radius=0.35, color=RED)
        arc_mid_min = arc_min.point_from_proportion(0.5)
        dir_min = arc_mid_min - earth_point
        dir_min /= np.linalg.norm(dir_min)
        label_min = MathTex(r"\theta_{E,\min}", color=RED).scale(0.8)
        label_min.move_to(earth_point + dir_min * 0.7)

        self.play(
            Create(vert_line),
            Create(arc_max), Write(label_max),
            Create(arc_min), Write(label_min),
        )
        self.wait(0.5)

        # fade them out before the binary search begins
        self.play(FadeOut(vert_line), FadeOut(arc_max), FadeOut(label_max),
                  FadeOut(arc_min), FadeOut(label_min))



        theta3, path3, points3= drawline3(theta1, path1, points1, theta2, path2, points2)
        y3= get_y_at_x0(points3)
        y2= get_y_at_x0(points2)
        y1= get_y_at_x0(points1) #WLOG y is above the star > y2 is below the star
        # Forward along path1, backward along path2 to close the region
        all_points = points1 + list(reversed(points3))

        shaded_region = VMobject(fill_color=YELLOW, fill_opacity=0.2, stroke_width=0)
        shaded_region.set_points_as_corners(all_points)

        
        galaxy_point = axes.coords_to_point(0, galaxy_y)

        # Arrow pointing to the galaxy from the upper right
        arrow = Arrow(
            start=axes.coords_to_point(0.5, galaxy_y + 0.3),
            end=galaxy_point,
            color=WHITE,
            buff=0.15,
            stroke_width=3,
            max_tip_length_to_length_ratio=0.2
        )

        arrow_label = MathTex(r"\text{Source Galaxy}", font_size=20, color=WHITE)
        arrow_label.next_to(arrow.get_start(), RIGHT, buff=0.1)

        circle = Circle(radius=0.4, color=WHITE, stroke_width=2)
        circle.move_to(galaxy_point)

        self.play(Create(circle), GrowArrow(arrow), Write(arrow_label), run_time=0.8)
        self.play(FadeIn(shaded_region))
        self.play(FadeOut(shaded_region))
        self.play(FadeOut(arrow), FadeOut(arrow_label), run_time=0.5)
        self.play(circle.animate.scale(1.3).set_opacity(0), run_time=0.6)
        self.remove(circle)
        
            
        if y3<galaxy_y:
            self.play(FadeOut(path2))
            self.remove(path2)
            theta2 = theta3
            path2=path3
            points2=points3
        else:
            self.play(FadeOut(path1))
            self.remove(path1)
            theta1 = theta3
            path1=path3
            points1=points3

        for i in range(5):
            theta3, path3, points3= drawline3(theta1, path1, points1, theta2, path2, points2)
            y3= get_y_at_x0(points3)
            y2= get_y_at_x0(points2)
            y1= get_y_at_x0(points1) #WLOG y is above the star > y2 is below the star
            
            if y3<galaxy_y:
                self.play(FadeOut(path2))
                self.remove(path2)
                theta2 = theta3
                path2=path3
                points2=points3
            else:
                self.play(FadeOut(path1))
                self.remove(path1)
                theta1 = theta3
                path1=path3
                points1=points3
    
     # ── Einstein Ring radius bracket ──────────────────────
        # Use whichever of path1/path2 is closer to galaxy_y
        final_points = points1 if abs(get_y_at_x0(points1) - galaxy_y) < abs(get_y_at_x0(points2) - galaxy_y) else points2

        # Get the y=0 crossing point in screen space
        coords = [axes.point_to_coords(p) for p in final_points]
        y0_idx = min(range(10, len(coords)), key=lambda i: abs(coords[i][1]))
        ring_x = coords[y0_idx][0]

        # Screen positions
        ring_center = axes.coords_to_point(0, 0)
        ring_radius = axes.coords_to_point(ring_x, 0)

        # Brace on the left side of the y axis
        brace = BraceBetweenPoints(
            ring_center,
            ring_radius,
            color=WHITE,
        )
        brace_label = brace.get_text("Einstein Ring Radius").scale(0.5)

        self.play(Create(brace), Write(brace_label))
        self.wait()

        # ── Theta angle label ─────────────────────────────────
        earth_point = axes.coords_to_point(0, earth_y)
        final_theta = (theta1 + theta2) / 2  # best estimate of ring angle

        L = 0.4  # length of the angle indicator lines

        # Vertical reference line from earth
        vert_line = Line(
            earth_point,
            axes.coords_to_point(0, earth_y + L),
            color=WHITE,
            stroke_width=1.5,
        )

        # Line along the ray direction
        ray_line = Line(
            earth_point,
            axes.coords_to_point(L * np.sin(final_theta), earth_y + L * np.cos(final_theta)),
            color=WHITE,
            stroke_width=1.5,
        )

        angle_arc = Angle(ray_line, vert_line, radius=0.6, color=WHITE)

        # Label at bisector of the arc
        arc_mid = angle_arc.point_from_proportion(0.5)
        direction = arc_mid - earth_point
        direction /= np.linalg.norm(direction)
        theta_label = MathTex(r"\theta_E", color=WHITE).scale(0.8)
        theta_label.move_to(earth_point + direction * 1)

        self.play( Create(angle_arc), Write(theta_label))

        