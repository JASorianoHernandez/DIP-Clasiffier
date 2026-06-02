"""
network.py — Detailed neural network animation for Freshness Classification.

Render with:
    manim -pql network.py NetworkScene       # low quality preview (~2 min)
    manim -pqh network.py NetworkScene       # high quality (~10 min)

Scenes:
    NetworkScene  — full detailed animation
"""

from manim import *
import numpy as np


# ── Palette ────────────────────────────────────────────────────
C_BG       = "#0d1117"
C_FROZEN   = "#4C72B0"
C_FINETUNE = "#DD8452"
C_HEAD     = "#55A868"
C_GREEN    = "#55A868"   # alias for C_HEAD
C_ORANGE   = "#DD8452"   # alias for C_FINETUNE
C_LINEAR   = "#9b59b6"
C_FRESH    = "#2ecc71"
C_ROTTEN   = "#e74c3c"
C_SIGNAL   = "#f1c40f"
C_WHITE    = "#f0f0f0"
C_GRAY     = "#888888"
C_ACCENT   = "#e94560"


# ── Neural layer helper ────────────────────────────────────────

def make_layer(n_nodes, color, spacing=0.45, node_radius=0.18,
               label=None, fill_opacity=0.85):
    """Return a VGroup of circles arranged vertically."""
    nodes = VGroup(*[
        Circle(radius=node_radius,
               fill_color=color, fill_opacity=fill_opacity,
               stroke_color=WHITE, stroke_width=1.5)
        for _ in range(n_nodes)
    ])
    nodes.arrange(DOWN, buff=spacing - node_radius * 2)
    if label:
        lbl = Text(label, font_size=13, color=color).next_to(
            nodes, DOWN, buff=0.2)
        return VGroup(nodes, lbl)
    return VGroup(nodes)


def connect_layers(layer_a_nodes, layer_b_nodes, color=C_GRAY,
                   opacity=0.15, stroke_width=0.8):
    """Draw connections between two layers of nodes."""
    edges = VGroup()
    for na in layer_a_nodes:
        for nb in layer_b_nodes:
            edges.add(Line(
                na.get_right(), nb.get_left(),
                stroke_color=color, stroke_opacity=opacity,
                stroke_width=stroke_width
            ))
    return edges


def signal_dot(color=C_SIGNAL, radius=0.10):
    return Dot(radius=radius, color=color)


# ──────────────────────────────────────────────────────────────
class NetworkScene(Scene):
    """
    Detailed neural network animation covering:
      1. Title
      2. Input image → pixel grid
      3. ResNet-18 backbone — frozen vs fine-tuned layers with signal flow
      4. Condition comparison — C1 vs C3 vs C4 side by side
      5. Projection head — dimensional reduction
      6. Softmax — probability output with animated confidence bars
      7. C1–C4 ablation summary
    """

    def construct(self):
        self.camera.background_color = C_BG
        self._title()
        self._input_image()
        self._backbone_flow()
        self._condition_comparison()
        self._projection_head()
        self._softmax_output()
        self._ablation_summary()

    # ── 1. Title ──────────────────────────────────────────────

    def _title(self):
        t1 = Text("Deep Transfer Learning", font_size=44,
                  color=C_WHITE, weight=BOLD)
        t2 = Text("Freshness Classification · Neural Network Visualization",
                  font_size=20, color=C_GRAY)
        group = VGroup(t1, t2).arrange(DOWN, buff=0.3)

        self.play(Write(t1), run_time=1.2)
        self.play(FadeIn(t2, shift=UP * 0.2))
        self.wait(1.5)
        self.play(FadeOut(group))

    # ── 2. Input image → pixel grid ──────────────────────────

    def _input_image(self):
        header = Text("Input — Image as Tensor",
                      font_size=28, color=C_ACCENT, weight=BOLD
                      ).to_edge(UP, buff=0.4)
        self.play(Write(header))

        # Simulate a 8×8 pixel grid (stand-in for 224×224)
        grid_size = 8
        cell = 0.42
        pixels = VGroup()
        np.random.seed(42)
        for r in range(grid_size):
            for c in range(grid_size):
                # Simulate a fruit image: center is "orange-ish"
                dist = ((r - 3.5)**2 + (c - 3.5)**2) ** 0.5
                if dist < 2.5:
                    rgb = interpolate_color(ORANGE, YELLOW, np.random.rand() * 0.5)
                else:
                    rgb = interpolate_color(GREEN_E, DARK_BROWN,
                                            np.random.rand() * 0.3)
                sq = Square(side_length=cell, fill_color=rgb,
                            fill_opacity=1, stroke_width=0.5,
                            stroke_color=DARK_GREY)
                sq.move_to([c * cell - (grid_size / 2) * cell + cell / 2,
                             -r * cell + (grid_size / 2) * cell - cell / 2, 0])
                pixels.add(sq)

        pixels.move_to(LEFT * 3.8)
        label_img = Text("224×224\nRGB image", font_size=15,
                         color=C_GRAY).next_to(pixels, DOWN, buff=0.3)

        # Channel decomposition
        ch_labels = ["R", "G", "B"]
        ch_colors = [RED, GREEN, BLUE]
        channels = VGroup()
        for i, (ch, col) in enumerate(zip(ch_labels, ch_colors)):
            ch_grid = VGroup(*[
                Square(side_length=cell * 0.55,
                       fill_color=col,
                       fill_opacity=np.random.rand() * 0.7 + 0.2,
                       stroke_width=0.3, stroke_color=DARK_GREY)
                for _ in range(grid_size * grid_size)
            ])
            ch_grid.arrange_in_grid(rows=grid_size, cols=grid_size, buff=0.02)
            ch_box = VGroup(
                ch_grid,
                Text(ch, font_size=14, color=col).next_to(ch_grid, DOWN, buff=0.1)
            )
            channels.add(ch_box)

        channels.arrange(RIGHT, buff=0.25)
        channels.scale(0.7).move_to(RIGHT * 2.5)

        arr1 = Arrow(pixels.get_right(), channels.get_left(),
                     buff=0.2, color=C_GRAY, stroke_width=2)
        dim_label = Text("3 × 224 × 224", font_size=16, color=C_SIGNAL,
                         weight=BOLD).next_to(channels, DOWN, buff=0.3)

        self.play(FadeIn(pixels), FadeIn(label_img))
        self.wait(0.5)
        self.play(GrowArrow(arr1))
        self.play(LaggedStart(*[FadeIn(c, scale=0.9) for c in channels],
                              lag_ratio=0.3))
        self.play(Write(dim_label))
        self.wait(1.5)
        self.play(FadeOut(VGroup(header, pixels, label_img,
                                 arr1, channels, dim_label)))

    # ── 3. Backbone signal flow ───────────────────────────────

    def _backbone_flow(self):
        header = Text("ResNet-18 Backbone — Signal Flow",
                      font_size=26, color=C_ACCENT, weight=BOLD
                      ).to_edge(UP, buff=0.35)
        self.play(Write(header))

        # Define layers
        layer_specs = [
            (5, C_FROZEN,   "conv1"),
            (5, C_FROZEN,   "layer1"),
            (5, C_FROZEN,   "layer2"),
            (5, C_FROZEN,   "layer3"),
            (5, C_FINETUNE, "layer4\n(C2/C4)"),
            (3, C_HEAD,     "GAP\n512-d"),
        ]

        x_positions = np.linspace(-5.5, 3.5, len(layer_specs))

        layers = []
        for (n, col, lbl), x in zip(layer_specs, x_positions):
            lg = make_layer(n, col, label=lbl)
            lg.move_to([x, 0, 0])
            layers.append(lg)

        layers_vg = VGroup(*layers)

        # Connections (sparse, just visual)
        all_edges = VGroup()
        for i in range(len(layers) - 1):
            nodes_a = layers[i][0] if isinstance(layers[i][0], VGroup) \
                else VGroup(layers[i][0])
            nodes_b = layers[i+1][0] if isinstance(layers[i+1][0], VGroup) \
                else VGroup(layers[i+1][0])
            # Only connect a subset to avoid clutter
            try:
                a_nodes = layers[i][0]
                b_nodes = layers[i+1][0]
                edges = connect_layers(a_nodes, b_nodes,
                                       opacity=0.12, stroke_width=0.7)
                all_edges.add(edges)
            except Exception:
                pass

        self.play(FadeIn(all_edges))
        self.play(LaggedStart(
            *[FadeIn(lg, shift=UP * 0.15) for lg in layers],
            lag_ratio=0.15
        ))

        # Animate signal flowing left to right
        signal = signal_dot()
        signal.move_to(layers[0][0][2].get_center())

        path_points = [layers[i][0][2].get_center()
                       for i in range(len(layers))]
        self.play(FadeIn(signal))
        for i in range(1, len(path_points)):
            self.play(signal.animate.move_to(path_points[i]),
                      run_time=0.35, rate_func=smooth)

        # Show output feature vector
        feat_label = Text("512-dim\nfeature vector", font_size=14,
                          color=C_HEAD).next_to(layers[-1], RIGHT, buff=0.5)
        arr_feat = Arrow(layers[-1][0].get_right(),
                         feat_label.get_left(), buff=0.1,
                         color=C_HEAD, stroke_width=2)
        self.play(GrowArrow(arr_feat), Write(feat_label))

        # Legend
        legend = VGroup(
            VGroup(
                Dot(color=C_FROZEN, radius=0.12),
                Text(" Frozen — ImageNet weights preserved",
                     font_size=13, color=C_GRAY)
            ).arrange(RIGHT, buff=0.1),
            VGroup(
                Dot(color=C_FINETUNE, radius=0.12),
                Text(" Fine-tuned — adapts to freshness domain (C2/C4)",
                     font_size=13, color=C_GRAY)
            ).arrange(RIGHT, buff=0.1),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        legend.to_corner(DR, buff=0.4)
        self.play(FadeIn(legend))

        self.wait(2)
        self.play(FadeOut(VGroup(
            header, layers_vg, all_edges, signal,
            arr_feat, feat_label, legend
        )))

    # ── 4. Condition comparison ───────────────────────────────

    def _condition_comparison(self):
        header = Text("Training Conditions — C1 · C3 · C4",
                      font_size=26, color=C_ACCENT, weight=BOLD
                      ).to_edge(UP, buff=0.35)
        self.play(Write(header))

        def make_condition(label, desc, backbone_color, head_nodes,
                           pos, show_head=True):
            """Build a mini network diagram for one condition."""
            # Backbone (compressed)
            bb_nodes = VGroup(*[
                Circle(radius=0.13,
                       fill_color=backbone_color, fill_opacity=0.85,
                       stroke_color=WHITE, stroke_width=1.2)
                for _ in range(4)
            ]).arrange(DOWN, buff=0.22)
            bb_label = Text("Backbone", font_size=12,
                            color=backbone_color).next_to(bb_nodes, DOWN, buff=0.1)
            bb = VGroup(bb_nodes, bb_label)

            result = VGroup(bb)

            if show_head:
                # Projection head nodes
                head = VGroup(*[
                    Circle(radius=0.13,
                           fill_color=C_HEAD, fill_opacity=0.85,
                           stroke_color=WHITE, stroke_width=1.2)
                    for _ in range(head_nodes)
                ]).arrange(DOWN, buff=0.22)
                h_label = Text("Head", font_size=12,
                               color=C_HEAD).next_to(head, DOWN, buff=0.1)
                h_vg = VGroup(head, h_label)
                h_vg.next_to(bb, RIGHT, buff=0.5)
                arr = Arrow(bb_nodes.get_right(), head.get_left(),
                            buff=0.05, color=C_GRAY, stroke_width=1.5)
                result.add(arr, h_vg)
                last = head
            else:
                last = bb_nodes

            # Linear classifier
            cls_node = Circle(radius=0.13,
                              fill_color=C_LINEAR, fill_opacity=0.85,
                              stroke_color=WHITE, stroke_width=1.2)
            cls_label = Text("C", font_size=11,
                             color=C_LINEAR).move_to(cls_node)
            cls_vg = VGroup(cls_node, cls_label)
            cls_vg.next_to(last, RIGHT, buff=0.5)
            arr2 = Arrow(last.get_right(), cls_node.get_left(),
                         buff=0.05, color=C_GRAY, stroke_width=1.5)
            result.add(arr2, cls_vg)

            # Title and desc
            title = Text(label, font_size=18, color=C_WHITE,
                         weight=BOLD)
            desc_t = Text(desc, font_size=12, color=C_GRAY)
            title_g = VGroup(title, desc_t).arrange(DOWN, buff=0.05)

            # Frame
            frame = RoundedRectangle(
                width=result.get_width() + 0.5,
                height=result.get_height() + 0.8,
                corner_radius=0.2,
                fill_color="#1e1e2e", fill_opacity=0.6,
                stroke_color=C_GRAY, stroke_width=1
            ).move_to(result)

            title_g.next_to(frame, UP, buff=0.15)
            full = VGroup(frame, result, title_g)
            full.move_to(pos)
            return full

        c1 = make_condition("C1 — Linear Probe",
                            "frozen · no head",
                            C_FROZEN, 0,
                            LEFT * 4.2, show_head=False)
        c3 = make_condition("C3 — Frozen + Head",
                            "frozen · projection head",
                            C_FROZEN, 3,
                            ORIGIN)
        c4 = make_condition("C4 — Layer4 + Head",
                            "fine-tuned · projection head",
                            C_FINETUNE, 3,
                            RIGHT * 4.2)

        self.play(LaggedStart(
            FadeIn(c1, shift=UP * 0.2),
            FadeIn(c3, shift=UP * 0.2),
            FadeIn(c4, shift=UP * 0.2),
            lag_ratio=0.3
        ))

        # Highlight best conditions
        hl_c3 = SurroundingRectangle(c3, color=C_GREEN,
                                     stroke_width=2.5, buff=0.05)
        hl_c4 = SurroundingRectangle(c4, color=C_ORANGE,
                                     stroke_width=2.5, buff=0.05)
        lbl_c3 = Text("Best for EfficientNet",
                      font_size=13, color=C_GREEN
                      ).next_to(hl_c3, DOWN, buff=0.2)
        lbl_c4 = Text("Best for ResNet-18",
                      font_size=13, color=C_ORANGE
                      ).next_to(hl_c4, DOWN, buff=0.2)

        self.play(Create(hl_c3), Write(lbl_c3))
        self.play(Create(hl_c4), Write(lbl_c4))
        self.wait(2.5)
        self.play(FadeOut(VGroup(
            header, c1, c3, c4, hl_c3, hl_c4, lbl_c3, lbl_c4
        )))

    # ── 5. Projection head ────────────────────────────────────

    def _projection_head(self):
        header = Text("Projection Head — Dimensional Reduction",
                      font_size=26, color=C_ACCENT, weight=BOLD
                      ).to_edge(UP, buff=0.35)
        self.play(Write(header))

        # Layers: input(512) → fc1(256) → relu → fc2(128) → relu → cls(2)
        layer_sizes = [
            (8, "512", C_FROZEN,   "Backbone\noutput"),
            (6, "256", C_HEAD,     "FC(256)\n+ ReLU"),
            (5, "128", C_HEAD,     "FC(128)\n+ ReLU"),
            (2, "C",   C_LINEAR,   "Classifier\n(fresh/rotten)"),
        ]

        x_pos = [-4.5, -1.5, 1.5, 4.5]
        all_layers = []
        for (n, dim, col, lbl), x in zip(layer_sizes, x_pos):
            nodes = VGroup(*[
                Circle(radius=0.22,
                       fill_color=col, fill_opacity=0.85,
                       stroke_color=WHITE, stroke_width=1.5)
                for _ in range(n)
            ]).arrange(DOWN, buff=0.35)
            dim_t = Text(dim, font_size=14, color=col,
                         weight=BOLD).next_to(nodes, UP, buff=0.15)
            lbl_t = Text(lbl, font_size=13, color=C_GRAY
                         ).next_to(nodes, DOWN, buff=0.15)
            lg = VGroup(nodes, dim_t, lbl_t)
            lg.move_to([x, 0, 0])
            all_layers.append(lg)

        # Connections
        all_edges = VGroup()
        for i in range(len(all_layers) - 1):
            nodes_a = all_layers[i][0]
            nodes_b = all_layers[i+1][0]
            edges = connect_layers(nodes_a, nodes_b,
                                   opacity=0.18, stroke_width=1.0)
            all_edges.add(edges)

        self.play(FadeIn(all_edges))
        self.play(LaggedStart(
            *[FadeIn(lg, shift=RIGHT * 0.2) for lg in all_layers],
            lag_ratio=0.2
        ))

        # Animate signal through head
        signal = signal_dot()
        centers = [lg[0][len(lg[0])//2].get_center()
                   for lg in all_layers]
        signal.move_to(centers[0])
        self.play(FadeIn(signal))
        for pt in centers[1:]:
            self.play(signal.animate.move_to(pt),
                      run_time=0.5, rate_func=smooth)

        # Show dimension arrows
        dim_arrow = DoubleArrow(
            all_layers[0][0].get_right(),
            all_layers[-1][0].get_left(),
            buff=0.1, color=C_SIGNAL, stroke_width=1.5
        ).shift(UP * 2.2)
        dim_label = Text("512 → 256 → 128 → C",
                         font_size=16, color=C_SIGNAL, weight=BOLD
                         ).next_to(dim_arrow, UP, buff=0.1)

        self.play(GrowFromCenter(dim_arrow), Write(dim_label))
        self.wait(2)
        self.play(FadeOut(VGroup(
            header, all_edges,
            VGroup(*all_layers), signal, dim_arrow, dim_label
        )))

    # ── 6. Softmax output ─────────────────────────────────────

    def _softmax_output(self):
        header = Text("Classification — Softmax Output",
                      font_size=26, color=C_ACCENT, weight=BOLD
                      ).to_edge(UP, buff=0.35)
        self.play(Write(header))

        # Logits
        logit_label = Text("Logits (raw scores):",
                           font_size=18, color=C_GRAY)
        logit_vals = Text("[  2.31 ,  -0.84  ]",
                          font_size=22, color=C_WHITE, weight=BOLD)
        logit_classes = Text("  fresh      rotten",
                             font_size=16, color=C_GRAY)
        logits_g = VGroup(logit_label,
                          VGroup(logit_vals, logit_classes).arrange(DOWN,
                                                                     buff=0.05)
                          ).arrange(DOWN, buff=0.2).shift(UP * 1.5 + LEFT * 3)

        self.play(FadeIn(logits_g))

        # Softmax formula
        formula = MathTex(
            r"P(c) = \frac{e^{z_c}}{\sum_k e^{z_k}}",
            font_size=32, color=C_WHITE
        ).shift(UP * 1.5 + RIGHT * 1)
        self.play(Write(formula))

        # Arrow down
        arr_down = Arrow(ORIGIN + UP * 0.3, ORIGIN + DOWN * 0.3,
                         color=C_GRAY, stroke_width=2)

        # Result probabilities
        result_label = Text("Probabilities:", font_size=18,
                            color=C_GRAY).shift(DOWN * 0.3 + LEFT * 3)
        result_vals = Text("[  95.7% ,   4.3%  ]",
                           font_size=22, color=C_WHITE,
                           weight=BOLD).shift(DOWN * 0.75 + LEFT * 3)
        self.play(GrowArrow(arr_down), FadeIn(result_label),
                  Write(result_vals))

        # Confidence bars
        bar_data = [("fresh",  0.957, C_FRESH), ("rotten", 0.043, C_ROTTEN)]
        bars = VGroup()
        for cls_name, prob, col in bar_data:
            lbl = Text(cls_name, font_size=16,
                       color=C_WHITE).set_width(1.0)
            bg = Rectangle(width=3.5, height=0.4,
                           fill_color="#2a2a3e", fill_opacity=1,
                           stroke_width=0)
            fg = Rectangle(width=3.5 * prob, height=0.4,
                           fill_color=col, fill_opacity=0.9,
                           stroke_width=0).align_to(bg, LEFT)
            pct = Text(f"{prob*100:.1f}%", font_size=16,
                       color=col, weight=BOLD)
            row = VGroup(lbl, bg, pct).arrange(RIGHT, buff=0.25)
            fg.move_to(bg).align_to(bg, LEFT)
            bars.add(VGroup(row, fg))

        bars.arrange(DOWN, buff=0.35).shift(DOWN * 2.2 + RIGHT * 0.5)

        for bar in bars:
            row, fg = bar
            self.play(FadeIn(row), GrowFromEdge(fg, LEFT), run_time=0.8)

        # Final prediction
        pred = VGroup(
            RoundedRectangle(width=3.5, height=0.8, corner_radius=0.15,
                             fill_color=C_FRESH, fill_opacity=0.2,
                             stroke_color=C_FRESH, stroke_width=2.5),
            Text("FRESH  ✓  95.7% confidence",
                 font_size=18, color=C_FRESH, weight=BOLD)
        )
        pred[1].move_to(pred[0])
        pred.next_to(bars, RIGHT, buff=0.7)

        self.play(FadeIn(pred, scale=0.8))
        self.wait(2.5)
        self.play(FadeOut(VGroup(
            header, logits_g, formula, arr_down,
            result_label, result_vals, bars, pred
        )))

    # ── 7. Ablation summary ───────────────────────────────────

    def _ablation_summary(self):
        header = Text("Ablation Study — C1 to C4 Performance",
                      font_size=26, color=C_ACCENT, weight=BOLD
                      ).to_edge(UP, buff=0.35)
        self.play(Write(header))

        # Bar chart of F1 scores per condition (ResNet-18, MFV dataset)
        data = [
            ("C1\nfrozen",       78.3, C_FROZEN),
            ("C2\nlayer4",       91.0, C_FINETUNE),
            ("C3\nhead_frozen",  86.7, C_HEAD),
            ("C4\nhead_layer4",  90.6, C_LINEAR),
        ]

        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[60, 100, 10],
            x_length=8,
            y_length=4,
            axis_config={"color": C_GRAY, "stroke_width": 1.5},
            y_axis_config={"include_numbers": True,
                           "numbers_to_include": [60, 70, 80, 90, 100],
                           "font_size": 14},
        ).shift(DOWN * 0.3)

        y_label = Text("Macro F1 (%)", font_size=15,
                       color=C_GRAY).rotate(PI / 2).next_to(
            axes.y_axis, LEFT, buff=0.3)
        title_chart = Text("FruitVision Dataset (3-class, formalin)",
                           font_size=14, color=C_GRAY
                           ).next_to(axes, DOWN, buff=0.3)

        self.play(Create(axes), Write(y_label), Write(title_chart))

        bar_width = 0.55
        x_starts = [0.7, 1.7, 2.7, 3.7]

        for (label, f1, col), x in zip(data, x_starts):
            bar = axes.get_riemann_rectangles(
                axes.plot(lambda t: f1, x_range=[x, x + bar_width]),
                x_range=[x, x + bar_width],
                dx=bar_width,
                color=col,
                fill_opacity=0.85,
                stroke_width=0,
            )
            val_text = Text(f"{f1}%", font_size=14, color=col,
                            weight=BOLD).next_to(
                axes.c2p(x + bar_width / 2, f1), UP, buff=0.1)
            cond_text = Text(label, font_size=13,
                             color=C_GRAY).next_to(
                axes.c2p(x + bar_width / 2, 60), DOWN, buff=0.15)

            self.play(
                GrowFromEdge(bar, DOWN),
                Write(val_text),
                FadeIn(cond_text),
                run_time=0.7
            )

        self.wait(2)

        # Final message
        msg = Text(
            "C4 achieves best F1 for ResNet-18\n"
            "C3 optimal for EfficientNet (1280-dim features)",
            font_size=16, color=C_WHITE, line_spacing=1.3
        ).to_edge(DOWN, buff=0.4)
        self.play(Write(msg))
        self.wait(2.5)
        self.play(FadeOut(Group(*self.mobjects)))

        # End card
        end = VGroup(
            Text("End", font_size=52, color=C_WHITE, weight=BOLD),
            Text("Freshness Classification · Transfer Learning",
                 font_size=18, color=C_GRAY)
        ).arrange(DOWN, buff=0.3)
        self.play(FadeIn(end))
        self.wait(2)
        self.play(FadeOut(end))