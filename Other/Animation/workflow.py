"""
workflow.py — Animated pipeline for Freshness Classification project.

Render with:
    manim -pql workflow.py WorkflowScene       # low quality preview
    manim -pqh workflow.py WorkflowScene       # high quality
    manim -pqk workflow.py WorkflowScene       # 4K

Scenes available:
    WorkflowScene   — full pipeline (recommended)
"""

from manim import *


# ── Color palette ──────────────────────────────────────────────
C_BG        = "#1a1a2e"
C_ACCENT    = "#e94560"
C_BLUE      = "#4C72B0"
C_GREEN     = "#55A868"
C_ORANGE    = "#DD8452"
C_GRAY      = "#aaaaaa"
C_WHITE     = "#f0f0f0"
C_FRESH     = "#2ecc71"
C_ROTTEN    = "#e74c3c"
C_FORMALIN  = "#f39c12"


# ── Helper: styled box ─────────────────────────────────────────
def styled_box(text, width=2.8, height=0.9, color=C_BLUE,
               text_color=C_WHITE, font_size=20):
    rect = RoundedRectangle(
        width=width, height=height, corner_radius=0.15,
        fill_color=color, fill_opacity=0.9,
        stroke_color=color, stroke_width=2
    )
    label = Text(text, font_size=font_size, color=text_color,
                 weight=BOLD).move_to(rect)
    return VGroup(rect, label)


def arrow(start, end, color=C_GRAY):
    return Arrow(start, end, buff=0.1, color=color,
                 stroke_width=2, max_tip_length_to_length_ratio=0.15)


# ──────────────────────────────────────────────────────────────
class WorkflowScene(Scene):
    """
    Full pipeline animation:
      1. Title
      2. Dataset — raw images organized by fruit/state
      3. Preprocessing — resize, augment, normalize
      4. Backbone — CNN feature extraction
      5. Training condition — C3 projection head example
      6. Classification output — confidence prediction
    """

    def construct(self):
        self.camera.background_color = C_BG
        self._title()
        self._dataset()
        self._preprocessing()
        self._backbone()
        self._condition()
        self._output()
        self._summary()

    # ── 1. Title ──────────────────────────────────────────────

    def _title(self):
        title = Text(
            "Freshness Classification Pipeline",
            font_size=40, color=C_WHITE, weight=BOLD
        )
        subtitle = Text(
            "Transfer Learning · ResNet-18 · EfficientNet-B0/B2 · MobileNetV3",
            font_size=20, color=C_GRAY
        ).next_to(title, DOWN, buff=0.3)

        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(subtitle, shift=UP * 0.3))
        self.wait(1.5)
        self.play(FadeOut(title), FadeOut(subtitle))

    # ── 2. Dataset ────────────────────────────────────────────

    def _dataset(self):
        header = Text("Stage 01 — Dataset", font_size=30,
                      color=C_ACCENT, weight=BOLD).to_edge(UP, buff=0.4)
        self.play(Write(header))

        # Folder structure
        folders = VGroup()
        datasets = [
            ("KFR", "13,599 imgs"),
            ("KFS", "27,317 imgs"),
            ("MLM", " 1,956 imgs"),
            ("MFR", " 1,655 imgs"),
            ("MFV", "10,154 imgs"),
            ("KFQ", "   359 imgs"),
        ]
        colors = [C_BLUE, C_ORANGE, C_GREEN, C_BLUE, C_ACCENT, C_ORANGE]

        for i, ((code, count), col) in enumerate(zip(datasets, colors)):
            box = RoundedRectangle(
                width=2.0, height=0.7, corner_radius=0.1,
                fill_color=col, fill_opacity=0.8,
                stroke_color=col, stroke_width=1.5
            )
            t1 = Text(code,  font_size=18, color=C_WHITE, weight=BOLD)
            t2 = Text(count, font_size=13, color=C_WHITE)
            VGroup(t1, t2).arrange(DOWN, buff=0.05).move_to(box)
            folders.add(VGroup(box, t1, t2))

        folders.arrange_in_grid(rows=2, cols=3, buff=0.3)
        folders.move_to(ORIGIN + LEFT * 2.5)

        self.play(LaggedStart(
            *[FadeIn(f, shift=UP * 0.2) for f in folders],
            lag_ratio=0.12
        ))

        # Arrow to structure diagram
        arr = arrow(folders.get_right(), folders.get_right() + RIGHT * 0.4)

        # Nested structure
        struct = VGroup(
            Text("data/", font_size=16, color=C_GREEN),
            Text("  └─ apple/", font_size=14, color=C_GRAY),
            Text("       ├─ fresh/", font_size=14, color=C_FRESH),
            Text("       ├─ rotten/", font_size=14, color=C_ROTTEN),
            Text("       └─ formalin/", font_size=14, color=C_FORMALIN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        struct.next_to(folders, RIGHT, buff=0.7)

        self.play(GrowArrow(arr))
        self.play(Write(struct), run_time=1.2)
        self.wait(1.5)
        self.play(FadeOut(VGroup(header, folders, arr, struct)))

    # ── 3. Preprocessing ──────────────────────────────────────

    def _preprocessing(self):
        header = Text("Stage 01 — Preprocessing", font_size=30,
                      color=C_ACCENT, weight=BOLD).to_edge(UP, buff=0.4)
        self.play(Write(header))

        steps = [
            ("Raw Image\n224×224", C_GRAY),
            ("Resize\n256 px", C_BLUE),
            ("Augment\ncrop·flip·rotate\ncolor·blur", C_ORANGE),
            ("Normalize\nμ=[0.485…]\nσ=[0.229…]", C_GREEN),
            ("Tensor\n3×224×224", C_BLUE),
        ]

        boxes = VGroup()
        for text, col in steps:
            boxes.add(styled_box(text, width=2.2, height=1.1,
                                 color=col, font_size=16))
        boxes.arrange(RIGHT, buff=0.5).move_to(ORIGIN + DOWN * 0.3)

        arrows = VGroup()
        for i in range(len(boxes) - 1):
            arrows.add(arrow(
                boxes[i].get_right(),
                boxes[i + 1].get_left()
            ))

        note = Text(
            "Val: only resize + center crop (no augmentation)",
            font_size=15, color=C_GRAY
        ).next_to(boxes, DOWN, buff=0.5)

        self.play(LaggedStart(
            *[FadeIn(b, shift=RIGHT * 0.3) for b in boxes],
            lag_ratio=0.2
        ))
        self.play(LaggedStart(
            *[GrowArrow(a) for a in arrows],
            lag_ratio=0.15
        ))
        self.play(FadeIn(note))
        self.wait(2)
        self.play(FadeOut(VGroup(header, boxes, arrows, note)))

    # ── 4. Backbone ───────────────────────────────────────────

    def _backbone(self):
        header = Text("Stage 02 — Backbone (Feature Extraction)",
                      font_size=28, color=C_ACCENT, weight=BOLD
                      ).to_edge(UP, buff=0.4)
        self.play(Write(header))

        # Input tensor
        inp = styled_box("Input\n3×224×224", width=2.0, height=1.0,
                         color=C_GRAY)
        inp.move_to(LEFT * 5.5)

        # Backbone layers
        layer_data = [
            ("conv1\n+ pool", C_BLUE,   0.7),
            ("layer1\n(frozen)", C_BLUE, 0.7),
            ("layer2\n(frozen)", C_BLUE, 0.7),
            ("layer3\n(frozen)", C_BLUE, 0.7),
            ("layer4\n★ C2/C4", C_ORANGE, 0.7),
            ("GAP", C_GREEN,    0.5),
        ]
        layers = VGroup()
        for txt, col, w in layer_data:
            layers.add(styled_box(txt, width=w * 1.6, height=0.85,
                                  color=col, font_size=14))
        layers.arrange(RIGHT, buff=0.25)
        layers.next_to(inp, RIGHT, buff=0.5)

        # Output vector
        out = styled_box("Feature\nvector\n512-dim", width=1.6, height=1.1,
                         color=C_GREEN)
        out.next_to(layers, RIGHT, buff=0.5)

        # Arrows
        arr_in = arrow(inp.get_right(), layers[0].get_left())
        arr_out = arrow(layers[-1].get_right(), out.get_left())
        layer_arrows = VGroup(*[
            arrow(layers[i].get_right(), layers[i + 1].get_left())
            for i in range(len(layers) - 1)
        ])

        # Legend
        legend = VGroup(
            VGroup(
                Square(side_length=0.25, fill_color=C_BLUE,
                       fill_opacity=0.9, stroke_width=0),
                Text(" Frozen layers", font_size=14, color=C_GRAY)
            ).arrange(RIGHT, buff=0.1),
            VGroup(
                Square(side_length=0.25, fill_color=C_ORANGE,
                       fill_opacity=0.9, stroke_width=0),
                Text(" Fine-tuned (C2/C4)", font_size=14, color=C_GRAY)
            ).arrange(RIGHT, buff=0.1),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        legend.to_corner(DR, buff=0.5)

        self.play(FadeIn(inp))
        self.play(GrowArrow(arr_in))
        self.play(LaggedStart(
            *[FadeIn(l, shift=RIGHT * 0.2) for l in layers],
            lag_ratio=0.15
        ))
        self.play(LaggedStart(
            *[GrowArrow(a) for a in layer_arrows],
            lag_ratio=0.1
        ))
        self.play(GrowArrow(arr_out), FadeIn(out))
        self.play(FadeIn(legend))
        self.wait(2)
        self.play(FadeOut(VGroup(
            header, inp, layers, layer_arrows, arr_in, arr_out, out, legend
        )))

    # ── 5. Training condition ─────────────────────────────────

    def _condition(self):
        header = Text("Stage 02 — Training Conditions (C1–C4)",
                      font_size=28, color=C_ACCENT, weight=BOLD
                      ).to_edge(UP, buff=0.4)
        self.play(Write(header))

        # 2×2 grid of conditions
        cond_data = [
            ("C1", "Frozen + Linear", "Linear probe\nno adaptation", C_GRAY),
            ("C2", "Layer4 + Linear", "Partial fine-tuning\nno head", C_BLUE),
            ("C3", "Frozen + Head ★", "Best for EfficientNet\nprojection head", C_GREEN),
            ("C4", "Layer4 + Head ★", "Best for ResNet-18\nfull adaptation", C_ORANGE),
        ]

        boxes = VGroup()
        for code, name, desc, col in cond_data:
            outer = RoundedRectangle(
                width=3.0, height=1.4, corner_radius=0.15,
                fill_color=col, fill_opacity=0.15,
                stroke_color=col, stroke_width=2
            )
            t_code = Text(code, font_size=26, color=col, weight=BOLD)
            t_name = Text(name, font_size=15, color=C_WHITE)
            t_desc = Text(desc, font_size=12, color=C_GRAY)
            VGroup(t_code, t_name, t_desc).arrange(
                DOWN, buff=0.08).move_to(outer)
            boxes.add(VGroup(outer, t_code, t_name, t_desc))

        boxes.arrange_in_grid(rows=2, cols=2, buff=0.4)
        boxes.move_to(ORIGIN + DOWN * 0.3)

        self.play(LaggedStart(
            *[FadeIn(b, scale=0.8) for b in boxes],
            lag_ratio=0.2
        ))

        # Highlight C3 as example
        highlight = SurroundingRectangle(
            boxes[2], color=C_GREEN, stroke_width=3, buff=0.08
        )
        label = Text("← example used in animation",
                     font_size=14, color=C_GREEN
                     ).next_to(highlight, RIGHT, buff=0.2)
        self.play(Create(highlight), Write(label))
        self.wait(2)
        self.play(FadeOut(VGroup(header, boxes, highlight, label)))

    # ── 6. Output ─────────────────────────────────────────────

    def _output(self):
        header = Text("Stage 03 — Classification Output",
                      font_size=28, color=C_ACCENT, weight=BOLD
                      ).to_edge(UP, buff=0.4)
        self.play(Write(header))

        # Pipeline summary: feature → head → softmax → prediction
        feat   = styled_box("Feature\nvector\n512-dim", color=C_GREEN,
                            width=1.8, height=1.0)
        head   = styled_box("Projection\nHead\n512→256→128", color=C_ORANGE,
                            width=2.0, height=1.0)
        cls    = styled_box("Classifier\n128→C", color=C_BLUE,
                            width=1.6, height=1.0)
        softm  = styled_box("Softmax", color=C_GRAY,
                            width=1.4, height=0.8)

        pipeline = VGroup(feat, head, cls, softm).arrange(
            RIGHT, buff=0.6).shift(UP * 0.8)

        p_arrows = VGroup(*[
            arrow(pipeline[i].get_right(), pipeline[i + 1].get_left())
            for i in range(len(pipeline) - 1)
        ])

        self.play(
            LaggedStart(*[FadeIn(p, shift=RIGHT * 0.2) for p in pipeline],
                        lag_ratio=0.2),
        )
        self.play(LaggedStart(*[GrowArrow(a) for a in p_arrows],
                              lag_ratio=0.15))

        # Confidence bars
        classes = [
            ("fresh",   0.957, C_FRESH),
            ("rotten",  0.043, C_ROTTEN),
        ]

        bars_group = VGroup()
        for name, prob, col in classes:
            label = Text(f"{name}", font_size=16,
                         color=C_WHITE).set_width(1.0)
            bar_bg = Rectangle(width=4.0, height=0.35,
                               fill_color="#333333", fill_opacity=1,
                               stroke_width=0)
            bar_fg = Rectangle(width=4.0 * prob, height=0.35,
                               fill_color=col, fill_opacity=0.9,
                               stroke_width=0).align_to(bar_bg, LEFT)
            pct = Text(f"{prob*100:.1f}%", font_size=16,
                       color=col, weight=BOLD)
            row = VGroup(label, bar_bg, pct).arrange(RIGHT, buff=0.2)
            bar_fg.move_to(bar_bg).align_to(bar_bg, LEFT)
            bars_group.add(VGroup(row, bar_fg))

        bars_group.arrange(DOWN, buff=0.35).shift(DOWN * 1.5)

        arr_to_bars = arrow(softm.get_bottom(),
                            bars_group.get_top() + UP * 0.1)

        self.play(GrowArrow(arr_to_bars))

        for bar_row in bars_group:
            row, bar_fg = bar_row
            self.play(FadeIn(row), GrowFromEdge(bar_fg, LEFT),
                      run_time=0.7)

        # Final prediction box
        pred_box = styled_box("✓  FRESH  (95.7%)", width=3.2, height=0.8,
                              color=C_FRESH, font_size=22)
        pred_box.next_to(bars_group, RIGHT, buff=0.8)
        arr_pred = arrow(bars_group.get_right(), pred_box.get_left())

        self.play(GrowArrow(arr_pred), FadeIn(pred_box, scale=0.8))
        self.wait(2)
        self.play(FadeOut(VGroup(
            header, pipeline, p_arrows, bars_group, arr_to_bars,
            arr_pred, pred_box
        )))

    # ── 7. Summary ────────────────────────────────────────────

    def _summary(self):
        title = Text("Full Pipeline Summary",
                     font_size=36, color=C_WHITE, weight=BOLD
                     ).to_edge(UP, buff=0.4)
        self.play(Write(title))

        stages = [
            ("01", "Data Preparation",   "6 datasets · fruit/state/ structure",     C_BLUE),
            ("02", "Training",           "4 backbones · 4 conditions · 60 epochs",  C_ORANGE),
            ("03", "Evaluation",         "Cross-dataset · 121 real photos",          C_GREEN),
            ("04", "Reporting",          "Excel tracker · eval report · LaTeX",      C_ACCENT),
        ]

        rows = VGroup()
        for num, stage, desc, col in stages:
            n  = Text(num, font_size=28, color=col, weight=BOLD).set_width(0.5)
            s  = Text(stage, font_size=20, color=C_WHITE, weight=BOLD).set_width(2.8)
            d  = Text(desc,  font_size=16, color=C_GRAY).set_width(4.5)
            row = VGroup(n, s, d).arrange(RIGHT, buff=0.4)
            rows.add(row)

        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        rows.move_to(ORIGIN + DOWN * 0.3)

        self.play(LaggedStart(
            *[FadeIn(r, shift=LEFT * 0.3) for r in rows],
            lag_ratio=0.25
        ))

        # Final result callout
        result = Text(
            "Best: KFR-EB0-C3-ST → 100.0% F1",
            font_size=22, color=C_GREEN, weight=BOLD
        ).to_edge(DOWN, buff=0.5)
        self.play(Write(result))
        self.wait(3)
        self.play(FadeOut(VGroup(title, rows, result)))

        # End card
        end = Text("Thank you", font_size=50,
                   color=C_WHITE, weight=BOLD)
        self.play(Write(end))
        self.wait(2)
        self.play(FadeOut(end))